"""LangGraph node implementations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.grounding import (
    build_facts_table,
    build_footer,
    substitute_placeholders,
    template_fallback_reply,
    verify_composed_text,
)
from src.prompts import COMPOSE_SYSTEM, COMPOSE_USER, FUZZY_JUDGE_SYSTEM, FUZZY_JUDGE_USER, UNDERSTAND_SYSTEM, UNDERSTAND_USER
from src.sops import (
    SopModel,
    apply_supersedes,
    collect_metrics,
    evaluate_sops_for_request,
    load_sops,
    rank_fired,
    sop_matches_tags,
    evaluate_threshold_at_location,
    FuzzySop,
    ThresholdSop,
)
from src.weather import geocode_search, fetch_forecast


class UnderstandOut(BaseModel):
    in_scope: bool
    tags: list[str] = Field(default_factory=list)
    locations: list[dict[str, Any]] = Field(default_factory=list)
    country_code: str | None = None
    when: dict[str, Any] = Field(default_factory=lambda: {"day_offset": 0, "part": "all_day"})
    needs_location: bool = False


class ComposeOut(BaseModel):
    text: str


class FuzzyJudgeOut(BaseModel):
    verdict: str
    rubric_items_used: list[str] = Field(default_factory=list)


def _cap_messages(messages: list[Any], n: int = 6) -> list[Any]:
    return messages[-n:] if messages else []


def _cap_decision_log(log: list[dict], n: int = 5) -> list[dict]:
    return log[-n:] if log else []


def _format_messages(messages: list[Any]) -> str:
    lines = []
    for m in messages[-6:]:
        role = getattr(m, "type", "human")
        content = getattr(m, "content", str(m))
        lines.append(f"{role}: {content[:400]}")
    return "\n".join(lines) or "(none)"


def make_understand_node(llm, sops_dir: str):
    async def understand(state: dict[str, Any]) -> dict[str, Any]:
        sops, skipped, tag_vocab = load_sops(sops_dir)
        user_msg = state.get("user_message", "")

        reset = {
            "error": None,
            "no_sop_reason": None,
            "resolved_locations": [],
            "failed_locations": [],
            "weather_by_location": {},
            "facts": [],
            "fired": [],
            "fuzzy_candidates": [],
            "checked_sop_ids": [],
            "unknown": [],
            "compose_text": None,
            "verify_error": None,
            "retries": 0,
            "reply_body": None,
            "trace": {},
            "fetched_at": None,
            "understand": None,
            "composer_calls_this_turn": 0,
        }

        structured = llm.with_structured_output(UnderstandOut)
        prompt_tags = tag_vocab
        out: UnderstandOut = await structured.ainvoke(
            [
                {"role": "system", "content": UNDERSTAND_SYSTEM},
                {
                    "role": "user",
                    "content": UNDERSTAND_USER.format(
                        tags=", ".join(prompt_tags) or "(none)",
                        messages=_format_messages(state.get("messages") or []),
                        decision_log=json.dumps(_cap_decision_log(state.get("decision_log") or []), indent=2),
                        message=user_msg[:500],
                    ),
                },
            ]
        )

        # Filter tags to vocabulary
        valid_tags = [t for t in out.tags if t in prompt_tags]
        out.tags = valid_tags

        # Session inherit location if follow-up
        if out.needs_location and state.get("decision_log"):
            last = state["decision_log"][-1]
            if last.get("locations") and not out.locations:
                out.locations = last["locations"]
                out.needs_location = False
            if last.get("when") and out.when.get("part") == "all_day":
                out.when = last.get("when", out.when)

        return {
            **reset,
            "sops": sops,
            "skipped_sops": skipped,
            "tag_vocab": prompt_tags,
            "understand": out.model_dump(),
            "in_scope": out.in_scope,
            "tags": out.tags,
            "locations": out.locations,
            "country_code": out.country_code,
            "user_when": out.when,
            "needs_location": out.needs_location,
        }

    return understand


def ask_clarification(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "reply_body": "Which city or area should I check the weather for?",
        "no_sop_reason": "location_unresolved",
    }


def no_sop_node(state: dict[str, Any]) -> dict[str, Any]:
    reason = state.get("no_sop_reason") or "no_topic"
    if reason == "no_topic":
        body = "I only help with weather-related safety questions tied to our playbooks. I can't answer that."
    elif reason == "no_trigger":
        checked = state.get("checked_sop_ids") or []
        body = (
            "I checked the relevant playbooks but none triggered for these conditions. "
            f"Checked: {', '.join(checked) if checked else 'none'}. "
            "That is not a guarantee of safety."
        )
    elif reason == "weather_unavailable":
        body = "I couldn't retrieve a forecast right now, so I can't give weather-based guidance."
    elif reason == "location_unresolved":
        body = "I couldn't resolve that location, so I can't fetch a forecast."
    else:
        body = "I don't have guidance for that."
    return {"reply_body": body}


async def resolve_locations(state: dict[str, Any], geocode=geocode_search) -> dict[str, Any]:
    locs = state.get("locations") or []
    cc = state.get("country_code")
    resolved: list[dict[str, Any]] = []
    failed: list[str] = []

    for loc in locs[:5]:
        name = loc.get("name") or ""
        if not name.strip():
            continue
        try:
            results = await geocode(name, country_code=cc, count=5)
            if not results:
                failed.append(name)
                continue
            first = results[0]
            display = first.name
            if first.admin1:
                display = f"{first.name}, {first.admin1}"
            resolved.append(
                {
                    "name": first.name,
                    "query_name": name,
                    "display_name": display,
                    "latitude": first.latitude,
                    "longitude": first.longitude,
                    "country_code": first.country_code,
                    "admin1": first.admin1,
                    "role": loc.get("role", "here"),
                    "inferred": bool(loc.get("inferred")),
                }
            )
        except Exception:
            failed.append(name)

    return {"resolved_locations": resolved, "failed_locations": failed}


async def fetch_weather_node(state: dict[str, Any], fetch_forecast_fn=fetch_forecast) -> dict[str, Any]:
    sops: list[SopModel] = state.get("sops") or []
    metrics = sorted(collect_metrics(sops))
    weather: dict[str, dict[str, Any]] = {}
    failed: list[str] = list(state.get("failed_locations") or [])

    for loc in state.get("resolved_locations") or []:
        key = loc["query_name"]
        try:
            payload = await fetch_forecast_fn(loc["latitude"], loc["longitude"], metrics)
            weather[key] = payload.model_dump()
        except Exception:
            failed.append(key)

    return {
        "weather_by_location": weather,
        "failed_locations": failed,
        "fetched_at": datetime.now(timezone.utc),
    }


def evaluate_sops_node(state: dict[str, Any]) -> dict[str, Any]:
    sops: list[SopModel] = state.get("sops") or []
    fired, fuzzy, checked, unknown = evaluate_sops_for_request(
        sops,
        state.get("tags") or [],
        state.get("weather_by_location") or {},
        state.get("user_when"),
    )

    # Fuzzy tag match filter
    fuzzy = [
        f
        for f in fuzzy
        if (s := _sop_by_id(sops, f["id"])) and sop_matches_tags(s, state.get("tags") or [])
    ]

    hourly_units: dict[str, str] = {}
    for p in (state.get("weather_by_location") or {}).values():
        hourly_units = p.get("hourly_units") or {}
        break

    facts = build_facts_table(
        fired,
        state.get("weather_by_location") or {},
        state.get("resolved_locations") or [],
        state.get("user_when"),
        hourly_units,
    )

    return {
        "fired": fired,
        "fuzzy_candidates": fuzzy,
        "checked_sop_ids": checked,
        "unknown": unknown,
        "facts": facts,
    }


def _sop_by_id(sops: list[SopModel], sid: str) -> SopModel | None:
    for s in sops:
        if s.id == sid:
            return s
    return None


def make_judge_fuzzy_node(llm):
    async def judge_fuzzy(state: dict[str, Any]) -> dict[str, Any]:
        candidates = state.get("fuzzy_candidates") or []
        if not candidates:
            return {}
        sop_meta = candidates[0]
        snapshot = _metrics_snapshot(state, sop_meta.get("metrics") or [])
        structured = llm.with_structured_output(FuzzyJudgeOut)
        out: FuzzyJudgeOut = await structured.ainvoke(
            [
                {"role": "system", "content": FUZZY_JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": FUZZY_JUDGE_USER.format(
                        sop_id=sop_meta["id"],
                        title=sop_meta["title"],
                        rubric="\n".join(f"- {r}" for r in sop_meta.get("rubric") or []),
                        metrics_snapshot=json.dumps(snapshot, indent=2),
                        verdict_keys=", ".join(sop_meta.get("verdicts", {}).keys()),
                        message=state.get("user_message", "")[:500],
                    ),
                },
            ]
        )

        verdict_key = out.verdict if out.verdict in sop_meta.get("verdicts", {}) else "mixed"
        verdict_key = _apply_vetoes(verdict_key, sop_meta, state)

        v = sop_meta["verdicts"][verdict_key]
        fired = list(state.get("fired") or [])
        fired.append(
            {
                "id": sop_meta["id"],
                "title": sop_meta["title"],
                "severity": v["severity"],
                "guidance": v["guidance"],
                "tags": sop_meta.get("tags", []),
                "supersedes": sop_meta.get("supersedes", []),
                "location": (state.get("resolved_locations") or [{}])[0].get("display_name"),
                "evidence": [],
                "fuzzy_verdict": verdict_key,
            }
        )
        fired = apply_supersedes(fired)
        fired = rank_fired(fired)[:3]

        return {
            "fired": fired,
            "fuzzy_verdict": verdict_key,
            "composer_calls_this_turn": state.get("composer_calls_this_turn", 0) + 1,
        }

    return judge_fuzzy


def _metrics_snapshot(state: dict[str, Any], metrics: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for loc in state.get("resolved_locations") or []:
        payload = (state.get("weather_by_location") or {}).get(loc["query_name"])
        if not payload:
            continue
        hourly = payload.get("hourly") or {}
        snap: dict[str, Any] = {}
        for m in metrics:
            series = hourly.get(m)
            if series:
                snap[m] = {"min": min(series), "max": max(series), "first": series[0]}
        out[loc["display_name"]] = snap
    return out


def _apply_vetoes(verdict: str, sop_meta: dict, state: dict[str, Any]) -> str:
    sops: list[SopModel] = state.get("sops") or []
    sop = _sop_by_id(sops, sop_meta["id"])
    if not isinstance(sop, FuzzySop):
        return verdict

    for veto in sop.vetoes:
        if veto.forbid != verdict:
            continue
        when = veto.when
        if _veto_condition(when, state):
            # downgrade good -> mixed -> poor
            if verdict == "good":
                verdict = "mixed"
            elif verdict == "mixed":
                verdict = "poor"
    return verdict


def _veto_condition(when: dict[str, Any], state: dict[str, Any]) -> bool:
    """Evaluate simple veto conditions against weather."""
    from src.sops import ConditionLeaf, _eval_when_node

    unknown: list[str] = []
    for payload in (state.get("weather_by_location") or {}).values():
        ok, _ = _eval_when_node(when, payload, state.get("user_when"), unknown)
        if ok:
            return True
    return False


def make_compose_node(llm):
    async def compose(state: dict[str, Any]) -> dict[str, Any]:
        fired = state.get("fired") or []
        facts = state.get("facts") or []
        structured = llm.with_structured_output(ComposeOut)
        sops_txt = "\n".join(
            f"- {f['title']} ({f['severity']}): {f.get('guidance','')}" for f in fired
        )
        facts_txt = "\n".join(
            f"{f['id']}: {f['label']} @ {f['location']} [{f['window']}] = {f['value']} {f.get('unit','')}"
            for f in facts
        )
        extra = ""
        if state.get("verify_error"):
            extra = f"\nPrevious draft failed verification: {state['verify_error']}. Fix it."

        out: ComposeOut = await structured.ainvoke(
            [
                {"role": "system", "content": COMPOSE_SYSTEM + extra},
                {
                    "role": "user",
                    "content": COMPOSE_USER.format(
                        facts=facts_txt or "(none)",
                        sops=sops_txt or "(none)",
                        decision_log=json.dumps(_cap_decision_log(state.get("decision_log") or []), indent=2),
                        messages=_format_messages(state.get("messages") or []),
                        message=state.get("user_message", "")[:500],
                    ),
                },
            ]
        )
        return {
            "compose_text": out.text,
            "composer_calls_this_turn": state.get("composer_calls_this_turn", 0) + 1,
        }

    return compose


def verify_node(state: dict[str, Any]) -> dict[str, Any]:
    text = state.get("compose_text") or ""
    fact_ids = {f["id"] for f in state.get("facts") or []}
    err = verify_composed_text(text, fact_ids)
    return {"verify_error": err}


def template_fallback_node(state: dict[str, Any]) -> dict[str, Any]:
    text = template_fallback_reply(state.get("fired") or [], state.get("facts") or [])
    return {"compose_text": text, "verify_error": None}


def finalize_node(state: dict[str, Any]) -> dict[str, Any]:
    body = state.get("reply_body")
    if body is None:
        body = state.get("compose_text") or ""
        body = substitute_placeholders(body, state.get("facts") or [])

    # Partial location failures
    failed = state.get("failed_locations") or []
    if failed and state.get("resolved_locations"):
        body += f" Note: I couldn't fetch weather for: {', '.join(failed)}."

    footer = build_footer(
        state.get("fired") or [],
        state.get("no_sop_reason"),
        state.get("resolved_locations") or [],
        state.get("weather_by_location") or {},
        state.get("fetched_at"),
    )
    reply = f"{body.strip()}\n\n---\n{footer}"

    entry = {
        "question": state.get("user_message"),
        "locations": state.get("resolved_locations"),
        "when": state.get("user_when"),
        "fired": [{"id": f["id"], "severity": f["severity"]} for f in (state.get("fired") or [])],
        "facts": {f["id"]: f["value"] for f in (state.get("facts") or [])},
        "verdicts": state.get("fuzzy_verdict"),
    }
    decision_log = _cap_decision_log((state.get("decision_log") or []) + [entry], 5)

    trace = {
        "in_scope": state.get("in_scope"),
        "tags": state.get("tags"),
        "understand": state.get("understand"),
        "resolved_locations": state.get("resolved_locations"),
        "failed_locations": state.get("failed_locations"),
        "fired": state.get("fired"),
        "checked_sop_ids": state.get("checked_sop_ids"),
        "unknown": state.get("unknown"),
        "skippedSops": state.get("skipped_sops"),
        "no_sop_reason": state.get("no_sop_reason"),
        "composer_calls": state.get("composer_calls_this_turn", 0),
        "verify_error": state.get("verify_error"),
    }

    messages = list(state.get("messages") or [])
    messages.append(HumanMessage(content=state.get("user_message", "")))
    messages.append(AIMessage(content=reply))
    messages = _cap_messages(messages, 6)

    return {"reply": reply, "trace": trace, "decision_log": decision_log, "messages": messages}
