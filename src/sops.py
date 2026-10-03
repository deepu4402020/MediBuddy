"""Load, validate, and evaluate SOP YAML files (re-read from disk every request)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"critical": 0, "high": 1, "caution": 2, "info": 3}
OPS = {">=", ">", "<=", "<", "==", "in"}


class ConditionLeaf(BaseModel):
    metric: str
    agg: Literal["max", "min", "sum", "avg", "first", "last", "delta"]
    window: Literal["now", "user_window", "today", "next_24h"]
    op: str
    value: float | int | list[float | int]

    @field_validator("op")
    @classmethod
    def valid_op(cls, v: str) -> str:
        if v not in OPS:
            raise ValueError(f"invalid op: {v}")
        return v


class ThresholdWhen(BaseModel):
    all: list[ConditionLeaf | dict] | None = None
    any: list[ConditionLeaf | dict] | None = None


class FuzzyVerdict(BaseModel):
    severity: Literal["info", "caution", "high", "critical"]
    guidance: str


class FuzzyVeto(BaseModel):
    forbid: str
    when: dict[str, Any]


class ThresholdSop(BaseModel):
    id: str
    title: str
    category: str
    severity: Literal["info", "caution", "high", "critical"]
    tags: list[str]
    type: Literal["threshold"] = "threshold"
    when: dict[str, Any]
    supersedes: list[str] = Field(default_factory=list)
    guidance: str

    @field_validator("guidance")
    @classmethod
    def no_digits_in_guidance(cls, v: str) -> str:
        if any(c.isdigit() for c in v):
            raise ValueError("guidance must not contain digits")
        return v


class FuzzySop(BaseModel):
    id: str
    title: str
    category: str
    severity: Literal["info", "caution", "high", "critical"] = "caution"
    tags: list[str]
    type: Literal["fuzzy"] = "fuzzy"
    rubric: list[str]
    metrics: list[str]
    verdicts: dict[str, FuzzyVerdict]
    vetoes: list[FuzzyVeto] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)


SopModel = ThresholdSop | FuzzySop

# WMO weather_code proxies (Open-Meteo has no named systems)
THUNDERSTORM_CODES = {95, 96, 99}
HEAVY_RAIN_CODES = {63, 65, 66, 67, 80, 81, 82}
FOG_CODES = {45, 48}


def load_sops(sops_dir: str | Path) -> tuple[list[SopModel], list[str], list[str]]:
    """Returns (valid_sops, skipped_files, tag_vocabulary)."""
    path = Path(sops_dir)
    valid: list[SopModel] = []
    skipped: list[str] = []
    tags: set[str] = set()

    if not path.is_dir():
        logger.error("SOPS_DIR not found: %s", path)
        return [], [str(path)], []

    for file in sorted(path.glob("*.yaml")):
        try:
            raw = yaml.safe_load(file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("root must be a mapping")
            stype = raw.get("type", "threshold")
            if stype == "fuzzy":
                sop = FuzzySop.model_validate(raw)
            else:
                sop = ThresholdSop.model_validate(raw)
            valid.append(sop)
            for t in sop.tags:
                if t != "*":
                    tags.add(t)
        except Exception as e:
            msg = f"{file.name}: {e}"
            logger.error("SKIPPED SOP: %s", msg)
            skipped.append(msg)

    return valid, skipped, sorted(tags)


def collect_metrics(sops: list[SopModel]) -> set[str]:
    metrics: set[str] = set()
    for sop in sops:
        if isinstance(sop, FuzzySop):
            metrics.update(sop.metrics)
            continue
        _walk_conditions(sop.when, metrics)
    return metrics


def _walk_conditions(node: dict[str, Any], metrics: set[str]) -> None:
    for key in ("all", "any"):
        if key not in node:
            continue
        for item in node[key]:
            if isinstance(item, dict) and "metric" in item:
                metrics.add(item["metric"])
            elif isinstance(item, dict):
                _walk_conditions(item, metrics)


def _compare(op: str, observed: float, expected: float | int | list) -> bool:
    if op == ">=":
        return observed >= float(expected)
    if op == ">":
        return observed > float(expected)
    if op == "<=":
        return observed <= float(expected)
    if op == "<":
        return observed < float(expected)
    if op == "==":
        return observed == float(expected)
    if op == "in":
        if not isinstance(expected, list):
            return False
        return observed in [float(x) for x in expected]
    return False


def _hour_part(h: int) -> str:
    if 6 <= h <= 11:
        return "morning"
    if 12 <= h <= 16:
        return "afternoon"
    if 17 <= h <= 20:
        return "evening"
    return "night"


def select_hour_indices(
    hourly_times: list[str],
    utc_offset_seconds: int,
    window: str,
    user_when: dict[str, Any] | None,
) -> list[int]:
    """Map window + user_when to indices in hourly arrays."""
    from datetime import datetime, timedelta, timezone

    if not hourly_times:
        return []

    tz = timezone(timedelta(seconds=utc_offset_seconds))
    parsed = [
        datetime.fromisoformat(t).replace(tzinfo=tz) if "T" in t else datetime.fromisoformat(t + "T00:00").replace(tzinfo=tz)
        for t in hourly_times
    ]
    now = datetime.now(tz)
    day_offset = (user_when or {}).get("day_offset", 0)
    part = (user_when or {}).get("part", "all_day")
    target_date = (now + timedelta(days=day_offset)).date()

    indices: list[int] = []
    for i, dt in enumerate(parsed):
        if window == "now":
            if abs((dt - now).total_seconds()) < 3600:
                indices.append(i)
            continue
        if window in ("user_window", "today"):
            if dt.date() != target_date:
                continue
            if window == "today" or part == "all_day":
                indices.append(i)
            elif part == "now":
                if abs((dt - now).total_seconds()) < 3600:
                    indices.append(i)
            else:
                if _hour_part(dt.hour) == part:
                    indices.append(i)
            continue
        if window == "next_24h":
            if 0 <= (dt - now).total_seconds() < 86400:
                indices.append(i)

    if window == "now" and not indices and parsed:
        # nearest hour to now
        best = min(range(len(parsed)), key=lambda i: abs((parsed[i] - now).total_seconds()))
        indices = [best]
    return indices


def aggregate(values: list[float | None], agg: str) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    if agg == "max":
        return max(clean)
    if agg == "min":
        return min(clean)
    if agg == "sum":
        return sum(clean)
    if agg == "avg":
        return sum(clean) / len(clean)
    if agg == "first":
        return clean[0]
    if agg == "last":
        return clean[-1]
    if agg == "delta":
        return clean[-1] - clean[0]
    return None


def _eval_leaf(
    leaf: ConditionLeaf,
    payload: dict[str, Any],
    user_when: dict[str, Any] | None,
    unknown: list[str],
) -> tuple[bool | None, dict[str, Any] | None]:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    series = hourly.get(leaf.metric)
    if series is None:
        unknown.append(f"{leaf.metric} missing")
        return None, None

    idx = select_hour_indices(times, payload.get("utc_offset_seconds", 0), leaf.window, user_when)
    if not idx:
        unknown.append(f"{leaf.metric} empty window {leaf.window}")
        return None, None

    vals: list[float | None] = []
    for i in idx:
        v = series[i] if i < len(series) else None
        vals.append(float(v) if v is not None else None)

    if leaf.metric == "weather_code":
        ints = [int(v) for v in vals if v is not None]
        if leaf.agg == "max":
            observed = float(max(ints)) if ints else None
        else:
            observed = aggregate(vals, leaf.agg)
    else:
        observed = aggregate(vals, leaf.agg)

    if observed is None:
        unknown.append(f"{leaf.metric} null in window")
        return None, None

    # Special metric semantics for weather_code with >= threshold on code sets
    expected = leaf.value
    if leaf.metric == "weather_code" and leaf.op == ">=" and isinstance(expected, (int, float)):
        # treat as "any code in severe set >= code" — use membership for heavy/thunder
        code = int(observed)
        if expected >= 63:
            fired = code in HEAVY_RAIN_CODES or code in THUNDERSTORM_CODES
            return fired, {"metric": leaf.metric, "value": code, "window": leaf.window}
        if expected >= 95:
            fired = code in THUNDERSTORM_CODES
            return fired, {"metric": leaf.metric, "value": code, "window": leaf.window}
        if expected >= 45:
            fired = code in FOG_CODES
            return fired, {"metric": leaf.metric, "value": code, "window": leaf.window}

    if leaf.metric == "weather_code" and leaf.op == "in" and isinstance(expected, list):
        fired = int(observed) in [int(x) for x in expected]
        return fired, {"metric": leaf.metric, "value": int(observed), "window": leaf.window}

    fired = _compare(leaf.op, observed, expected)
    return fired, {"metric": leaf.metric, "value": observed, "window": leaf.window}


def _eval_when_node(
    node: dict[str, Any],
    payload: dict[str, Any],
    user_when: dict[str, Any] | None,
    unknown: list[str],
) -> tuple[bool | None, list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []

    if "all" in node:
        results: list[bool | None] = []
        for item in node["all"]:
            if isinstance(item, dict) and "metric" in item:
                leaf = ConditionLeaf.model_validate(item)
                ok, ev = _eval_leaf(leaf, payload, user_when, unknown)
                results.append(ok)
                if ev:
                    evidence.append(ev)
            elif isinstance(item, dict):
                ok, sub_ev = _eval_when_node(item, payload, user_when, unknown)
                results.append(ok)
                evidence.extend(sub_ev)
        if any(r is None for r in results):
            return None, evidence
        return all(r for r in results if r is not None), evidence

    if "any" in node:
        any_true = False
        any_known = False
        for item in node["any"]:
            if isinstance(item, dict) and "metric" in item:
                leaf = ConditionLeaf.model_validate(item)
                ok, ev = _eval_leaf(leaf, payload, user_when, unknown)
                if ok is not None:
                    any_known = True
                if ok:
                    any_true = True
                if ev:
                    evidence.append(ev)
            elif isinstance(item, dict):
                ok, sub_ev = _eval_when_node(item, payload, user_when, unknown)
                if ok is not None:
                    any_known = True
                if ok:
                    any_true = True
                evidence.extend(sub_ev)
        if not any_known:
            return None, evidence
        return any_true, evidence

    return False, evidence


def sop_matches_tags(sop: SopModel, tags: list[str]) -> bool:
    if "*" in sop.tags:
        return True
    return bool(set(sop.tags) & set(tags))


def evaluate_threshold_at_location(
    sop: ThresholdSop,
    payload: dict[str, Any],
    user_when: dict[str, Any] | None,
    unknown: list[str],
) -> tuple[bool, list[dict[str, Any]]]:
    ok, evidence = _eval_when_node(sop.when, payload, user_when, unknown)
    if ok is None:
        return False, evidence
    return bool(ok), evidence


def rank_fired(fired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple:
        sev = SEVERITY_ORDER.get(item.get("severity", "info"), 9)
        global_first = 0 if "*" in item.get("tags", []) else 1
        return (sev, global_first, item.get("id", ""))

    return sorted(fired, key=key)


def apply_supersedes(fired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suppressed: set[str] = set()
    for item in fired:
        for sid in item.get("supersedes") or []:
            suppressed.add(sid)
    return [f for f in fired if f["id"] not in suppressed]


def evaluate_sops_for_request(
    sops: list[SopModel],
    tags: list[str],
    weather_by_location: dict[str, dict[str, Any]],
    user_when: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    """
    Returns (fired_ranked, fuzzy_candidates, checked_ids, unknown_notes).
    fired entries include id, title, severity, guidance, tags, evidence, location.
    """
    checked: list[str] = []
    unknown: list[str] = []
    fired: list[dict[str, Any]] = []
    fuzzy_candidates: list[dict[str, Any]] = []

    for sop in sops:
        checked.append(sop.id)
        if not sop_matches_tags(sop, tags):
            continue

        if isinstance(sop, FuzzySop):
            fuzzy_candidates.append(
                {
                    "id": sop.id,
                    "title": sop.title,
                    "severity": sop.severity,
                    "tags": sop.tags,
                    "rubric": sop.rubric,
                    "metrics": sop.metrics,
                    "verdicts": {k: v.model_dump() for k, v in sop.verdicts.items()},
                    "vetoes": [v.model_dump() for v in sop.vetoes],
                    "supersedes": sop.supersedes,
                }
            )
            continue

        for loc_name, payload in weather_by_location.items():
            loc_unknown: list[str] = []
            ok, evidence = evaluate_threshold_at_location(sop, payload, user_when, loc_unknown)
            unknown.extend([f"{loc_name}: {u}" for u in loc_unknown])
            if ok:
                fired.append(
                    {
                        "id": sop.id,
                        "title": sop.title,
                        "severity": sop.severity,
                        "guidance": sop.guidance,
                        "tags": sop.tags,
                        "supersedes": sop.supersedes,
                        "location": loc_name,
                        "evidence": [{**e, "location": loc_name} for e in evidence],
                    }
                )
                break  # ANY location fires

    fired = apply_supersedes(fired)
    fired = rank_fired(fired)[:3]
    return fired, fuzzy_candidates, checked, unknown
