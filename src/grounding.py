"""Facts table, placeholder substitution, verifier, footer (code-only)."""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from typing import Any

DIGIT_OUTSIDE_PLACEHOLDER = re.compile(r"(?<!\{)\d+(?:\.\d+)?(?!\})")
PLACEHOLDER = re.compile(r"\{\{(F\d+)\}\}")
SOP_WORD = re.compile(r"\bSOP\b", re.I)


def build_facts_table(
    fired: list[dict[str, Any]],
    weather_by_location: dict[str, dict[str, Any]],
    resolved: list[dict[str, Any]],
    user_when: dict[str, Any] | None,
    hourly_units: dict[str, str],
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    idx = 1

    window_label = _window_label(user_when)

    for item in fired:
        for ev in item.get("evidence") or []:
            metric = ev.get("metric")
            loc = ev.get("location") or item.get("location")
            value = ev.get("value")
            key = (metric, loc, value, ev.get("window"))
            if key in seen:
                continue
            seen.add(key)
            unit = hourly_units.get(metric, "")
            facts.append(
                {
                    "id": f"F{idx}",
                    "label": _metric_label(metric),
                    "location": loc,
                    "window": window_label or ev.get("window", ""),
                    "value": value,
                    "unit": unit,
                }
            )
            idx += 1

    # Ensure at least location context in facts for compose when no numeric evidence
    if not facts and resolved:
        for loc in resolved[:1]:
            payload = weather_by_location.get(loc["query_name"] or loc["name"])
            if not payload:
                continue
            facts.append(
                {
                    "id": "F1",
                    "label": "location",
                    "location": loc["display_name"],
                    "window": window_label,
                    "value": loc["display_name"],
                    "unit": "",
                }
            )
            break

    return facts


def _metric_label(metric: str) -> str:
    return metric.replace("_", " ")


def _window_label(user_when: dict[str, Any] | None) -> str:
    if not user_when:
        return "today"
    part = user_when.get("part", "all_day")
    day = user_when.get("day_offset", 0)
    if day == 0:
        day_s = "today"
    elif day == 1:
        day_s = "tomorrow"
    else:
        day_s = f"day+{day}"
    if part == "all_day":
        return day_s
    return f"{part} ({day_s})"


def substitute_placeholders(text: str, facts: list[dict[str, Any]]) -> str:
    by_id = {f["id"]: f for f in facts}

    def repl(m: re.Match[str]) -> str:
        fid = m.group(1)
        f = by_id.get(fid)
        if not f:
            return m.group(0)
        val = f["value"]
        unit = f.get("unit") or ""
        if isinstance(val, float) and val == int(val):
            val = int(val)
        if unit and unit not in ("", "wmo code"):
            return f"{val} {unit}".strip()
        return str(val)

    return PLACEHOLDER.sub(repl, text)


def verify_composed_text(text: str, fact_ids: set[str]) -> str | None:
    if not text or not text.strip():
        return "empty reply"
    if SOP_WORD.search(text):
        return "must not mention SOP"
    for m in PLACEHOLDER.finditer(text):
        if m.group(1) not in fact_ids:
            return f"unknown placeholder {m.group(0)}"
    # Strip placeholders before digit check
    stripped = PLACEHOLDER.sub("", text)
    if DIGIT_OUTSIDE_PLACEHOLDER.search(stripped):
        return "digits outside {{F#}} placeholders"
    return None


def build_footer(
    fired: list[dict[str, Any]],
    no_sop_reason: str | None,
    resolved: list[dict[str, Any]],
    weather_by_location: dict[str, dict[str, Any]],
    fetched_at: datetime | None,
) -> str:
    if no_sop_reason:
        return f"Policy: none applied ({no_sop_reason})"

    if not fired:
        return "Policy: none applied (no_trigger)"

    parts = []
    for i, f in enumerate(fired):
        role = "lead" if i == 0 else ""
        sev = f.get("severity", "info")
        suffix = f", {role}" if role else ""
        parts.append(f"{f['id']} ({sev}{suffix})")
    policy = "Policy: " + ", ".join(parts)

    data_bits = ["Data: Open-Meteo"]
    if resolved:
        loc = resolved[0]
        name = loc.get("display_name") or loc.get("name")
        admin = loc.get("admin1") or ""
        cc = loc.get("country_code") or ""
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        data_bits.append(f"{name}, {admin}, {cc} ({lat:.2f}, {lon:.2f})".replace(", ,", ","))
    tz_name = None
    for payload in weather_by_location.values():
        tz_name = payload.get("timezone")
        break
    if fetched_at and tz_name:
        # display in location tz approx using utc_offset from first payload
        off = 0
        for payload in weather_by_location.values():
            off = payload.get("utc_offset_seconds", 0)
            break
        local = fetched_at.astimezone(timezone(timedelta(seconds=off)))
        data_bits.append(f"fetched {local.strftime('%H:%M %Z')}")

    return policy + " | " + ", ".join(data_bits)


def template_fallback_reply(fired: list[dict[str, Any]], facts: list[dict[str, Any]]) -> str:
    if not fired:
        return "I don't have specific guidance for those conditions right now."
    lead = fired[0]
    text = lead.get("guidance", "").strip()
    if len(fired) > 1:
        text += " Additional advisories also apply for your activity."
    # Insert first fact if guidance has no placeholders
    if facts and "{{" not in text:
        f = facts[0]
        if isinstance(f.get("value"), (int, float)):
            text += f" (see {{{{{f['id']}}}}} for key conditions.)"
    return text


def extract_numbers_from_final(text: str) -> list[str]:
    return DIGIT_OUTSIDE_PLACEHOLDER.findall(text)
