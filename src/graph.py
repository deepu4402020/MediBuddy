"""LangGraph wiring — export build_graph for eval injection."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes import (
    ask_clarification,
    evaluate_sops_node,
    fetch_weather_node,
    finalize_node,
    make_compose_node,
    make_judge_fuzzy_node,
    make_understand_node,
    no_sop_node,
    resolve_locations,
    template_fallback_node,
    verify_node,
)
from src.weather import fetch_forecast, geocode_search


class GraphState(dict):
    """Typed dict stand-in; LangGraph accepts dict state."""


def _route_after_understand(state: dict[str, Any]) -> str:
    if not state.get("in_scope"):
        return "no_sop"
    if state.get("needs_location") and not state.get("locations"):
        return "ask_clarification"
    if not state.get("tags"):
        return "no_sop"
    return "resolve_locations"


def _route_after_resolve(state: dict[str, Any]) -> str:
    resolved = state.get("resolved_locations") or []
    if not resolved:
        return "fail_location"
    return "fetch_weather"


def _route_after_fetch(state: dict[str, Any]) -> str:
    weather = state.get("weather_by_location") or {}
    if not weather:
        return "fail_weather"
    return "evaluate_sops"


def _route_after_evaluate(state: dict[str, Any]) -> str:
    fired = state.get("fired") or []
    fuzzy = state.get("fuzzy_candidates") or []
    if fired:
        return "compose"
    if fuzzy:
        return "judge_fuzzy"
    return "no_trigger"


def _route_after_judge(state: dict[str, Any]) -> str:
    return "compose"


def _route_after_verify(state: dict[str, Any]) -> str:
    if state.get("verify_error"):
        retries = state.get("retries") or 0
        if retries < 1:
            return "retry_compose"
        return "template_fallback"
    return "finalize"


def build_graph(
    llm,
    sops_dir: str,
    geocode=geocode_search,
    fetch_forecast_fn=fetch_forecast,
    checkpointer=None,
):
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(dict)

    async def resolve_wrapped(state: dict[str, Any]) -> dict[str, Any]:
        return await resolve_locations(state, geocode=geocode)

    async def fetch_wrapped(state: dict[str, Any]) -> dict[str, Any]:
        return await fetch_weather_node(state, fetch_forecast_fn=fetch_forecast_fn)

    def no_topic(state: dict[str, Any]) -> dict[str, Any]:
        return {"no_sop_reason": "no_topic"}

    def no_trigger(state: dict[str, Any]) -> dict[str, Any]:
        return {"no_sop_reason": "no_trigger"}

    def fail_location(state: dict[str, Any]) -> dict[str, Any]:
        return {"no_sop_reason": "location_unresolved"}

    def fail_weather(state: dict[str, Any]) -> dict[str, Any]:
        return {"no_sop_reason": "weather_unavailable"}

    def bump_retry(state: dict[str, Any]) -> dict[str, Any]:
        return {"retries": (state.get("retries") or 0) + 1}

    graph.add_node("understand", make_understand_node(llm, sops_dir))
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("no_sop", no_sop_node)
    graph.add_node("no_topic", no_topic)
    graph.add_node("no_trigger", no_trigger)
    graph.add_node("resolve_locations", resolve_wrapped)
    graph.add_node("fail_location", fail_location)
    graph.add_node("fetch_weather", fetch_wrapped)
    graph.add_node("fail_weather", fail_weather)
    graph.add_node("evaluate_sops", evaluate_sops_node)
    graph.add_node("judge_fuzzy", make_judge_fuzzy_node(llm))
    graph.add_node("compose", make_compose_node(llm))
    graph.add_node("verify", verify_node)
    graph.add_node("retry_bump", bump_retry)
    graph.add_node("template_fallback", template_fallback_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "understand")

    graph.add_conditional_edges(
        "understand",
        _route_after_understand,
        {
            "ask_clarification": "ask_clarification",
            "no_sop": "no_topic",
            "resolve_locations": "resolve_locations",
        },
    )

    graph.add_edge("ask_clarification", "finalize")
    graph.add_edge("no_topic", "no_sop")
    graph.add_edge("no_trigger", "no_sop")
    graph.add_edge("fail_location", "no_sop")
    graph.add_edge("fail_weather", "no_sop")
    graph.add_edge("no_sop", "finalize")

    graph.add_conditional_edges(
        "resolve_locations",
        _route_after_resolve,
        {"fetch_weather": "fetch_weather", "fail_location": "fail_location"},
    )

    graph.add_conditional_edges(
        "fetch_weather",
        _route_after_fetch,
        {"evaluate_sops": "evaluate_sops", "fail_weather": "fail_weather"},
    )

    graph.add_conditional_edges(
        "evaluate_sops",
        _route_after_evaluate,
        {
            "compose": "compose",
            "judge_fuzzy": "judge_fuzzy",
            "no_trigger": "no_trigger",
        },
    )

    graph.add_edge("judge_fuzzy", "compose")
    graph.add_edge("compose", "verify")

    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {
            "finalize": "finalize",
            "retry_compose": "retry_bump",
            "template_fallback": "template_fallback",
        },
    )

    graph.add_edge("retry_bump", "compose")
    graph.add_edge("template_fallback", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


async def run_turn(graph, session_id: str, message: str, prior: dict[str, Any] | None = None) -> dict[str, Any]:
    config = {"configurable": {"thread_id": session_id}}
    base = prior or {}
    input_state = {
        **base,
        "user_message": message,
    }
    result = await graph.ainvoke(input_state, config=config)
    return result
