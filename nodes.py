"""
The pipeline's node functions - one plain function per StateGraph node.

Single responsibility: implement each node's step as
`state -> partial state update`. Nodes orchestrate only; they delegate real
work to tools.py, policy.py, critic.py, memory_store.py and llm_client.py.
The graph wiring itself lives in graph.py, not here.

Every function: GraphState -> dict partial update.
"""

from __future__ import annotations

import json
import logging
import re

import config
import critic
import memory_store
import policy
import tools
from llm_client import get_llm_client
from state import GraphState

logger = logging.getLogger(__name__)
_llm = get_llm_client()

# Default investigation plan when memory has nothing to reuse.
_DEFAULT_TOOLS = [
    "get_customer",
    "get_payment",
    "get_order",
    "check_inventory",
    "get_customer_history",
]

_ID_PATTERNS = {
    "transaction_id": re.compile(r"\bTX[A-Z0-9]+\b"),
    "order_id": re.compile(r"\bO\d+\b"),
    "customer_id": re.compile(r"\bC\d+\b"),
}


def _extract_ids(state: GraphState) -> dict[str, str]:
    text = state.get("raw_complaint", "")
    ids: dict[str, str] = {}
    for key, pattern in _ID_PATTERNS.items():
        match = pattern.search(text)
        if match:
            ids[key] = match.group(0)
    # An explicit customer_id on the input always wins.
    if state.get("customer_id"):
        ids["customer_id"] = state["customer_id"]
    return ids


def _build_tool_params(ids: dict[str, str]) -> dict[str, dict]:
    return {
        "get_customer": {"customer_id": ids.get("customer_id")},
        "get_payment": {"transaction_id": ids.get("transaction_id"), "retry_count": 0},
        "get_order": {"order_id": ids.get("order_id")},
        "check_inventory": {},  # product_id is discovered from the order at investigate time
        "get_customer_history": {"customer_id": ids.get("customer_id")},
    }


# --------------------------------------------------------------------------- #
def planner_node(state: GraphState) -> dict:
    intent = _llm.complete(
        system="You classify customer support complaints into a single intent tag.",
        prompt=state.get("raw_complaint", ""),
        purpose="classify_intent",
        context={"raw_complaint": state.get("raw_complaint", "")},
    ).strip()

    ids = _extract_ids(state)
    tool_params = _build_tool_params(ids)

    reused = memory_store.get_successful_strategy(intent)
    if reused:
        planned_tools = [step["tool"] for step in reused["strategy"]]
        line = (
            f"PLANNER  intent={intent!r}  reused strategy from memory "
            f"(score={reused['score']:.2f}): {planned_tools}"
        )
    else:
        planned_tools = list(_DEFAULT_TOOLS)
        line = f"PLANNER  intent={intent!r}  no prior strategy; default plan: {planned_tools}"

    logger.info(line)
    return {
        "intent": intent,
        "planned_tools": planned_tools,
        "reused_strategy": reused,
        "tool_params": tool_params,
        "logs": [line],
    }


# --------------------------------------------------------------------------- #
def investigator_node(state: GraphState) -> dict:
    planned_tools = state.get("planned_tools", [])
    tool_params = {k: dict(v) for k, v in state.get("tool_params", {}).items()}
    evidence: dict = {}
    lines: list[str] = []

    for name in planned_tools:
        fn = tools.TOOLS.get(name)
        if fn is None:
            lines.append(f"INVESTIGATOR  unknown tool {name!r}; skipped")
            continue

        params = dict(tool_params.get(name, {}))

        # check_inventory needs a product_id, which only the order record knows.
        if name == "check_inventory" and not params.get("product_id"):
            order_data = (evidence.get("get_order") or {}).get("data") or {}
            if order_data.get("product_id"):
                params["product_id"] = order_data["product_id"]
                tool_params.setdefault("check_inventory", {})["product_id"] = order_data["product_id"]

        # Drop unset args so the tool falls back to its own defaults cleanly.
        call_params = {k: v for k, v in params.items() if v is not None}
        result = fn(**call_params)
        evidence[name] = result

        status = (result.get("data") or {}).get("status") if isinstance(result.get("data"), dict) else None
        retry_note = ""
        if name == "get_payment":
            retry_note = f"  retry_count={params.get('retry_count', 0)}  source={result.get('source')}"
        lines.append(
            f"INVESTIGATOR  {name}({call_params}) -> found={result['found']}"
            f"{f'  status={status}' if status else ''}{retry_note}"
        )

    for line in lines:
        logger.info(line)
    return {"evidence": evidence, "tool_params": tool_params, "logs": lines}


# --------------------------------------------------------------------------- #
def resolver_node(state: GraphState) -> dict:
    evidence = state.get("evidence", {})
    tool_params = state.get("tool_params", {})
    context = {
        "customer_id": state.get("customer_id")
        or tool_params.get("get_customer", {}).get("customer_id"),
        "order_id": tool_params.get("get_order", {}).get("order_id"),
        "transaction_id": tool_params.get("get_payment", {}).get("transaction_id"),
        "evidence": evidence,
    }

    raw = _llm.complete(
        system=(
            "You are a support resolver. Given evidence, propose EXACTLY ONE "
            "action from {create_order, initiate_refund, escalate}. Do not execute it. "
            "Respond with a JSON object {\"action\": ..., \"args\": {...}}."
        ),
        prompt="Propose the single best resolution for this evidence.",
        purpose="propose_resolution",
        context=context,
    )
    try:
        proposed_action = json.loads(raw)
    except (ValueError, TypeError):
        proposed_action = {"action": "escalate", "args": {"reason": "resolver output unparseable"}}

    policy_decision = policy.classify_action(proposed_action, evidence)

    line = (
        f"RESOLVER  proposed={proposed_action.get('action')}  args={proposed_action.get('args')}  "
        f"policy: risk={policy_decision['risk']} auto_executable={policy_decision['auto_executable']}"
    )
    logger.info(line)
    return {
        "proposed_action": proposed_action,
        "policy_decision": policy_decision,
        "logs": [line],
    }


# --------------------------------------------------------------------------- #
def critic_node(state: GraphState) -> dict:
    checks = critic.run_checks(
        state.get("proposed_action", {}),
        state.get("evidence", {}),
        state.get("policy_decision", {}),
    )
    critic_score, verdict = critic.score(checks)

    lines = [
        f"CRITIC   score={critic_score:.2f} ({sum(c['passed'] for c in checks)}/{len(checks)})  verdict={verdict}"
    ]
    for c in checks:
        lines.append(f"CRITIC     [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    for line in lines:
        logger.info(line)

    return {
        "critic_checks": checks,
        "critic_score": critic_score,
        "verdict": verdict,
        "logs": lines,
    }


# --------------------------------------------------------------------------- #
def replan_node(state: GraphState) -> dict:
    """FAIL path only. Nudge Investigator params for ambiguous evidence.

    Deliberately does NOT re-run the Planner: the plan was fine, the data
    just needed a second look.
    """
    tool_params = {k: dict(v) for k, v in state.get("tool_params", {}).items()}
    retry_count = state.get("retry_count", 0) + 1
    evidence = state.get("evidence", {})

    adjustments: list[str] = []
    payment = (evidence.get("get_payment") or {}).get("data") or {}
    payment_found = (evidence.get("get_payment") or {}).get("found")
    if not payment_found or payment.get("status") in {"UNKNOWN", "PENDING", None}:
        tool_params.setdefault("get_payment", {})["retry_count"] = retry_count
        adjustments.append(f"get_payment.retry_count -> {retry_count}")

    line = (
        f"REPLAN   retry_count -> {retry_count}  "
        f"adjustments: {', '.join(adjustments) or 'none'}"
    )
    logger.info(line)
    return {"tool_params": tool_params, "retry_count": retry_count, "logs": [line]}


# --------------------------------------------------------------------------- #
def memory_node(state: GraphState) -> dict:
    verdict = state.get("verdict", "FAIL")
    succeeded = verdict == "PASS"
    intent = state.get("intent", "unknown")

    strategy = [
        {"tool": name, "params": state.get("tool_params", {}).get(name, {})}
        for name in state.get("planned_tools", [])
    ]

    episode_id = memory_store.store_episode(
        intent=intent,
        strategy=strategy,
        resolution=state.get("proposed_action", {}),
        score=state.get("critic_score", 0.0),
        succeeded=succeeded,
    )

    outcome = "resolved" if succeeded else "exhausted_retries"
    line = (
        f"MEMORY   stored episode #{episode_id}  intent={intent!r}  "
        f"succeeded={succeeded}  outcome={outcome}"
    )
    logger.info(line)
    return {"episode_stored": True, "outcome": outcome, "logs": [line]}
