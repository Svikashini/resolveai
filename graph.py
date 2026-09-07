"""
LangGraph StateGraph assembly for the ResolveAI resolution pipeline.

Single responsibility: define the state contract (`AgentState`), the five
node functions, and the topology - including the one conditional edge that
implements the capped replan loop. Real work is delegated:

    * back-end look-ups   -> tools.TOOLS / tools.py
    * risk classification -> policy.classify_action / policy.py

Nothing here is reimplemented from those modules.

Topology
--------
    START
      -> planner        extract IDs from the task; choose the tool list
      -> investigator   run the planned tools; collect results
      -> resolver       propose ONE resolution; attach policy risk class
      -> critic         checks_passed / total_checks -> PASS | FAIL
      -> route_from_critic (conditional)
             PASS                              -> memory
             FAIL and retry_count >= 2         -> memory   (give up, still record)
             FAIL and retry_count < 2          -> replan
    replan  -> investigator     (lighter loop: NOT back through the planner;
                                 just bumps retry_count and re-investigates)
    memory  -> END

Public API
----------
AgentState        - TypedDict carried between nodes
build_graph()     - construct, wire and compile the StateGraph
route_from_critic - the conditional-edge predicate (pure)
"""

from __future__ import annotations

import operator
import json
import re
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

import config
import memory_store
import policy
import tools
from llm_client import get_llm_client

# retry_count at which the replan loop stops and we fall through to memory.
MAX_RETRIES: int = 2

# One shared model wrapper. get_llm_client() returns the offline MockLLMClient
# unless RESOLVEAI_LLM_MODE=anthropic, so importing this module needs no API key.
_llm = get_llm_client()


# --------------------------------------------------------------------------- #
# State contract                                                             #
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    # -- input ---------------------------------------------------------------
    task: str                       # raw task / complaint description

    # -- planner: classified intent -------------------------------------
    intent: str                     # e.g. "payment_order_mismatch"

    # -- planner: extracted IDs -------------------------------------------
    customer_id: str
    order_id: str
    transaction_id: str
    product_id: str                 # usually discovered during investigation

    # -- planner: chosen plan -------------------------------------------
    planned_tools: list[str]
    memory_decision: str            # "Memory hit ..." | "Memory miss ..."

    # -- investigator --------------------------------------------------
    investigation: dict[str, Any]   # tool name -> {found, data, source}

    # -- resolver ---------------------------------------------------
    proposed_resolution: dict[str, Any]   # {"action": str, "args": {...}}

    # -- policy validator ----------------------------------------
    risk_classification: dict[str, Any]   # {"risk", "auto_executable", "reason"}

    # -- critic --------------------------------------------------
    critic_checks: list[dict]       # [{name, passed, detail}, ...]
    checks_passed: int
    total_checks: int
    verdict: str                    # "PASS" | "FAIL"

    # -- replan bookkeeping ----------------------------------
    retry_count: int

    # -- memory ---------------------------------------------
    memory_record: dict[str, Any]   # the final episode record

    # -- cross-cutting trace (append-only) ----------------
    trace: Annotated[list[str], operator.add]


# --------------------------------------------------------------------------- #
# Helpers                                                                    #
# --------------------------------------------------------------------------- #
_ID_PATTERNS = {
    "customer_id": re.compile(r"\bC\d+\b"),
    "order_id": re.compile(r"\bO\d+\b"),
    "transaction_id": re.compile(r"\bTX[A-Z0-9]+\b"),
    "product_id": re.compile(r"\b[A-Z]{3,}\d{2,}\b"),
}

_DEFAULT_PLAN = [
    "get_customer",
    "get_payment",
    "get_order",
    "check_inventory",
    "get_customer_history",
]


def _data(investigation: dict[str, Any], tool_name: str) -> dict[str, Any]:
    """The `data` payload of one tool result, or {} if absent/None."""
    env = investigation.get(tool_name) or {}
    return env.get("data") or {}


# --------------------------------------------------------------------------- #
# Nodes                                                                      #
# --------------------------------------------------------------------------- #
_PLAN_BY_INTENT: dict[str, list[str]] = {
    "payment_order_mismatch": _DEFAULT_PLAN,
    "refund_request": ["get_customer", "get_payment", "get_customer_history"],
}


def planner_node(state: AgentState) -> dict:
    """Classify the intent (llm_client), pull IDs from the task, pick the tools.

    Before deriving a tool list from scratch, ask memory_store for the most
    recent *successful* episode with this same intent. If one exists, reuse its
    stored tool order verbatim (a "memory hit") instead of re-deriving; the
    trace records exactly which episode was reused. With nothing to reuse (a
    "memory miss") it falls back to the intent-keyed default plan.
    """
    task = state.get("task", "")

    intent = _llm.complete(
        system="Classify the customer support complaint into one short intent tag.",
        prompt=task,
        purpose="classify_intent",
        context={"raw_complaint": task},
    ).strip()

    found_ids: dict[str, str] = {}
    for field, pattern in _ID_PATTERNS.items():
        m = pattern.search(task)
        if m:
            found_ids[field] = m.group(0)
    if state.get("customer_id"):                     # explicit input wins
        found_ids["customer_id"] = state["customer_id"]

    prior = memory_store.get_successful_strategy(intent)
    if prior:
        planned_tools = list(prior["strategy"])
        memory_decision = (
            f"Memory hit: reusing strategy from episode #{prior['id']}, "
            f"tool order: {planned_tools}"
        )
    else:
        planned_tools = list(_PLAN_BY_INTENT.get(intent, _DEFAULT_PLAN))
        memory_decision = "Memory miss: no prior strategy found, deriving fresh."

    line = f"PLANNER   intent={intent!r}  ids={found_ids or '{}'}  plan={planned_tools}"
    return {
        **found_ids,
        "intent": intent,
        "planned_tools": planned_tools,
        "memory_decision": memory_decision,
        "retry_count": state.get("retry_count", 0),
        "trace": [line, f"PLANNER   {memory_decision}"],
    }


def investigator_node(state: AgentState) -> dict:
    """Run each planned tool; thread the retry count into get_payment."""
    planned_tools = state.get("planned_tools", [])
    retry_count = state.get("retry_count", 0)
    investigation: dict[str, Any] = {}
    product_id = state.get("product_id")
    lines: list[str] = []

    for name in planned_tools:
        fn = tools.TOOLS.get(name)
        if fn is None:
            lines.append(f"INVESTIGATOR  unknown tool {name!r}; skipped")
            continue

        if name == "get_customer" or name == "get_customer_history":
            result = fn(state.get("customer_id", ""))
        elif name == "get_order":
            result = fn(state.get("order_id", ""))
        elif name == "get_payment":
            # retry_count > 0 -> tools.get_payment reads payments_retry.json
            result = fn(state.get("transaction_id", ""), retry_count=retry_count)
        elif name == "check_inventory":
            pid = product_id or _data(investigation, "get_order").get("product_id")
            product_id = pid
            result = fn(pid or "")
        else:  # pragma: no cover - registry is closed
            continue

        investigation[name] = result
        status = result.get("data", {}).get("status") if isinstance(result.get("data"), dict) else None
        extra = f"  source={result['source']}" if name == "get_payment" else ""
        lines.append(
            f"INVESTIGATOR  {name} -> found={result['found']}"
            f"{f'  status={status}' if status else ''}{extra}"
        )

    lines.insert(0, f"INVESTIGATOR  pass (retry_count={retry_count})")
    return {
        "investigation": investigation,
        "product_id": product_id,
        "trace": lines,
    }


def _rule_based_resolution(state: AgentState, inv: dict, order: dict, payment: dict) -> dict:
    """Deterministic fallback when the model output can't be used."""
    if (inv.get("get_order") or {}).get("found") and order.get("status") == "NOT_CREATED":
        return {
            "action": "create_order",
            "args": {
                "order_id": state.get("order_id"),
                "customer_id": state.get("customer_id"),
                "product_id": state.get("product_id") or order.get("product_id"),
            },
        }
    if (inv.get("get_payment") or {}).get("found") and payment.get("status") == "SUCCESS":
        return {
            "action": "initiate_refund",
            "args": {
                "customer_id": state.get("customer_id"),
                "amount": payment.get("amount"),
                "transaction_id": state.get("transaction_id"),
            },
        }
    return {
        "action": "escalate",
        "args": {"customer_id": state.get("customer_id"), "reason": "insufficient evidence"},
    }


def resolver_node(state: AgentState) -> dict:
    """Propose ONE resolution (llm_client) then classify its risk via policy.py."""
    inv = state.get("investigation", {})
    order = _data(inv, "get_order")
    payment = _data(inv, "get_payment")

    raw = _llm.complete(
        system=(
            "You are a support resolver. Propose EXACTLY ONE action from "
            "{create_order, initiate_refund, escalate} as JSON "
            '{"action": ..., "args": {...}}. Do not execute it.'
        ),
        prompt="Propose the single best resolution for this evidence.",
        purpose="propose_resolution",
        context={
            "customer_id": state.get("customer_id"),
            "order_id": state.get("order_id"),
            "transaction_id": state.get("transaction_id"),
            "evidence": inv,
        },
    )
    try:
        proposed = json.loads(raw)
        if not (isinstance(proposed, dict) and proposed.get("action")):
            raise ValueError("missing action")
    except (ValueError, TypeError):
        proposed = _rule_based_resolution(state, inv, order, payment)

    risk = policy.classify_action(proposed, inv)   # <- policy.py, not reimplemented
    line = (
        f"RESOLVER  action={proposed['action']}  "
        f"risk={risk['risk']}  auto_executable={risk['auto_executable']}"
    )
    return {
        "proposed_resolution": proposed,
        "risk_classification": risk,
        "trace": [line],
    }


def critic_node(state: AgentState) -> dict:
    """Deterministic checks; verdict = PASS only when every check passes."""
    inv = state.get("investigation", {})
    planned_tools = state.get("planned_tools", [])
    risk = state.get("risk_classification", {})

    payment = _data(inv, "get_payment")
    order = _data(inv, "get_order")
    inventory = _data(inv, "check_inventory")

    checks = [
        {
            "name": "all_lookups_succeeded",
            "passed": all((inv.get(t) or {}).get("found") for t in planned_tools),
            "detail": "every planned tool returned a record",
        },
        {
            "name": "payment_settled",
            "passed": payment.get("status") == "SUCCESS",
            "detail": f"payment status = {payment.get('status')!r}",
        },
        {
            "name": "order_status_known",
            "passed": bool(order.get("status")),
            "detail": f"order status = {order.get('status')!r}",
        },
        {
            "name": "inventory_available",
            "passed": inventory.get("stock", 0) > 0,
            "detail": f"stock = {inventory.get('stock', 0)}",
        },
        {
            "name": "risk_classified",
            "passed": bool(risk.get("risk")),
            "detail": f"risk = {risk.get('risk')!r}",
        },
    ]

    checks_passed = sum(1 for c in checks if c["passed"])
    total_checks = len(checks)
    verdict = "PASS" if checks_passed == total_checks else "FAIL"

    lines = [f"CRITIC    {checks_passed}/{total_checks} checks passed -> {verdict}"]
    lines += [f"CRITIC      [{'ok ' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}" for c in checks]
    return {
        "critic_checks": checks,
        "checks_passed": checks_passed,
        "total_checks": total_checks,
        "verdict": verdict,
        "trace": lines,
    }


def replan_node(state: AgentState) -> dict:
    """Lighter than a full replan: bump retry_count and go re-investigate.

    Does NOT re-enter the planner - the plan was fine, the payment record
    just needs a second look (which tools.get_payment serves from
    payments_retry.json once retry_count > 0).
    """
    retry_count = state.get("retry_count", 0) + 1
    line = f"REPLAN    retry_count -> {retry_count}; re-invoking investigator"
    return {"retry_count": retry_count, "trace": [line]}


def memory_node(state: AgentState) -> dict:
    """Persist this episode to the SQLite `episodes` table, then expose the record.

    Terminal node. The strategy stored is the *ordered tool list* actually
    planned; the score is checks_passed / total_checks. memory_store.store_episode
    opens config.DB_PATH, INSERTs, and commits - this is a real file on disk,
    not in-process state.
    """
    verdict = state.get("verdict", "FAIL")
    succeeded = verdict == "PASS"
    checks_passed = state.get("checks_passed", 0)
    total_checks = state.get("total_checks", 0) or 1
    score = round(checks_passed / total_checks, 3)
    strategy = list(state.get("planned_tools", []))          # tool order
    resolution = state.get("proposed_resolution", {})

    episode_id = memory_store.store_episode(
        intent=state.get("intent", "unknown"),
        strategy=strategy,
        resolution=resolution,
        score=score,
        succeeded=succeeded,
    )

    record = {
        "episode_id": episode_id,
        "db_path": str(config.DB_PATH),
        "intent": state.get("intent"),
        "task": state.get("task"),
        "customer_id": state.get("customer_id"),
        "ids": {
            "order_id": state.get("order_id"),
            "transaction_id": state.get("transaction_id"),
            "product_id": state.get("product_id"),
        },
        "strategy": strategy,
        "resolution": resolution,
        "risk_classification": state.get("risk_classification", {}),
        "critic": {
            "checks_passed": checks_passed,
            "total_checks": total_checks,
            "score": score,
            "verdict": verdict,
        },
        "retry_count": state.get("retry_count", 0),
        "resolved": succeeded,
    }
    line = (
        f"MEMORY    committed episode #{episode_id} to {config.DB_PATH.name}  "
        f"succeeded={succeeded}  score={score}"
    )
    return {"memory_record": record, "trace": [line]}


# --------------------------------------------------------------------------- #
# Routing                                                                    #
# --------------------------------------------------------------------------- #
def route_from_critic(state: AgentState) -> Literal["memory", "replan"]:
    """PASS, or retries exhausted -> memory. Otherwise -> replan."""
    if state.get("verdict") == "PASS":
        return "memory"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "memory"
    return "replan"


# --------------------------------------------------------------------------- #
# Assembly                                                                   #
# --------------------------------------------------------------------------- #
def build_graph():
    """Construct, wire and compile the resolution StateGraph."""
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("investigator", investigator_node)
    g.add_node("resolver", resolver_node)
    g.add_node("critic", critic_node)
    g.add_node("replan", replan_node)
    g.add_node("memory", memory_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "investigator")
    g.add_edge("investigator", "resolver")
    g.add_edge("resolver", "critic")
    g.add_conditional_edges(
        "critic",
        route_from_critic,
        {"memory": "memory", "replan": "replan"},
    )
    g.add_edge("replan", "investigator")
    g.add_edge("memory", END)

    return g.compile()


if __name__ == "__main__":
    compiled = build_graph()

    print("=" * 70)
    print("  ResolveAI graph - nodes & edges")
    print("=" * 70)
    print("\nNodes:")
    for name in ["planner", "investigator", "resolver", "critic", "replan", "memory"]:
        print(f"  - {name}")

    print("\nStatic edges:")
    print("  START      -> planner")
    print("  planner    -> investigator")
    print("  investigator -> resolver")
    print("  resolver   -> critic")
    print("  replan     -> investigator")
    print("  memory     -> END")

    print("\nConditional edge (route_from_critic):")
    print(f"  critic --PASS------------------------> memory")
    print(f"  critic --FAIL & retry_count >= {MAX_RETRIES}------> memory")
    print(f"  critic --FAIL & retry_count <  {MAX_RETRIES}------> replan")

    print("\nMermaid:")
    try:
        print(compiled.get_graph().draw_mermaid())
    except Exception as exc:  # pragma: no cover - drawing is best-effort
        print(f"  (draw_mermaid unavailable: {exc})")

    print("\nCompiled OK.")
