"""
CLI runner for the two ResolveAI demo scenarios.

Runs Scenario A (clean pass) and Scenario B (fail-then-recover) in sequence
through the compiled graph, printing every node's output as it executes:

    Planner       -> classified intent + chosen tool list
    Investigator  -> the evidence gathered from each tool
    Resolver      -> proposed action + the policy validator's risk class
    Critic        -> checks_passed / total_checks + PASS|FAIL verdict
    Replan        -> retry_count bump (only on the fail-then-recover path)
    Memory        -> the final record written

LLM calls go through llm_client.get_llm_client(); with RESOLVEAI_LLM_MODE
unset it returns the offline MockLLMClient, so this runs with no API key.

Usage
-----
    python main.py                 # both scenarios in sequence
    python main.py --scenario A    # just one
    python main.py --scenario B
"""

from __future__ import annotations

import argparse
import json
import logging

import config
import memory_store
from graph import build_graph
from llm_client import get_llm_client

BAR = "=" * 78
SUB = "-" * 78


SCENARIOS: dict[str, dict] = {
    "A": {
        "title": "Scenario A - clean pass  (Rahul Sharma)",
        "customer_id": "C1001",
        "task": (
            "Rahul Sharma (customer C1001) reports that Rs 4,999 was deducted "
            "via transaction TX9988 for order O5001, but the order was not confirmed."
        ),
        "expect": (
            "payment SUCCESS, order NOT_CREATED, stock available "
            "-> Critic PASS first try -> create_order"
        ),
    },
    "B": {
        "title": "Scenario B - fail-then-recover  (Ananya Rao)",
        "customer_id": "C1002",
        "task": (
            "Ananya Rao (customer C1002) reports that Rs 2,999 was deducted "
            "via transaction TX9989 for order O5002, but the order is missing."
        ),
        "expect": (
            "payment UNKNOWN first -> Critic FAIL -> replan retries the payment "
            "lookup -> SUCCESS -> Critic PASS -> create_order"
        ),
    },
    "C": {
        "title": "Scenario C - memory reuse  (Karthik Iyer)",
        "customer_id": "C1003",
        "task": (
            "Karthik Iyer (customer C1003) reports that Rs 3,499 was deducted "
            "via transaction TX9990 for order O5003, but the order was not confirmed."
        ),
        "expect": (
            "same payment_order_mismatch intent -> Planner finds a prior successful "
            "episode -> MEMORY HIT, reuses the stored tool order instead of "
            "re-deriving -> clean data -> Critic PASS first try -> create_order"
        ),
    },
}


# --------------------------------------------------------------------------- #
# Per-node renderers                                                         #
# --------------------------------------------------------------------------- #
def _show_planner(u: dict) -> None:
    print("  PLANNER")
    print(f"    intent        : {u.get('intent')!r}")
    print(
        f"    extracted ids : customer_id={u.get('customer_id')}  "
        f"order_id={u.get('order_id')}  transaction_id={u.get('transaction_id')}"
    )

    # Make the memory hit/miss decision impossible to miss in the trace.
    decision = u.get("memory_decision") or "(no memory decision recorded)"
    hit = decision.lower().startswith("memory hit")
    tag = "MEMORY HIT " if hit else "MEMORY MISS" if decision.lower().startswith("memory miss") else "MEMORY ?????"
    print(f"    {'>' * 8} {tag} {'<' * 8}")
    print(f"    memory        : {decision}")
    print(f"    tool list     : {u.get('planned_tools')}"
          f"   ({'reused from prior episode' if hit else 'derived fresh'})")


def _show_investigator(u: dict) -> None:
    print("  INVESTIGATOR - evidence")
    for name, env in (u.get("investigation") or {}).items():
        data = env.get("data")
        status = data.get("status") if isinstance(data, dict) else None
        head = f"    - {name:<21} found={str(env.get('found')):<5} source={env.get('source')}"
        print(head + (f"  status={status}" if status else ""))
        print(f"      {data}")
    if u.get("product_id"):
        print(f"    (product_id discovered from the order record: {u['product_id']})")


def _show_resolver(u: dict) -> None:
    action = u.get("proposed_resolution") or {}
    risk = u.get("risk_classification") or {}
    print("  RESOLVER")
    print(f"    proposed action : {action.get('action')}   args={action.get('args')}")
    print("  POLICY VALIDATOR")
    print(
        f"    risk classification : risk={risk.get('risk')}  "
        f"auto_executable={risk.get('auto_executable')}"
    )
    print(f"    reason              : {risk.get('reason')}")


def _show_critic(u: dict) -> None:
    print("  CRITIC")
    for c in u.get("critic_checks") or []:
        print(f"    [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(
        f"    -> checks_passed = {u.get('checks_passed')}/{u.get('total_checks')}   "
        f"verdict = {u.get('verdict')}"
    )


def _show_replan(u: dict) -> None:
    print("  REPLAN")
    print(
        f"    retry_count -> {u.get('retry_count')}   "
        "(re-invokes the Investigator, not the Planner)"
    )


def _show_memory(u: dict) -> None:
    print("  MEMORY - record written")
    print(json.dumps(u.get("memory_record") or {}, indent=2, default=str))


_RENDERERS = {
    "planner": _show_planner,
    "investigator": _show_investigator,
    "resolver": _show_resolver,
    "critic": _show_critic,
    "replan": _show_replan,
    "memory": _show_memory,
}


# --------------------------------------------------------------------------- #
def run_scenario(graph, spec: dict) -> None:
    print("\n" + BAR)
    print(f"  {spec['title']}")
    print(BAR)
    print(f"  task   : {spec['task']}")
    print(f"  expect : {spec['expect']}")
    print(BAR)

    initial_state = {
        "task": spec["task"],
        "customer_id": spec["customer_id"],
        "retry_count": 0,
    }

    final: dict = {}
    for step, chunk in enumerate(graph.stream(initial_state, stream_mode="updates"), start=1):
        for node, update in chunk.items():
            print(f"\n[{step:02d}] node: {node}")
            print(SUB)
            _RENDERERS.get(node, lambda u: print(f"    {u}"))(update)
            final.update(update)

    rec = final.get("memory_record") or {}
    action = (final.get("proposed_resolution") or {}).get("action")
    print("\n" + SUB)
    print(
        f"  SUMMARY  intent={final.get('intent')!r}  "
        f"verdict={final.get('verdict')}  "
        f"checks={final.get('checks_passed')}/{final.get('total_checks')}  "
        f"retries={final.get('retry_count', 0)}  "
        f"action={action}  resolved={rec.get('resolved')}"
    )
    print(BAR)


def main() -> None:
    parser = argparse.ArgumentParser(description="ResolveAI demo runner")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), help="run only one scenario")
    args = parser.parse_args()

    # Keep module loggers quiet - the node-by-node print trace is the output.
    config.configure_logging(logging.WARNING)

    client = get_llm_client()
    print(f"LLM backend : {type(client).__name__}  (RESOLVEAI_LLM_MODE={config.LLM_MODE!r})")

    # Start from a fresh episodes table each run so `python check_memory.py`
    # shows exactly the episodes written by this run.
    try:
        config.DB_PATH.unlink()
    except FileNotFoundError:
        pass
    memory_store.init_db()
    print(f"memory db   : {config.DB_PATH}")

    graph = build_graph()

    for key in ([args.scenario] if args.scenario else sorted(SCENARIOS)):
        run_scenario(graph, SCENARIOS[key])


if __name__ == "__main__":
    main()
