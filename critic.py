"""
Deterministic validation of the Resolver's output against the evidence.

Single responsibility: run a fixed list of boolean checks and compute the
score as checks_passed / total_checks. The score is COMPUTED here, never
invented by a model. No state mutation - `critic_node` in nodes.py wraps
these functions. Split out from the node (and mirroring policy.py) so the
scoring rules are unit-testable in isolation.

Checks
------
1. evidence_for_every_claim  - every tool the proposed action relies on
                               returned a record
2. action_consistent         - the proposed action is supported by the
                               evidence (e.g. create_order only when the
                               order is NOT_CREATED, inventory is available
                               and the payment settled SUCCESS)
3. no_missing_or_ambiguous    - no gathered evidence item is absent or has
                               status "UNKNOWN"
4. policy_decision_present    - the Resolver output carried a policy decision
"""

from __future__ import annotations

from typing import Any

PASS_THRESHOLD: float = 0.7

# Which evidence a given proposed action depends on.
_REQUIRED_EVIDENCE: dict[str, tuple[str, ...]] = {
    "create_order": ("get_payment", "get_order", "check_inventory"),
    "initiate_refund": ("get_payment",),
    "escalate": (),
}

# Only a record that carries a status field set to one of these is "ambiguous".
# A record with no status field at all (e.g. inventory) is not ambiguous.
_AMBIGUOUS_STATUSES = {"UNKNOWN", "PENDING", "PROCESSING", ""}


def _env(evidence: dict[str, Any], name: str) -> dict[str, Any]:
    return (evidence or {}).get(name) or {}


def _status(evidence: dict[str, Any], name: str) -> Any:
    return (_env(evidence, name).get("data") or {}).get("status")


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def run_checks(
    proposed_action: dict, evidence: dict, policy_decision: dict
) -> list[dict]:
    action = (proposed_action or {}).get("action")
    required = _REQUIRED_EVIDENCE.get(action, ())

    # 1. evidence_for_every_claim
    missing = [t for t in required if not _env(evidence, t).get("found")]
    checks = [
        _check(
            "evidence_for_every_claim",
            not missing,
            "all required lookups returned a record"
            if not missing
            else f"no record from: {', '.join(missing)}",
        )
    ]

    # 2. action_consistent
    if action == "create_order":
        order_status = _status(evidence, "get_order")
        payment_status = _status(evidence, "get_payment")
        stock = (_env(evidence, "check_inventory").get("data") or {}).get("stock", 0)
        consistent = (
            order_status == "NOT_CREATED"
            and payment_status == "SUCCESS"
            and stock > 0
        )
        detail = (
            f"order={order_status}, payment={payment_status}, stock={stock}"
        )
    elif action == "initiate_refund":
        payment_status = _status(evidence, "get_payment")
        consistent = payment_status == "SUCCESS"
        detail = f"payment={payment_status}"
    elif action == "escalate":
        consistent = True
        detail = "escalation is always evidence-consistent"
    else:
        consistent = False
        detail = f"unrecognised action {action!r}"
    checks.append(_check("action_consistent", consistent, detail))

    # 3. no_missing_or_ambiguous
    ambiguous: list[str] = []
    for tool_name, env in (evidence or {}).items():
        env = env or {}
        if not env.get("found"):
            ambiguous.append(f"{tool_name}:not-found")
            continue
        data = env.get("data")
        if isinstance(data, dict) and "status" in data and data["status"] in _AMBIGUOUS_STATUSES:
            ambiguous.append(f"{tool_name}:{data['status']}")
    checks.append(
        _check(
            "no_missing_or_ambiguous",
            not ambiguous,
            "every evidence item is present and settled"
            if not ambiguous
            else f"ambiguous/missing: {', '.join(ambiguous)}",
        )
    )

    # 4. policy_decision_present
    has_policy = bool(policy_decision) and "risk" in policy_decision
    checks.append(
        _check(
            "policy_decision_present",
            has_policy,
            f"risk={policy_decision.get('risk')}" if has_policy else "no policy decision attached",
        )
    )

    return checks


def score(checks: list[dict]) -> tuple[float, str]:
    if not checks:
        return 0.0, "FAIL"
    passed = sum(1 for c in checks if c["passed"])
    ratio = passed / len(checks)
    return ratio, ("PASS" if ratio >= PASS_THRESHOLD else "FAIL")
