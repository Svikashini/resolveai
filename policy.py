"""
Rule-based risk classifier for proposed resolution actions.

Single responsibility: take the Resolver's *proposed* action plus the
evidence it was based on, and classify it as low-risk (auto-executable) or
high-risk (requires human approval). Pure and deterministic - no LLM call.
Every invocation emits a log line recording the inputs and the verdict, so
the classification is auditable regardless of outcome.

Rules (initial)
---------------
- action == "initiate_refund":
      amount < REFUND_AUTO_APPROVE_LIMIT (Rs 1000)
        AND a payment record exists whose amount matches the refund   -> low-risk
      amount >= limit, OR no matching payment record                  -> high-risk
- action == "create_order":
      payment settled SUCCESS
        AND paid amount < AUTO_EXECUTE_AMOUNT_LIMIT (Rs 10 000)
        AND inventory available for the requested item                -> low-risk
      payment not SUCCESS / amount over threshold / no stock          -> high-risk
- action == "escalate":
      no state change by definition                                   -> low-risk
- unrecognised action                                                 -> high-risk (fail closed)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

REFUND_AUTO_APPROVE_LIMIT: int = 1000
AUTO_EXECUTE_AMOUNT_LIMIT: int = 10_000


def _result(risk: str, reason: str) -> dict[str, Any]:
    return {"risk": risk, "auto_executable": risk == "low", "reason": reason}


def _iter_evidence(evidence: dict[str, Any]):
    """Yield the `data` payload of every tool envelope in `evidence`."""
    for env in (evidence or {}).values():
        if isinstance(env, dict):
            yield env.get("data")


def _inventory_available(evidence: dict[str, Any]) -> bool:
    inv = (evidence or {}).get("check_inventory") or {}
    data = inv.get("data") or {}
    return bool(data) and data.get("stock", 0) > 0


def _payment_record(evidence: dict[str, Any]) -> dict[str, Any]:
    """The get_payment tool's `data` payload, or {} if absent."""
    env = (evidence or {}).get("get_payment") or {}
    return env.get("data") or {}


def _matching_payment(evidence: dict[str, Any], amount: Any) -> bool:
    for data in _iter_evidence(evidence):
        if isinstance(data, dict) and data.get("amount") == amount:
            return True
    return False


def classify_action(proposed_action: dict, evidence: dict) -> dict:
    action = (proposed_action or {}).get("action")
    args = (proposed_action or {}).get("args", {}) or {}

    if action == "initiate_refund":
        amount = args.get("amount")
        if (
            isinstance(amount, (int, float))
            and amount < REFUND_AUTO_APPROVE_LIMIT
            and _matching_payment(evidence, amount)
        ):
            decision = _result(
                "low",
                f"refund Rs {amount} is under the Rs {REFUND_AUTO_APPROVE_LIMIT} "
                "auto-approve limit and matches a payment record",
            )
        else:
            decision = _result(
                "high",
                f"refund Rs {amount} needs approval (>= Rs {REFUND_AUTO_APPROVE_LIMIT} "
                "or no matching payment record)",
            )

    elif action == "create_order":
        payment = _payment_record(evidence)
        payment_status = payment.get("status")
        paid_amount = payment.get("amount")
        under_threshold = (
            isinstance(paid_amount, (int, float)) and paid_amount < AUTO_EXECUTE_AMOUNT_LIMIT
        )

        if payment_status == "SUCCESS" and under_threshold and _inventory_available(evidence):
            decision = _result(
                "low",
                f"payment confirmed SUCCESS (Rs {paid_amount}) and the order amount is "
                f"under the Rs {AUTO_EXECUTE_AMOUNT_LIMIT} auto-execute threshold; "
                "inventory is available",
            )
        elif payment_status != "SUCCESS":
            decision = _result(
                "high",
                f"payment status is {payment_status!r}, not SUCCESS - order creation "
                "needs human approval until the payment settles",
            )
        elif not under_threshold:
            decision = _result(
                "high",
                f"order amount Rs {paid_amount} is at or above the Rs "
                f"{AUTO_EXECUTE_AMOUNT_LIMIT} auto-execute threshold - needs human approval",
            )
        else:
            decision = _result("high", "inventory not confirmed for the requested item")

    elif action == "escalate":
        decision = _result("low", "escalation changes no system state")

    else:
        decision = _result("high", f"unrecognised action {action!r}; failing closed")

    logger.info(
        "policy: action=%s args=%s -> risk=%s auto=%s (%s)",
        action,
        args,
        decision["risk"],
        decision["auto_executable"],
        decision["reason"],
    )
    return decision
