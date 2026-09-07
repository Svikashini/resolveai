"""
Model wrapper with two interchangeable backends, selected by config.LLM_MODE.

Single responsibility: expose ONE small interface for "given a prompt, get a
completion" so nodes never import the Anthropic SDK directly. Switching
between the offline mock and the real API is a single env var
(RESOLVEAI_LLM_MODE); call sites are identical either way.

Call convention
---------------
    client.complete(system=..., prompt=..., purpose="classify_intent", context={...})

`purpose` is a short tag the MockLLMClient uses to pick a canned, deterministic
response so the two demo scenarios are fully reproducible with no network and
no API key. `context` is an optional structured dict (evidence, ids) the mock
reasons over; the Anthropic backend folds it into the prompt text.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

import config

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def complete(self, *, system: str, prompt: str, **kwargs: Any) -> str: ...


# --------------------------------------------------------------------------- #
# Offline mock                                                                #
# --------------------------------------------------------------------------- #
class MockLLMClient:
    """Deterministic canned responses keyed by a `purpose` tag."""

    def complete(self, *, system: str, prompt: str, **kwargs: Any) -> str:
        purpose = kwargs.get("purpose")
        context = kwargs.get("context") or {}

        if purpose == "classify_intent":
            return self._classify_intent(context)
        if purpose == "propose_resolution":
            return json.dumps(self._propose_resolution(context))

        logger.warning("MockLLMClient: unknown purpose %r; returning empty string", purpose)
        return ""

    # -- canned reasoning ------------------------------------------------ #
    @staticmethod
    def _classify_intent(context: dict[str, Any]) -> str:
        complaint = (context.get("raw_complaint") or "").lower()
        deducted = any(w in complaint for w in ("deduct", "debited", "charged"))
        order_problem = any(
            w in complaint for w in ("not confirmed", "not created", "missing", "no order")
        )
        if deducted and order_problem:
            return "payment_order_mismatch"
        if "refund" in complaint:
            return "refund_request"
        return "general_inquiry"

    @staticmethod
    def _propose_resolution(context: dict[str, Any]) -> dict[str, Any]:
        evidence = context.get("evidence") or {}
        order_env = evidence.get("get_order") or {}
        payment_env = evidence.get("get_payment") or {}
        order = order_env.get("data") or {}
        payment = payment_env.get("data") or {}
        customer_id = context.get("customer_id") or order.get("customer_id")

        # The customer paid and has no order yet -> the fix is to create it.
        # The Critic is what gates whether the evidence is solid enough to act.
        if order_env.get("found") and order.get("status") == "NOT_CREATED":
            return {
                "action": "create_order",
                "args": {
                    "order_id": context.get("order_id"),
                    "customer_id": customer_id,
                    "product_id": order.get("product_id"),
                },
            }

        if payment_env.get("found") and isinstance(payment.get("amount"), (int, float)):
            return {
                "action": "initiate_refund",
                "args": {
                    "customer_id": customer_id,
                    "amount": payment.get("amount"),
                    "transaction_id": context.get("transaction_id"),
                },
            }

        return {
            "action": "escalate",
            "args": {"customer_id": customer_id, "reason": "insufficient evidence to act"},
        }


# --------------------------------------------------------------------------- #
# Real Anthropic backend                                                      #
# --------------------------------------------------------------------------- #
class AnthropicLLMClient:
    """Thin wrapper over anthropic.Anthropic().messages.create(...)."""

    def __init__(self) -> None:
        import anthropic  # imported lazily so `mock` mode needs no dependency

        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._model = config.ANTHROPIC_MODEL

    def complete(self, *, system: str, prompt: str, **kwargs: Any) -> str:
        context = kwargs.get("context")
        if context:
            prompt = f"{prompt}\n\nStructured context:\n{json.dumps(context, indent=2, default=str)}"

        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text").strip()


def get_llm_client() -> LLMClient:
    """Factory. Returns MockLLMClient or AnthropicLLMClient per config."""
    if config.LLM_MODE == "anthropic":
        logger.info("llm: using AnthropicLLMClient (model=%s)", config.ANTHROPIC_MODEL)
        return AnthropicLLMClient()
    logger.info("llm: using MockLLMClient (offline, deterministic)")
    return MockLLMClient()
