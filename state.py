"""
Shared graph state schema for the ResolveAI resolution pipeline.

Single responsibility: define the typed dictionary that flows through every
node in the StateGraph. Nothing here executes logic - it only describes the
shape of the data so each node has an explicit, reviewable contract for what
it reads and what it writes. Lives in its own module so `nodes.py` and
`graph.py` can both import it without a circular dependency.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class GraphState(TypedDict, total=False):
    # -- input ------------------------------------------------------------
    customer_id: str
    raw_complaint: str

    # -- Planner writes -------------------------------------------------
    intent: str
    planned_tools: list[str]
    reused_strategy: dict | None

    # -- Investigator writes -----------------------------------------
    tool_params: dict[str, dict[str, Any]]
    evidence: dict[str, Any]

    # -- Resolver writes -------------------------------------------
    proposed_action: dict
    policy_decision: dict

    # -- Critic writes -----------------------------------------
    critic_checks: list[dict]
    critic_score: float
    verdict: str

    # -- replan bookkeeping --------------------------------
    retry_count: int

    # -- Memory writes ------------------------------------
    episode_stored: bool
    outcome: str

    # -- cross-cutting ----------------------------------
    # Reducer: every node appends its own trace lines; they accumulate
    # across the whole run (including repeated Investigator passes).
    logs: Annotated[list[str], operator.add]
