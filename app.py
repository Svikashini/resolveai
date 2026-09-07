"""
Streamlit UI for ResolveAI.

Thin presentation layer only. It builds the compiled LangGraph via
graph.build_graph(), streams ONE customer complaint through it, and renders
the per-node state updates it streams back. No node logic is reimplemented
here: every value shown - intent, memory hit/miss, evidence, proposed
action, risk class, critic checks, episode record - comes straight out of
the graph's own state. tools.py / policy.py / memory_store.py are imported
and used exactly as they are.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sqlite3

import streamlit as st

import config
import memory_store
import policy
import tools
from graph import build_graph
from main import SCENARIOS

# --------------------------------------------------------------------------- #
# constants / small render helpers                                           #
# --------------------------------------------------------------------------- #
GREEN = "#1a7f37"
ORANGE = "#bc4c00"
GREY = "#57606a"

NODE_TITLES = {
    "planner": "Planner",
    "investigator": "Investigator",
    "resolver": "Resolver",
    "critic": "Critic",
    "replan": "Replan",
    "memory": "Memory",
}

CUSTOM_CHOICE = "Custom complaint"


def _pill(text: str, bg: str, fg: str = "#ffffff") -> str:
    return (
        f"<span style='background:{bg};color:{fg};padding:2px 10px;border-radius:12px;"
        f"font-size:0.78rem;font-weight:700;letter-spacing:.3px'>{text}</span>"
    )


@st.cache_resource(show_spinner=False)
def get_compiled_graph():
    """Compile the StateGraph once per session and make sure the DB exists."""
    memory_store.init_db()
    return build_graph()


def episode_counts() -> tuple[int, int]:
    """(total, succeeded) rows in the episodes table - sidebar display only."""
    if not config.DB_PATH.exists():
        return 0, 0
    conn = sqlite3.connect(str(config.DB_PATH))
    try:
        total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        ok = conn.execute("SELECT COUNT(*) FROM episodes WHERE succeeded = 1").fetchone()[0]
    finally:
        conn.close()
    return total, ok


def plain_summary(final: dict) -> str:
    """One-line English gloss of the final state. Pure formatting of values
    the graph already produced - no decision logic of its own."""
    action = final.get("proposed_resolution", {}) or {}
    args = action.get("args", {}) or {}
    inv = final.get("investigation", {}) or {}
    payment = (inv.get("get_payment") or {}).get("data") or {}
    verdict = final.get("verdict")
    a = action.get("action")

    if a == "create_order":
        base = (
            f"Customer {args.get('customer_id')} paid Rs {payment.get('amount')} on "
            f"transaction {final.get('transaction_id')}, but order {args.get('order_id')} "
            f"was never created - recreate it for item {args.get('product_id')}."
        )
    elif a == "initiate_refund":
        base = (
            f"Customer {args.get('customer_id')} was charged Rs {args.get('amount')} on "
            f"transaction {args.get('transaction_id')} with no valid order - issue a refund."
        )
    else:
        base = "Evidence is missing or ambiguous - hand the case to a human agent."

    if verdict == "PASS":
        return base + " The Critic verified every check passed."
    return base + " The Critic could not confirm this even after retrying - human review required."


# --------------------------------------------------------------------------- #
# per-node trace renderers                                                   #
# --------------------------------------------------------------------------- #
def render_planner(u: dict) -> None:
    decision = (u.get("memory_decision") or "").strip()
    low = decision.lower()
    if low.startswith("memory hit"):
        st.markdown(_pill("MEMORY HIT", GREEN) + " &nbsp; reusing a stored strategy", unsafe_allow_html=True)
    elif low.startswith("memory miss"):
        st.markdown(_pill("MEMORY MISS", GREY) + " &nbsp; deriving a fresh plan", unsafe_allow_html=True)
    else:
        st.markdown(_pill("MEMORY  ?", GREY), unsafe_allow_html=True)

    st.write(f"**Decision:** {decision or '(none recorded)'}")
    st.write(f"**Intent:** `{u.get('intent')}`")
    st.write(
        f"**Extracted IDs:** customer `{u.get('customer_id')}` &nbsp;|&nbsp; "
        f"order `{u.get('order_id')}` &nbsp;|&nbsp; txn `{u.get('transaction_id')}`"
    )
    st.write("**Planned tool order:** " + " -> ".join(f"`{t}`" for t in (u.get("planned_tools") or [])))


def render_investigator(u: dict, attempt: int) -> None:
    inv = u.get("investigation", {}) or {}
    st.write(f"**Attempt {attempt}** - ran {len(inv)} tools")
    rows = []
    for name, env in inv.items():
        data = env.get("data")
        if isinstance(data, dict):
            status = data.get("status", "-")
        elif isinstance(data, list):
            status = f"{len(data)} record(s)"
        else:
            status = "-"
        rows.append(
            {
                "tool": name,
                "found": "PASS" if env.get("found") else "FAIL",
                "status": status,
                "source": env.get("source"),
                "data": json.dumps(data, default=str, ensure_ascii=False),
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    if u.get("product_id"):
        st.caption(f"product_id discovered from the order record: `{u['product_id']}`")


def render_resolver(u: dict) -> None:
    action = u.get("proposed_resolution", {}) or {}
    st.write(f"**Proposed action:** `{action.get('action')}` &nbsp; *(proposed only - not executed)*")
    st.json(action.get("args", {}) or {})


def render_policy(u: dict) -> None:
    risk = u.get("risk_classification", {}) or {}
    action = (u.get("proposed_resolution", {}) or {}).get("action")
    level = (risk.get("risk") or "").lower()
    escalation = action == "escalate"

    if level == "low" and not escalation:
        st.markdown(_pill("LOW RISK  -  AUTO-EXECUTABLE", GREEN), unsafe_allow_html=True)
    elif escalation:
        st.markdown(_pill("ESCALATION  -  HUMAN REVIEW", ORANGE), unsafe_allow_html=True)
    else:
        st.markdown(_pill(f"{level.upper() or 'UNKNOWN'} RISK  -  NEEDS APPROVAL", ORANGE), unsafe_allow_html=True)
    st.caption(risk.get("reason", ""))


def render_critic(u: dict, attempt: int) -> None:
    verdict = u.get("verdict")
    st.markdown(
        f"**Attempt {attempt}** &nbsp; " + _pill(verdict, GREEN if verdict == "PASS" else ORANGE),
        unsafe_allow_html=True,
    )
    for c in u.get("critic_checks", []) or []:
        st.write(("✅ " if c["passed"] else "❌ ") + f"**{c['name']}** - {c['detail']}")
    st.write(f"**Checks passed: {u.get('checks_passed')} / {u.get('total_checks')}**")


def render_replan(u: dict) -> None:
    st.write(
        f"Ambiguous evidence - bump `retry_count -> {u.get('retry_count')}` and re-run the "
        "Investigator (the Planner is **not** re-entered)."
    )


def render_memory(u: dict) -> None:
    rec = u.get("memory_record", {}) or {}
    crit = rec.get("critic", {}) or {}
    st.write(
        f"Episode **#{rec.get('episode_id')}** committed to `{config.DB_PATH.name}` "
        f"&nbsp;|&nbsp; resolved=`{rec.get('resolved')}` &nbsp;|&nbsp; "
        f"score=`{crit.get('score')}` &nbsp;|&nbsp; verdict=`{crit.get('verdict')}`"
    )


def render_trace(chunks: list[dict]) -> None:
    critic_verdicts = [c["update"].get("verdict") for c in chunks if c["node"] == "critic"]
    if len(critic_verdicts) > 1:
        seq = "  ->  ".join(f"Attempt {i}: {v}" for i, v in enumerate(critic_verdicts, start=1))
        st.warning(f"Replan loop fired &nbsp; - &nbsp; {seq}")

    counters = {"investigator": 0, "critic": 0}
    for c in chunks:
        node, u, step = c["node"], c["update"], c["step"]
        with st.container(border=True):
            st.markdown(f"**[{step:02d}]  {NODE_TITLES.get(node, node)}**")
            if node == "planner":
                render_planner(u)
            elif node == "investigator":
                counters["investigator"] += 1
                render_investigator(u, counters["investigator"])
            elif node == "resolver":
                render_resolver(u)
                st.divider()
                st.markdown("**Policy Validator**")
                render_policy(u)
            elif node == "critic":
                counters["critic"] += 1
                render_critic(u, counters["critic"])
            elif node == "replan":
                render_replan(u)
            elif node == "memory":
                render_memory(u)
            else:
                st.json(u)


def render_resolution(final: dict) -> None:
    action = final.get("proposed_resolution", {}) or {}
    risk = final.get("risk_classification", {}) or {}
    rec = final.get("memory_record", {}) or {}
    verdict = final.get("verdict")
    a = action.get("action")

    score = (rec.get("critic", {}) or {}).get("score")
    if score is None:
        cp = final.get("checks_passed", 0)
        tc = final.get("total_checks", 0) or 1
        score = round(cp / tc, 3)
    pct = f"{score * 100:.0f}%"

    if a == "escalate" or verdict != "PASS":
        disposition, disp_colour = "ESCALATED  -  HUMAN AGENT", ORANGE
    elif risk.get("auto_executable"):
        disposition, disp_colour = "AUTO-EXECUTED", GREEN
    else:
        disposition, disp_colour = "SENT FOR HUMAN APPROVAL", ORANGE

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Proposed action", a or "-")
        c2.metric("Confidence score", pct)
        c3.markdown("**Disposition**")
        c3.markdown(_pill(disposition, disp_colour), unsafe_allow_html=True)
        st.markdown(f"**In plain English:** {plain_summary(final)}")
        with st.expander("Episode record as persisted to SQLite"):
            st.json(rec or {"note": "run did not reach the memory node"})


# --------------------------------------------------------------------------- #
# page                                                                       #
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="ResolveAI", page_icon="🧩", layout="wide")

graph = get_compiled_graph()

st.session_state.setdefault("complaint_text", "")
st.session_state.setdefault("scenario_customer_id", "")
st.session_state.setdefault("chunks", None)
st.session_state.setdefault("final_state", None)

TITLE_TO_KEY = {spec["title"]: key for key, spec in SCENARIOS.items()}


def _on_scenario_change() -> None:
    key = TITLE_TO_KEY.get(st.session_state["scenario_choice"])
    if key:
        st.session_state["complaint_text"] = SCENARIOS[key]["task"]
        st.session_state["scenario_customer_id"] = SCENARIOS[key]["customer_id"]
    else:
        st.session_state["scenario_customer_id"] = ""


# ---- sidebar : the reused modules, on display --------------------------- #
with st.sidebar:
    st.header("Runtime")
    st.caption(f"LLM mode: `{config.LLM_MODE}`")
    total, ok = episode_counts()
    st.metric("Episodes in memory", total, f"{ok} succeeded", delta_color="off")
    st.caption(f"SQLite file: `{config.DB_PATH.name}`")
    if st.button("Reset memory (clear episodes)"):
        try:
            config.DB_PATH.unlink()
        except FileNotFoundError:
            pass
        memory_store.init_db()
        st.session_state["chunks"] = None
        st.session_state["final_state"] = None
        st.rerun()

    st.divider()
    st.caption("**Tool registry** &nbsp; `tools.TOOLS`")
    st.write(", ".join(f"`{name}`" for name in tools.TOOLS))
    st.caption("**Policy thresholds** &nbsp; `policy.py`")
    st.write(f"- refund auto-approve under Rs {policy.REFUND_AUTO_APPROVE_LIMIT:,}")
    st.write(f"- order auto-execute under Rs {policy.AUTO_EXECUTE_AMOUNT_LIMIT:,}")


# ---- 1 . complaint input --------------------------------------------- #
st.title("🧩 ResolveAI")
st.subheader("1 . Customer complaint")

st.radio(
    "Quick-select a demo scenario (pre-fills the box below)",
    [CUSTOM_CHOICE, *TITLE_TO_KEY],
    key="scenario_choice",
    on_change=_on_scenario_change,
)
st.text_area("Complaint text", key="complaint_text", height=110)
resolve_clicked = st.button("Resolve", type="primary")

if resolve_clicked:
    task = (st.session_state["complaint_text"] or "").strip()
    if not task:
        st.warning("Type a complaint or pick a scenario first.")
    else:
        initial_state: dict = {"task": task, "retry_count": 0}
        cust = st.session_state.get("scenario_customer_id", "")
        if cust:
            initial_state["customer_id"] = cust

        chunks: list[dict] = []
        with st.spinner("Running the resolution graph ..."):
            for step, chunk in enumerate(graph.stream(initial_state, stream_mode="updates"), start=1):
                for node, update in chunk.items():
                    chunks.append({"step": step, "node": node, "update": update})

        final_state: dict = {}
        for c in chunks:
            final_state.update(c["update"])

        st.session_state["chunks"] = chunks
        st.session_state["final_state"] = final_state
        st.rerun()


# ---- 2 . agent trace ------------------------------------------------ #
st.subheader("2 . Agent Trace")
if st.session_state["chunks"]:
    render_trace(st.session_state["chunks"])
else:
    st.info("Run a complaint to see the Planner -> Investigator -> Resolver -> Policy -> Critic -> Memory trace.")


# ---- 3 . resolution card ------------------------------------------ #
st.subheader("3 . Resolution")
if st.session_state["final_state"]:
    render_resolution(st.session_state["final_state"])
else:
    st.info("The resolution summary card appears here after you click Resolve.")
