# ResolveAI

An agentic customer-support resolution system built on a LangGraph
`StateGraph`. Given a raw customer complaint, it plans an investigation,
gathers evidence from mock back-end tools, proposes a single resolution,
critiques that proposal against the evidence, and remembers strategies that
worked so future complaints of the same kind are handled faster.



## Pipeline

```
START
  -> planner        classify intent; reuse a past successful strategy from memory if one exists
  -> investigator   run the planned mock tools; collect structured evidence
  -> resolver       propose ONE action (create_order | initiate_refund | escalate) - does NOT execute
                    -> policy validator tags it low-risk (auto) or high-risk (needs approval), logged either way
  -> critic         deterministic score = checks_passed / total_checks; < 0.7 => FAIL
  -> route_after_critic
       PASS                       -> memory
       FAIL & retries < 2         -> replan -> investigator   (loop, adjusted tool params)
       FAIL & retries >= 2        -> memory                   (fall through; episode still stored)
  memory  -> persist episode to SQLite
  -> END
```

The FAIL path never re-enters the Planner. `replan` only nudges the
Investigator's parameters (e.g. retry a lookup that returned `UNKNOWN`),
capped at 2 retries.

Safety: propose, don't execute

The Resolver never executes an action directly. It proposes one (create_order | initiate_refund | escalate), and policy.py independently classifies it as low-risk (auto-executable) or high-risk (requires human approval) based on the actual evidence gathered — payment status, amount, order state. This is logged regardless of outcome, so every proposed action has an audit trail of why it was or wasn't allowed to run automatically.

This isn't just a design intention — it fires unprompted in Scenario B: on the first pass, payment status is UNKNOWN, and the policy layer independently flags the proposed create_order as high-risk and blocks auto-execution, in the same run where the Critic separately fails the attempt for insufficient evidence. Two independent checks, same underlying ambiguity, both catching it.

## Module responsibilities

| File              | Responsibility |
|-------------------|----------------|
| `state.py`        | `GraphState` TypedDict - the contract passed between nodes |
| `config.py`       | Env-driven settings + logging setup (no business logic) |
| `tools.py`        | Five mock tools reading `data/*.json`; uniform `{found, data, source}` envelope |
| `policy.py`       | Rule-based low/high-risk classification of a proposed action |
| `critic.py`       | Deterministic checks + `checks_passed / total` scoring |
| `memory_store.py` | SQLite `episodes` table - one read, one write |
| `llm_client.py`   | One completion interface; `mock` or `anthropic` backend via env var |
| `nodes.py`        | The 6 node functions (`state -> partial update`); orchestration only |
| `graph.py`        | StateGraph wiring + the replan routing predicate; topology only |
| `main.py`         | CLI harness for the two demo scenarios |

## Mock tools

All read-only, backed by local JSON (`data/<name>.json`, an object keyed by
lookup id):

- `get_customer(customer_id)` -> `customers.json`
- `get_order(order_id)` -> `orders.json`
- `get_payment(transaction_id)` -> `payments.json`
- `check_inventory(product_id)` -> `inventory.json`
- `get_customer_history(customer_id)` -> `tickets.json`

## Demo scenarios

- **A - clean pass**: payment `SUCCESS`, order `NOT_CREATED`, inventory
  available -> Critic passes first try -> resolution `create_order`.
- **B - fail-then-recover**: first investigation returns payment `UNKNOWN`
  -> Critic FAILs (missing evidence) -> `replan` retries the payment lookup
  -> `SUCCESS` -> Critic passes.
  **C - memory reuse: a third complaint of the same payment_order_mismatch intent. The Planner queries memory_store.py for the most recent successful episode of that intent, finds one, and reuses its stored tool order instead of deriving a plan from scratch (logged explicitly as a "memory hit" vs. a "memory miss"). Clean evidence -> Critic passes first try, same as Scenario A, but the investigation strategy itself came from memory rather than being re-derived.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # defaults to offline mock mode
```

## Run

```bash
python main.py --scenario A
python main.py --scenario B
```

Set `RESOLVEAI_LLM_MODE=anthropic` (plus `ANTHROPIC_API_KEY`) in `.env` to
swap the mock model for the real API - no code changes.

Run (Streamlit UI)
bash
streamlit run app.py

Opens at localhost:8501. Pick one of the three demo scenarios from the dropdown (pre-fills the complaint text) or type a complaint freehand, then click Resolve. The Agent Trace panel renders each node as it fires — including the memory hit/miss badge, the policy validator's risk classification (green = auto-executable, orange = needs approval), and, if a replan happens, both critic attempts shown explicitly rather than only the final result. app.py calls the compiled graph directly; no node logic is duplicated in the UI layer.

## Status

A resolves on the first Critic pass (create_order, score 1.00).
B fails the first Critic pass on ambiguous payment evidence (UNKNOWN) — the policy validator independently marks it high-risk in the same pass — replan retries the payment lookup (retry_count=1, reading data/payments_retry.json), and the second Critic pass succeeds and auto-executes.
C hits memory on the Planner's first step, reusing the strategy Scenario A (or B) stored, and resolves cleanly on the first try.

Memory persistence has been verified across process restarts, not just within a single run: after populating the database via main.py, closing the environment entirely, and reopening it, check_memory.py reads back the same episodes directly from resolveai_memory.db — confirming the "learns from experience" claim is backed by real disk persistence, not in-process state that resets on every run.

The anthropic backend path exists but is untested against the live API.