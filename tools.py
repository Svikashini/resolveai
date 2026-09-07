"""
Mock integration layer: five read-only "tools" backed by local JSON files.

Single responsibility: given an id, return a single record (or a well-formed
"not found" result) from the corresponding data/*.json file. These stand in
for real service calls - CRM, order management, payments, inventory,
ticketing. No graph awareness, no LLM, no state mutation: pure functions of
their arguments and the JSON on disk.

Data-file contract
------------------
Each data/<name>.json is a JSON object keyed by the lookup id:

    { "CUST-001": { ...record... }, "CUST-002": { ... } }

Public API
----------
get_customer(customer_id: str) -> dict            # data/customers.json
get_order(order_id: str) -> dict                  # data/orders.json
get_payment(transaction_id: str, retry_count: int = 0) -> dict
    # data/payments.json on the first look-up (retry_count == 0);
    # data/payments_retry.json on any retry (retry_count > 0). This models a
    # real payment processor where a settlement that reads back "UNKNOWN"
    # has resolved to a final status a few seconds later.
check_inventory(product_id: str) -> dict          # data/inventory.json
get_customer_history(customer_id: str) -> dict    # data/tickets.json

Every function returns a uniform envelope so the Investigator and Critic can
reason about presence/absence without try/except:

    {"found": bool, "data": dict | list | None, "source": "orders.json"}

TOOLS: dict[str, Callable[..., dict]]
    name -> function registry, so the Investigator can dispatch by the tool
    names the Planner produced.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Callable

import config


@lru_cache(maxsize=None)
def _load_file(filename: str) -> dict[str, Any]:
    """Read and parse one data/*.json file (cached for the process)."""
    path = config.DATA_DIR / filename
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _lookup(filename: str, key: str) -> dict[str, Any]:
    """Return the uniform envelope for `key` in `filename`."""
    table = _load_file(filename)
    record = table.get(key)
    return {
        "found": record is not None,
        "data": record,
        "source": filename,
    }


def get_customer(customer_id: str) -> dict[str, Any]:
    return _lookup("customers.json", customer_id)


def get_order(order_id: str) -> dict[str, Any]:
    return _lookup("orders.json", order_id)


def get_payment(transaction_id: str, retry_count: int = 0) -> dict[str, Any]:
    """Look up a payment.

    First attempt (retry_count == 0) reads payments.json, which may carry a
    not-yet-settled status such as "UNKNOWN". A retry (retry_count > 0) reads
    payments_retry.json instead - the same processor queried a moment later,
    now returning the settled status. The branch is explicit on purpose: the
    second attempt is NOT hard-coded to "SUCCESS", it just reads a different
    source file that happens to hold the settled record.
    """
    source = "payments.json" if retry_count == 0 else "payments_retry.json"
    return _lookup(source, transaction_id)


def check_inventory(product_id: str) -> dict[str, Any]:
    return _lookup("inventory.json", product_id)


def get_customer_history(customer_id: str) -> dict[str, Any]:
    """Past tickets for a customer. `data` is a list (possibly empty)."""
    table = _load_file("tickets.json")
    tickets = table.get(customer_id)
    return {
        "found": tickets is not None,
        "data": tickets,
        "source": "tickets.json",
    }


TOOLS: dict[str, Callable[..., dict]] = {
    "get_customer": get_customer,
    "get_order": get_order,
    "get_payment": get_payment,
    "check_inventory": check_inventory,
    "get_customer_history": get_customer_history,
}
