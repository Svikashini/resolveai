"""
Standalone smoke test for tools.py - no pytest, no graph, no LLM.

Calls each of the five mock tools directly with the demo IDs and prints
every return value, clearly labeled. The interesting case is the last one:
get_payment("TX9989", retry_count=0) reads payments.json and comes back
UNKNOWN; get_payment("TX9989", retry_count=1) reads payments_retry.json and
comes back SUCCESS - the fail-then-recover behaviour Scenario B depends on.

Run:  python test_tools.py
"""

from __future__ import annotations

import json

from tools import (
    check_inventory,
    get_customer,
    get_customer_history,
    get_order,
    get_payment,
)


def show(label: str, value: object) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(json.dumps(value, indent=2, default=str))


def main() -> None:
    print("=" * 70)
    print("  tools.py - direct call smoke test")
    print("=" * 70)

    # ---- happy-path IDs (Scenario A) --------------------------------------
    show("get_customer('C1001')", get_customer("C1001"))
    show("get_order('O5001')", get_order("O5001"))
    show("get_payment('TX9988')  [retry_count defaults to 0]", get_payment("TX9988"))
    show("check_inventory('LAPTOP001')", check_inventory("LAPTOP001"))
    show("get_customer_history('C1001')", get_customer_history("C1001"))

    # ---- Scenario B IDs -------------------------------------------------
    show("get_customer('C1002')", get_customer("C1002"))
    show("get_order('O5002')", get_order("O5002"))
    show("check_inventory('LAPTOP001')  [same product as O5002]", check_inventory("LAPTOP001"))
    show("get_customer_history('C1002')", get_customer_history("C1002"))

    # ---- the retry case ----------------------------------------------
    print("\n" + "=" * 70)
    print("  RETRY CASE: TX9989 lookup, first attempt vs. retry")
    print("=" * 70)

    first = get_payment("TX9989", retry_count=0)
    retry = get_payment("TX9989", retry_count=1)

    show("get_payment('TX9989', retry_count=0)  -> reads payments.json", first)
    show("get_payment('TX9989', retry_count=1)  -> reads payments_retry.json", retry)

    print("\n" + "-" * 70)
    print("  SUMMARY")
    print("-" * 70)
    print(f"  first attempt status : {first['data']['status']!r}  (from {first['source']})")
    print(f"  retry attempt status : {retry['data']['status']!r}  (from {retry['source']})")
    changed = first["data"]["status"] != retry["data"]["status"]
    print(f"  status changed on retry : {changed}  "
          f"({'as expected for Scenario B' if changed else 'UNEXPECTED'})")
    print("=" * 70)


if __name__ == "__main__":
    main()
