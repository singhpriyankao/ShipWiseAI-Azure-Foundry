"""
Prints the recorded requests, newest last: what was asked, which agents ran, every guardrail event, and whether
a reviewer has marked it yet.

Usage:
    python show_run_records.py [--cosmos] [--id RUN_ID]

Reads run_records.json next to this script, or the Cosmos `recommendations` container with --cosmos.
With --id, prints that one record in full (every agent's output and the reply).
"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

import run_log

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    args = sys.argv[1:]
    load_dotenv(os.path.join(HERE, ".env"))
    store = run_log.CosmosRunStore() if "--cosmos" in args else run_log.LocalRunStore(os.path.join(HERE, "run_records.json"))
    if "--id" in args:
        record = store.get(args[args.index("--id") + 1])
        print(json.dumps(record, indent=2) if record else "No record with that id.")
        return
    runs = store.list_runs()
    print(f"{len(runs)} recorded request(s)\n")
    for r in runs:
        fired = [e for e in r["guardrail_events"] if e["fired"]]
        verdict = (r.get("review") or {}).get("verdict") or "not reviewed"
        print(f"{r['created_at'][:19]}  {r['id']}  {r.get('duration_seconds')}s  {verdict}")
        print(f"  message: {r['message'][:110]}")
        print(f"  routed: {r.get('request_type')} -> {r.get('agents_run')}")
        print(f"  guardrails: {len(r['guardrail_events'])} checked, {len(fired)} fired" + ("".join(f"\n    - {e['agent']}: {e['rule']}" + (f" ({e['detail']})" if e.get('detail') else "") for e in fired)))
        if r.get("error"):
            print(f"  ERROR: {r['error']}")


if __name__ == "__main__":
    main()
