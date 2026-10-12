"""
Runs every golden case through the Orchestrator once and saves what came back, so a person can review it.

Usage:
    python run_golden_cases.py [--only CASE_ID ...] [--force] [--record cosmos|local|none]

Each case is sent as a fresh single message (no saved session), as a user called golden-eval in the company
golden-eval, so these runs never appear on a real user's Dashboard. The result of each case goes to
evals/golden_runs/<case id>.json: the message, the agents that ran, every agent's output, the guardrail events, the
reply and how long it took. A case that already has a file is skipped unless --force is given, so an interrupted run
can be continued.

Makes real calls (parse and writing steps, plus whichever sub-agents routing picks). Cases go one at a time because of
the gpt-4o token limit; expect about a minute each.
"""
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import run_log

from foundry_clients import Clients
from orchestrator import handle_message

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_FILE = os.path.join(HERE, "..", "evals", "golden_hhh_cases.json")
OUT_DIR = os.path.join(HERE, "..", "evals", "golden_runs")
USER = COMPANY = "golden-eval"


def load_cases():
    return json.load(open(CASES_FILE, encoding="utf-8"))["cases"]


def run_case(case, clients, record_mode):
    collector = run_log.begin()
    outcome, failure = None, None
    try:
        outcome = handle_message(case["message"], None, clients, on_event=lambda *a, **k: None, user_id=USER, company_id=COMPANY)
    except Exception as e:
        failure = f"{type(e).__name__}: {e}"
    finally:
        record = run_log.build_record(case["message"], outcome, collector, USER, COMPANY, error=failure)
        run_log.end()
    if record_mode != "none":
        try:
            (run_log.CosmosRunStore() if record_mode == "cosmos" else run_log.LocalRunStore(os.path.join(HERE, "run_records.json"))).add(record)
        except Exception as e:
            print(f"   (record not saved to {record_mode}: {type(e).__name__})")
    return record


def main():
    args = sys.argv[1:]
    force = "--force" in args
    record_mode = args[args.index("--record") + 1] if "--record" in args else "cosmos"
    only = []
    if "--only" in args:
        i = args.index("--only") + 1
        while i < len(args) and not args[i].startswith("--"):
            only.append(args[i])
            i += 1
    cases = [c for c in load_cases() if not only or c["id"] in only]
    os.makedirs(OUT_DIR, exist_ok=True)

    clients = Clients()
    for n, case in enumerate(cases, 1):
        path = os.path.join(OUT_DIR, f"{case['id']}.json")
        if os.path.exists(path) and not force:
            print(f"[{n}/{len(cases)}] {case['id']}: already saved, skipped")
            continue
        started = time.monotonic()
        record = run_case(case, clients, record_mode)
        saved = {"case_id": case["id"], "message": case["message"], "record": record}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2, ensure_ascii=False, default=str)
        status = f"ERROR {record['error']}" if record.get("error") else f"agents={record['agents_run']}"
        print(f"[{n}/{len(cases)}] {case['id']}: {status} ({time.monotonic() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
