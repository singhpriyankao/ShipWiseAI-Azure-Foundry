"""
Routing eval for the Orchestrator: runs every case in
evals/orchestrator_routing_cases.json through the parse step and the routing
rules (no sub-agents are called, so it's cheap) and reports the pass rate
against the 90% target.

A case passes only if all of these match:
- request_type
- the set of agents that would run
- whether the PM is asked a question
- each constraint: where the case expects none, the parse step must not have
  extracted a number; where it expects one, the number must appear

Usage: python run_orchestrator_routing_eval.py [runs]
`runs` repeats the whole set (default 1) to see how stable the parse step is.
One browser sign-in; one small model call per case per run.
"""
import json
import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")

from foundry_clients import Clients
from orchestrator import CONSTRAINT_KEYS, decide, parse_request

CASES_PATH = os.path.join(os.path.dirname(__file__), "..", "evals", "orchestrator_routing_cases.json")
SHORT_TO_FULL = {"accuracy": "accuracy_target", "latency": "latency_target", "budget": "budget"}


def grade(case, parse, decision):
    exp = case["expected"]
    # Some injection cases are refused by Azure's own content filter before the
    # model sees them. That is an acceptable (arguably better) outcome than
    # routing them, but only where the case says so, and only if the user was
    # asked to rephrase and nothing was sent to any sub-agent.
    if case.get("acceptable_if_blocked_by_filter") and parse.get("blocked_by_filter"):
        return [] if not decision["agents"] and decision["ask_user"] else ["blocked by filter but the user was not asked to rephrase"]
    problems = []
    if decision["request_type"] != exp["request_type"]:
        problems.append(f"request_type: got {decision['request_type']!r}, expected {exp['request_type']!r}")
    if set(decision["agents"]) != set(exp["agents"]):
        problems.append(f"agents: got {sorted(decision['agents'])}, expected {sorted(exp['agents'])}")
    if decision["ask_user"] != exp["ask_user"]:
        problems.append(f"ask_user: got {decision['ask_user']}, expected {exp['ask_user']}")
    if "chosen_option" in exp:
        got_words = (parse.get("chosen_option") or "").lower()
        if exp["chosen_option"] is None and got_words:
            problems.append(f"chosen_option: expected none, parse gave {got_words!r}")
        elif exp["chosen_option"] is not None and exp["chosen_option"].lower() not in got_words:
            problems.append(f"chosen_option: expected something containing {exp['chosen_option']!r}, parse gave {got_words!r}")
    for short, full in SHORT_TO_FULL.items():
        want, got = exp["constraints"][short], parse["constraints"][full]
        if want is None:
            if got and re.search(r"\d", got):
                problems.append(f"{full}: expected none, parse extracted {got!r}")
        elif not got or want not in got.replace(",", ""):
            problems.append(f"{full}: expected something containing {want!r}, parse gave {got!r}")
    return problems


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cases = json.load(open(CASES_PATH, encoding="utf-8"))["cases"]
    clients = Clients()

    total = passed = 0
    failures = {}
    for run in range(1, runs + 1):
        for case in cases:
            parse = parse_request(clients.model_client, case["input"], case["session_state"])
            decision = decide(parse, case["session_state"])
            problems = grade(case, parse, decision)
            if parse.get("blocked_by_filter"):
                print(f"  {case['id']}: blocked by the platform content filter")
            total += 1
            passed += not problems
            if problems:
                failures.setdefault(case["id"], []).append((problems, parse))
        print(f"run {run}/{runs} done")

    print(f"\n{passed}/{total} passed ({100 * passed / total:.0f}%)  [target: >= 90%]")
    for case_id, items in failures.items():
        print(f"\n{case_id}: failed {len(items)} of {runs} run(s)")
        problems, parse = items[0]
        for p in problems:
            print("   -", p)
        print("   parse output:", json.dumps({k: parse[k] for k in ("request_type", "feature_description", "requested_agents", "constraints")}))


if __name__ == "__main__":
    main()
