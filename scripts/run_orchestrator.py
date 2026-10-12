"""
Runs the Orchestrator on one PM message.

Usage:
    python run_orchestrator.py "message" [--session-file state.json] [--cosmos SESSION_ID]
                               [--user USER_ID] [--company COMPANY_ID] [--store-file PATH]
                               [--record local|cosmos|none]

--session-file: local JSON file that carries session state between runs, so
the "here are my targets" second turn can be tried without Cosmos.
--cosmos SESSION_ID: use the Cosmos `sessions` container instead (it must
exist first; see session_store.py).
--store-file: where recorded choices and reports are kept when --cosmos is not used (default
accepted_recommendations.json next to this script); handy for trying things on a copy.
--user / --company: who a recorded choice belongs to (default demo-user / demo-company; there is no
sign-in yet).

When the PM accepts an option ("I'll go with option 2"), the choice is recorded: in the Cosmos
`recommendations` container when --cosmos is used, otherwise in accepted_recommendations.json next to
this script (git-ignored). See recommendation_store.py.

--record: where the record of this request (message, every agent's output, guardrail events, reply) goes:
run_records.json next to this script (default, git-ignored), the Cosmos `recommendations` container, or nowhere.
See run_log.py.

Makes real calls: the parse and writing steps, and whichever sub-agents the
routing picks (each is a gpt-4o call, two with Bing grounding). One browser
sign-in.
"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import run_log

from foundry_clients import Clients
from orchestrator import handle_message
from recommendation_store import CosmosRecommendationStore, LocalRecommendationStore
from session_store import CosmosSessionStore


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    message = args.pop(0)
    session_file = args[args.index("--session-file") + 1] if "--session-file" in args else None
    cosmos_id = args[args.index("--cosmos") + 1] if "--cosmos" in args else None
    user_id = args[args.index("--user") + 1] if "--user" in args else "demo-user"
    company_id = args[args.index("--company") + 1] if "--company" in args else "demo-company"
    store_file = args[args.index("--store-file") + 1] if "--store-file" in args else os.path.join(os.path.dirname(os.path.abspath(__file__)), "accepted_recommendations.json")

    state, store = None, None
    if cosmos_id:
        store = CosmosSessionStore()
        state = store.get(cosmos_id)
    elif session_file and os.path.exists(session_file):
        state = json.load(open(session_file, encoding="utf-8"))

    record_mode = args[args.index("--record") + 1] if "--record" in args else "local"
    recommendations = CosmosRecommendationStore() if cosmos_id else LocalRecommendationStore(store_file)
    collector = run_log.begin()
    outcome, failure = None, None
    try:
        outcome = handle_message(message, state, Clients(), recommendation_store=recommendations, user_id=user_id, company_id=company_id)
    except Exception as e:
        failure = f"{type(e).__name__}: {e}"
        raise
    finally:
        # A failed request is kept too: it is the one most worth reviewing.
        if record_mode != "none":
            record = run_log.build_record(message, outcome, collector, user_id, company_id, session_id=cosmos_id, error=failure)
            local_store = run_log.LocalRunStore(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_records.json"))
            try:
                (run_log.CosmosRunStore() if record_mode == "cosmos" else local_store).add(record)
            except Exception as e:  # the record must not be lost because the database was unreachable
                local_store.add(record)
                record_mode = f"local (Cosmos write failed: {type(e).__name__})"
            print(f"\n(run recorded as {record['id']} in {record_mode}; {len(record['guardrail_events'])} guardrail events)")
        run_log.end()

    print("\n--- routing ---")
    d = outcome["decision"]
    print(f"request_type={d['request_type']} agents={d['agents']} ask_user={d['ask_user']}")
    for name, r in outcome["results"].items():
        extra = f" guardrail={r.get('guardrail_status')}" if r.get("guardrail_status") else ""
        print(f"  {name}: {r['status']}{extra}" + (f" ({r['error']})" if r["status"] == "error" else ""))
    print("\n--- reply ---")
    print(outcome["reply"])
    if outcome.get("accepted"):
        print(f"\n(recorded as {outcome['accepted']['id']})")

    # The sub-agents' raw outputs, so what the writing step was given can be checked
    # against what it wrote (the reply alone can't show that). Overwritten each run.
    saved_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_run.json")
    with open(saved_path, "w", encoding="utf-8") as f:
        json.dump({"message": message, "decision": d, "results": outcome["results"], "facts": outcome.get("facts")}, f, indent=2, default=str)
    print(f"\n(raw sub-agent outputs saved to {saved_path})")

    if outcome["session_state"] is not None:
        if cosmos_id:
            store.put(cosmos_id, outcome["session_state"])
        elif session_file:
            json.dump(outcome["session_state"], open(session_file, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
