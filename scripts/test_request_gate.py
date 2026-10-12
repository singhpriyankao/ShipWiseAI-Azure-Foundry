"""
Offline test that the Orchestrator takes requests one at a time through the sub-agents and the
writing step. The model calls are replaced with fakes that record when they run.

Run: python test_request_gate.py
"""
import threading
import time
import warnings
warnings.filterwarnings("ignore")

import orchestrator as orch

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


intervals, lock, events = [], threading.Lock(), []


def fake_parse(model_client, message, session_state=None, max_attempts=2):
    return {"request_type": "new_evaluation", "feature_description": message, "requested_agents": None,
            "constraints": {"accuracy_target": None, "latency_target": "under 800ms", "budget": "$40K annual"}, "clarifying_question": None}


received = []


def fake_run_agents(agents, feature, constraints, clients, source_text=None, calibration=None, progress=None):
    received.append(source_text)
    start = time.time()
    time.sleep(0.3)
    with lock:
        intervals.append((start, time.time()))
    return {a: {"status": "ok", "output": {"classification": "no_specific_obligation_identified", "answer": "x", "applicability_status": "y"}}
            if a == "regulations" else {"status": "ok", "output": {}} for a in agents}


def fake_synthesize(model_client, facts):
    return "Regulatory classification: no_specific_obligation_identified\n" + orch.DISCLAIMER


orch.parse_request, orch.run_agents, orch.synthesize = fake_parse, fake_run_agents, fake_synthesize
clients = type("C", (), {"model_client": None})()
outcomes = {}


def go(i):
    outcomes[i] = orch.handle_message(f"feature {i}", None, clients, on_event=lambda m: events.append(m))


threads = [threading.Thread(target=go, args=(i,)) for i in range(3)]
[t.start() for t in threads]
[t.join(timeout=20) for t in threads]

check("every request finished", len(outcomes) == 3 and all(o["reply"] for o in outcomes.values()))
ordered = sorted(intervals)
check("the sub-agent phases never overlapped", all(ordered[i][1] <= ordered[i + 1][0] + 0.01 for i in range(len(ordered) - 1)), ordered)
waiting = [e for e in events if "another request is using the model quota" in e]
check("the requests that had to wait were told so", len(waiting) == 2, len(waiting))
check("the gate is released afterwards", orch.REQUEST_GATE.acquire(blocking=False))
orch.REQUEST_GATE.release()

# ---- the PM's own words reach the sub-agents (the parse step's one-sentence summary drops details like a stated volume)
check("the sub-agents receive the PM's original message", sorted(received) == ["feature 0", "feature 1", "feature 2"], received)
orch.run_agents = fake_run_agents
received.clear()
first = orch.handle_message("We tag about 3,000 tickets a week", None, clients, on_event=lambda m: None)
check("the original message is saved in the session for a later turn", first["session_state"]["source_text"] == "We tag about 3,000 tickets a week", first["session_state"])
def fake_parse_constraints(model_client, message, session_state=None, max_attempts=2):
    return {"request_type": "provide_constraints", "feature_description": None, "requested_agents": None,
            "constraints": {"accuracy_target": None, "latency_target": "under 800ms", "budget": "$40K annual"}, "clarifying_question": None}
orch.parse_request = fake_parse_constraints
second = orch.handle_message("latency under 800ms, $40K a year", first["session_state"], clients, on_event=lambda m: None)
check("a follow-up that only gives targets reuses the saved original words, not the targets message", received[-1] == "We tag about 3,000 tickets a week", received)
check("...and keeps them in the session", second["session_state"]["source_text"] == "We tag about 3,000 tickets a week")

def boom(agents, feature, constraints, clients, source_text=None, calibration=None):
    raise RuntimeError("sub-agent exploded")
orch.run_agents = boom
try:
    orch.handle_message("feature x", None, clients, on_event=lambda m: None)
except RuntimeError:
    pass
check("the gate is released even when a request crashes", orch.REQUEST_GATE.acquire(blocking=False))
orch.REQUEST_GATE.release()

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
