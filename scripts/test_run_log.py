"""
Offline tests for run_log.py (the record of one request and the guardrail events). No network, no Cosmos.

Run: python test_run_log.py
"""
import json
import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")

import run_log
import feature_requirements as fr
import orchestrator as orch

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


# ---- events do nothing unless a request is being recorded
run_log.end()
run_log.event("regulations", "x", True)
check("an event with no recording running is ignored without error", True)

c = run_log.begin()
run_log.event("regulations", "citation_check", False)
run_log.event("regulations", "unverified_date_replaced", True, "2 August 2025")
check("events are collected with agent, rule, fired and detail",
      c.events == [{"agent": "regulations", "rule": "citation_check", "fired": False},
                   {"agent": "regulations", "rule": "unverified_date_replaced", "fired": True, "detail": "2 August 2025"}], c.events)
run_log.event("a", "b", None)
check("fired is always a real boolean", c.events[-1]["fired"] is False)

# ---- worker threads record into the same collector only when run in the context
with ThreadPoolExecutor(2) as pool:
    pool.submit(run_log.bind(run_log.event, "model_stack_analysis", "from_thread", True)).result()
    pool.submit(run_log.event, "model_stack_analysis", "lost_thread", True).result()
rules = [e["rule"] for e in c.events]
check("a worker thread submitted with bind() records into the request's collector", "from_thread" in rules, rules)
check("a plain worker thread does not (so run_agents must use bind)", "lost_thread" not in rules, rules)
run_log.end()

# ---- the Orchestrator's own run_agents hands the recorder to every sub-agent thread
real_run_one = orch._run_one
orch._run_one = lambda name, *a, **k: (run_log.event(name, "seen_in_worker", True), {"status": "ok", "output": {}})[1]
try:
    c = run_log.begin()
    orch.run_agents(["regulations", "competitive_analysis", "model_stack_analysis"], "f", {}, None)
    seen = sorted(e["agent"] for e in c.events if e["rule"] == "seen_in_worker")
    check("run_agents: all three parallel sub-agents record into the request's collector", seen == ["competitive_analysis", "model_stack_analysis", "regulations"], c.events)
    run_log.end()
    first = run_log.begin()
    orch.run_agents(["regulations"], "f", {}, None)
    run_log.end()
    second = run_log.begin()
    orch.run_agents(["regulations"], "f", {}, None)
    run_log.end()
    check("two requests in turn keep separate events", len(first.events) == 1 and len(second.events) == 1, (first.events, second.events))
finally:
    orch._run_one = real_run_one

# ---- the real guardrail code records events
c = run_log.begin()
fr.validate_interpretation({"avg_input_tokens": 200, "avg_output_tokens": 250, "reply_style": "single_response", "feature_category": "chat_assistant",
                            "stated_volume": None, "needs": {}, "reason": "x"}, "A billing chatbot.")
ev = {e["rule"]: e for e in c.events}
check("the chat-style correction is recorded as fired", ev["chat_assistant_reply_style_corrected"]["fired"] is True, c.events)
check("the token-range rule is recorded as fired, with what it changed", ev["token_estimate_held_to_typical_range"]["fired"] and "1,500" in ev["token_estimate_held_to_typical_range"]["detail"], c.events)
run_log.end()
c = run_log.begin()
fr.validate_interpretation({"avg_input_tokens": 2000, "avg_output_tokens": 250, "reply_style": "streamed_chat", "feature_category": "chat_assistant",
                            "stated_volume": None, "needs": {}, "reason": "x"}, "A billing chatbot.")
ev = {e["rule"]: e for e in c.events}
check("a clean interpretation records both rules as not fired", not ev["chat_assistant_reply_style_corrected"]["fired"] and not ev["token_estimate_held_to_typical_range"]["fired"], c.events)
run_log.end()

# ---- the Orchestrator's own checks
c = run_log.begin()
facts, classification = orch.apply_rules(
    {"regulations": {"status": "ok", "output": {"classification": "prohibited"}},
     "model_stack_analysis": {"status": "ok", "output": {"scenarios": []}}}, None, {})
check("a prohibited feature records the withheld model pick", any(e["rule"] == "prohibited_feature_model_withheld" and e["fired"] for e in c.events), c.events)
run_log.end()
c = run_log.begin()
orch.apply_rules({"regulations": {"status": "ok", "output": {"classification": "high_risk"}},
                  "model_stack_analysis": {"status": "ok", "output": {"scenarios": []}}}, None, {})
check("a high-risk feature records the pick flagged for validation", any(e["rule"] == "high_risk_pick_flagged_for_validation" for e in c.events), c.events)
run_log.end()

# ---- events read from sub-agent output
comp = {"competitors": [
    {"source_checks": {"price": "verified", "accuracy": "page_does_not_support_claim", "latency": "no_source_note_replaced"}},
    {"source_checks": {"price": "could_not_be_checked (page returned 403)", "accuracy": "verified"}}]}
evs = run_log.events_from_competitive(comp)
by = {e["rule"]: e for e in evs}
check("competitive source checks are counted by outcome", by["source_check:verified"]["detail"] == "2 claim(s)" and by["source_check:could_not_be_checked"]["detail"] == "1 claim(s)", evs)
check("replacing a claim counts as fired; verifying or not being able to check does not",
      by["source_check:page_does_not_support_claim"]["fired"] and by["source_check:no_source_note_replaced"]["fired"] and not by["source_check:verified"]["fired"] and not by["source_check:could_not_be_checked"]["fired"], evs)
check("no competitors gives no events", run_log.events_from_competitive({}) == [] and run_log.events_from_competitive(None) == [])
check("regulations overall result: passed is not fired, failed is", run_log.events_from_regulations("passed")[0]["fired"] is False and run_log.events_from_regulations("failed_guardrails")[0]["fired"] is True)

# ---- the record
outcome = {"decision": {"request_type": "new_evaluation", "agents": ["regulations", "competitive_analysis"], "constraints": {"budget": "40000"}},
           "results": {"regulations": {"status": "ok", "guardrail_status": "passed", "output": {"classification": "limited"}},
                       "competitive_analysis": {"status": "ok", "output": comp},
                       "model_stack_analysis": {"status": "error", "error": "RuntimeError: x"}},
           "reply": "The reply."}
c = run_log.begin()
run_log.event("orchestrator", "invented_question_removed", False)
rec = run_log.build_record("A billing chatbot", outcome, c, "u1", "acme", session_id="s1", trace_ids=["t1"])
run_log.end()
check("the record carries the message, routing, reply, user and company", rec["message"] == "A billing chatbot" and rec["request_type"] == "new_evaluation" and rec["agents_run"] == ["regulations", "competitive_analysis"] and rec["reply"] == "The reply." and rec["user_id"] == "u1" and rec["company_id"] == "acme" and rec["type"] == "run_record", rec)
check("every agent's full output and status is kept, including an error", rec["agent_results"]["regulations"]["output"] == {"classification": "limited"} and rec["agent_results"]["model_stack_analysis"]["error"] == "RuntimeError: x")
check("guardrail events from the code and from the agents' outputs are combined", {"invented_question_removed", "overall_guardrail_result", "source_check:verified"} <= {e["rule"] for e in rec["guardrail_events"]}, rec["guardrail_events"])
check("the review fields start empty for a person to fill in", rec["review"]["verdict"] is None and set(rec["review"]) >= {"reviewer", "verdict", "wrong_field", "notes"})
check("each record has its own id, and the trace id and session id are kept", rec["id"] and rec["trace_ids"] == ["t1"] and rec["session_id"] == "s1")
check("a request that raised still gives a record, with the error", run_log.build_record("m", None, None, "u", "c", error="RuntimeError: boom")["error"] == "RuntimeError: boom")
check("the record is plain JSON", json.loads(json.dumps(rec)) == rec)
big = {"decision": {}, "results": {"regulations": {"status": "ok", "output": {"answer": "x" * 2_000_000}}}, "reply": "r"}
r2 = run_log.build_record("m", big, None, "u", "c")
check("a record over the size limit drops the agent outputs and says so", r2.get("truncated") is True and "too large" in json.dumps(r2["agent_results"]) and len(json.dumps(r2)) < 1_500_000)

# ---- the local store
path = os.path.join(tempfile.mkdtemp(), "runs.json")
store = run_log.LocalRunStore(path)
check("an empty store lists nothing", store.list_runs() == [])
store.add(rec)
rec2 = run_log.build_record("second", outcome, None, "u2", "acme")
store.add(rec2)
check("records are stored and read back by id", store.get(rec["id"])["message"] == "A billing chatbot" and store.get("nope") is None)
check("list_runs returns run records oldest first", [r["message"] for r in store.list_runs()] == ["A billing chatbot", "second"])
store.add({"id": "x", "type": "accepted_recommendation", "created_at": "2026-01-01"})
check("other document types in the same file are not listed as runs", len(store.list_runs()) == 2)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
