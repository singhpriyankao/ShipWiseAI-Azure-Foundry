"""
Offline tests for app/server.py's /api/health and /api/evaluate. A fake Orchestrator and fake stores stand in
for Azure; Supabase is not contacted. Sign-in is replaced by a test-only stand-in (see `fake_user`) that
lets a request through as a test user when it carries the header x-spike-key: spike-secret. The server itself has no
such backdoor (a test below checks that the real sign-in refuses that header).

Run (from this folder): python test_server.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("NEXT_PUBLIC_SUPABASE_URL", "https://example.invalid")
os.environ.setdefault("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", "test-key")
from fastapi import Header, HTTPException
from fastapi.testclient import TestClient

import server
sys.path.insert(0, server.SCRIPTS)
import orchestrator
import run_log
import recommendation_store

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


TEST_USER = {"user_id": "spike-test", "company_id": "spike-test", "full_name": "Hosted test"}


def fake_user(x_spike_key: str = Header(default="")):
    """Test-only stand-in for server.current_user."""
    if x_spike_key == "spike-secret":
        return TEST_USER
    raise HTTPException(status_code=401, detail="Sign in first.")


server.app.dependency_overrides[server.current_user] = fake_user
client = TestClient(server.app)

r = client.get("/api/health")
check("health answers without any sign-in and says whether each setting is present, never its value",
      r.status_code == 200 and r.json()["ok"] and "COSMOS_KEY" in r.json()["settings_present"] and all(isinstance(v, bool) for v in r.json()["settings_present"].values()), r.text)
r = client.get("/api/health?deep=1")
check("deep health loads the Orchestrator", r.status_code == 200 and r.json()["orchestrator_import"] == "ok", r.text)

# ---- the evaluate endpoint, with the Orchestrator and the stores faked
saved, calls = [], []


class FakeRecStore:
    pass


class FakeRunStore:
    def add(self, record):
        saved.append(record)
        return record["id"]


def fake_handle(message, state, clients, recommendation_store=None, user_id=None, company_id=None, **kw):
    calls.append((message, user_id, company_id))
    run_log.event("orchestrator", "invented_question_removed", True)
    return {"decision": {"request_type": "new_evaluation", "agents": ["regulations"], "constraints": {}},
            "results": {"regulations": {"status": "ok", "guardrail_status": "passed", "output": {"classification": "limited"}}}, "reply": "The reply."}


orchestrator.handle_message = fake_handle
recommendation_store.CosmosRecommendationStore = FakeRecStore
run_log.CosmosRunStore = FakeRunStore
server.get_clients = lambda: object()

r = client.post("/api/evaluate", json={"message": "A chatbot"})
check("no token and no spike key is refused with 401", r.status_code == 401, r.text)
r = client.post("/api/evaluate", json={"message": "A chatbot"}, headers={"x-spike-key": "wrong"})
check("a wrong spike key is refused", r.status_code == 401, r.text)
r = client.post("/api/evaluate", json={"message": "A chatbot"}, headers={"x-spike-key": "spike-secret"})
body = r.json()
check("the spike key lets a test request through and returns the reply, routing and run id", r.status_code == 200 and body["reply"] == "The reply." and body["agents"] == ["regulations"] and body["run_id"], r.text)
check("the Orchestrator got the message and the (test) user and company", calls[-1] == ("A chatbot", "spike-test", "spike-test"), calls)
check("the request was recorded, with the guardrail event from the Orchestrator", len(saved) == 1 and any(e["rule"] == "invented_question_removed" for e in saved[0]["guardrail_events"]) and saved[0]["user_id"] == "spike-test", saved)
check("an empty message is refused before anything runs", client.post("/api/evaluate", json={"message": ""}, headers={"x-spike-key": "spike-secret"}).status_code == 422)
check("an over-long message is refused", client.post("/api/evaluate", json={"message": "x" * 4001}, headers={"x-spike-key": "spike-secret"}).status_code == 422)

# a failing Orchestrator still leaves a record and gives a clear error
def boom(*a, **k):
    raise RuntimeError("model down")


orchestrator.handle_message = boom
before = len(saved)
r = client.post("/api/evaluate", json={"message": "A chatbot"}, headers={"x-spike-key": "spike-secret"})
check("a failing request returns 502 with the run id, and is recorded with its error",
      r.status_code == 502 and r.json()["detail"]["run_id"] and len(saved) == before + 1 and "model down" in saved[-1]["error"], (r.status_code, r.text))

# a database that can't be reached must not lose the reply
orchestrator.handle_message = fake_handle


class DownRunStore:
    def add(self, record):
        raise ConnectionError("cosmos unreachable")


run_log.CosmosRunStore = DownRunStore
r = client.post("/api/evaluate", json={"message": "A chatbot"}, headers={"x-spike-key": "spike-secret"})
check("if the record can't be saved the reply is still returned, and the problem is reported", r.status_code == 200 and r.json()["reply"] == "The reply." and "cosmos unreachable" in r.json()["record_problem"], r.text)

# the server has no backdoor: with the real sign-in in place, the old header and a missing token are both refused
saved_override = server.app.dependency_overrides.pop(server.current_user)
r = client.post("/api/evaluate", json={"message": "A chatbot"}, headers={"x-spike-key": "spike-secret"})
check("the real sign-in refuses the old spike header (no backdoor)", r.status_code == 401, r.text)
check("the server no longer defines user_or_spike", not hasattr(server, "user_or_spike"))
server.app.dependency_overrides[server.current_user] = saved_override

# ---- the streaming endpoint and Approve
import json as _json
import session_store

SHARED = session_store.OwnedSessionStore()
session_store.SharedContainerSessionStore = lambda: SHARED
run_log.CosmosRunStore = FakeRunStore
server.get_clients = lambda: object()

MODEL = {"scenarios": [
    {"id": "s1", "role": "shortlist", "model_stack": "Model One", "quality_percentile": 90, "yearly_cost_usd": 1200, "latency_ms": 600,
     "latency_means": "time until the first answer token arrives", "verdicts": {"latency": "meets", "cost": "meets"}, "trade_offs": "t", "comparison": "c", "learned": []}],
    "volume_assumption": {"requests_per_month": 20000, "volume_basis": "assumed"}, "overview": "ov", "interpretation": {"feature_category": "chat_assistant"}}
REGS = {"classification": "transparency_obligated", "answer": "Article 50 applies.", "citations": [{"article_number": 50, "paragraph_number": 1}], "disclaimer": "d"}


def streaming_handle(message, state, clients, on_event=print, recommendation_store=None, user_id=None, company_id=None, progress=None, **kw):
    progress({"type": "reading"})
    progress({"type": "plan", "agents": ["regulations", "model_stack_analysis"], "message": "Plan"})
    progress({"type": "agent_started", "agent": "regulations", "label": "the regulations check"})
    progress({"type": "agent_finished", "agent": "regulations", "label": "the regulations check", "status": "ok", "seconds": 1.2})
    progress({"type": "writing"})
    results = {"regulations": {"status": "ok", "output": REGS, "guardrail_status": "passed"}, "model_stack_analysis": {"status": "ok", "output": MODEL}}
    constraints = {"latency_target": "under 800ms", "budget": "$40K per year"}
    return {"decision": {"request_type": "new_evaluation", "agents": list(results), "constraints": constraints}, "results": results, "facts": {"model_stack_analysis": MODEL},
            "reply": "The summary.", "session_state": {"feature_description": "a chatbot", "source_text": message, "constraints": constraints, "results": results,
                                                       "agents_already_run": list(results)}}


def read_events(response):
    return [_json.loads(line[6:]) for line in response.text.split("\n") if line.startswith("data: ")]


orchestrator.handle_message = streaming_handle
saved.clear()
r = client.post("/api/evaluate/stream", json={"message": "A billing chatbot"}, headers={"x-spike-key": "spike-secret"})
events = read_events(r)
check("the stream answers as server-sent events", r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream"), r.headers)
check("progress events arrive in order and the result comes last", [e["type"] for e in events] == ["reading", "plan", "agent_started", "agent_finished", "writing", "result"], [e["type"] for e in events])
res = events[-1]
check("the result carries the page's pieces: banner, cards, chips, summary, session id and run id",
      res["regulatory"]["tone"] == "notice" and len(res["cards"]) == 1 and res["chips"]["latency"] == "under 800ms" and res["summary"] == "The summary." and res["session_id"] and res["run_id"], res.keys())
check("the request was recorded with its session id", len(saved) == 1 and saved[0]["session_id"] == res["session_id"], saved)
check("the session was saved for the user who made it", SHARED.get(res["session_id"], "spike-test") is not None and SHARED.get(res["session_id"], "other-user") is None)
check("an empty message is refused before anything runs", client.post("/api/evaluate/stream", json={"message": ""}, headers={"x-spike-key": "spike-secret"}).status_code == 422)
check("the stream needs sign-in", client.post("/api/evaluate/stream", json={"message": "x"}).status_code == 401)

# a second message in the same session passes the saved state to the Orchestrator
seen_state = []
orig = orchestrator.handle_message
orchestrator.handle_message = lambda message, state, *a, **k: (seen_state.append(state), orig(message, state, *a, **k))[1]
client.post("/api/evaluate/stream", json={"message": "targets: under 500ms", "session_id": res["session_id"]}, headers={"x-spike-key": "spike-secret"})
check("a follow-up message in the same session gets the saved state", seen_state and seen_state[-1] and seen_state[-1].get("feature_description") == "a chatbot", seen_state)
seen_state.clear()
client.post("/api/evaluate/stream", json={"message": "hello", "session_id": "someone-elses-session"}, headers={"x-spike-key": "spike-secret"})
check("a session id that isn't the user's gives the Orchestrator no state", seen_state == [None], seen_state)
orchestrator.handle_message = orig


def failing(*a, **k):
    raise RuntimeError("secret internal detail")


orchestrator.handle_message = failing
before = len(saved)
r = client.post("/api/evaluate/stream", json={"message": "x"}, headers={"x-spike-key": "spike-secret"})
events = read_events(r)
check("a failure ends the stream with a plain error and no internal detail", events[-1]["type"] == "error" and "secret internal detail" not in r.text, events)
check("...and the failed request is still recorded with its error", len(saved) == before + 1 and "secret internal detail" in saved[-1]["error"])
orchestrator.handle_message = streaming_handle

# the result also carries the three verdicts, the guardrail summary and the Act's own wording
import act_text
act_text.quote_citations = lambda citations, *a, **k: [{"label": "Article 50(1)", "title": "Transparency obligations", "text": "Providers shall ensure ...", "source_url": "https://eur-lex/x"}] if citations else []
r = client.post("/api/evaluate/stream", json={"message": "A billing chatbot"}, headers={"x-spike-key": "spike-secret"})
res2 = read_events(r)[-1]
check("the result carries three verdict tiles", [t["key"] for t in res2["verdicts"]] == ["legal", "recommended", "market"], res2.get("verdicts"))
check("the result carries a guardrail summary", set(res2["guardrails"]) == {"checked", "fired", "lines"}, res2.get("guardrails"))
check("the legal block carries the cited provision's wording", res2["regulatory"]["quotes"][0]["label"] == "Article 50(1)" and res2["regulatory"]["quotes"][0]["text"].startswith("Providers shall"), res2["regulatory"])
act_text.quote_citations = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("index down"))
res3 = read_events(client.post("/api/evaluate/stream", json={"message": "A billing chatbot"}, headers={"x-spike-key": "spike-secret"}))[-1]
check("if the wording can't be fetched the result still arrives, without quotes", res3["type"] == "result" and res3["regulatory"]["quotes"] == [], res3.get("regulatory"))

# ---- Approve
import recommendation_store as rs


class FakeRecs:
    def __init__(self):
        self.docs = {}

    def add(self, record):
        self.docs[record["id"]] = record
        return record["id"]

    def get(self, rid):
        return self.docs.get(rid)


RECS = FakeRecs()
recommendation_store.CosmosRecommendationStore = lambda: RECS
sid = res["session_id"]
r = client.post("/api/accept", json={"session_id": sid, "option_id": "s1"}, headers={"x-spike-key": "spike-secret"})
body = r.json()
check("Approve records the chosen option and returns the follow-up dates", r.status_code == 200 and body["chosen"] == "Model One" and [f["day"] for f in body["follow_ups"]] == [30, 60, 90], r.text)
check("the saved recommendation belongs to the user and company", list(RECS.docs.values())[0]["user_id"] == "spike-test" and list(RECS.docs.values())[0]["company_id"] == "spike-test")
check("the confirmation tells the PM to report what happened later", "report what actually happened" in body["message"], body["message"])
r2 = client.post("/api/accept", json={"session_id": sid, "option_id": "s1"}, headers={"x-spike-key": "spike-secret"})
check("approving the same option twice does not record it twice", r2.status_code == 200 and len(RECS.docs) == 1 and "already recorded" in r2.json()["message"], r2.text)
check("an unknown option is refused with a plain message", client.post("/api/accept", json={"session_id": sid, "option_id": "nope"}, headers={"x-spike-key": "spike-secret"}).status_code == 400)
check("an unknown session is a 404", client.post("/api/accept", json={"session_id": "missing", "option_id": "s1"}, headers={"x-spike-key": "spike-secret"}).status_code == 404)
check("Approve needs sign-in", client.post("/api/accept", json={"session_id": sid, "option_id": "s1"}).status_code == 401)

# a prohibited feature cannot be approved
prohibited = dict(streaming_handle("p", None, None, progress=lambda e: None))
prohibited["results"] = {**prohibited["results"], "regulations": {"status": "ok", "output": {**REGS, "classification": "prohibited"}}}
prohibited["session_state"] = {**prohibited["session_state"], "results": prohibited["results"]}
orchestrator.handle_message = lambda *a, **k: prohibited
pres = read_events(client.post("/api/evaluate/stream", json={"message": "p"}, headers={"x-spike-key": "spike-secret"}))[-1]
check("a prohibited feature streams a blocked banner with no cards and no accept", pres["regulatory"]["tone"] == "blocked" and pres["cards"] == [] and pres["can_accept"] is False, pres.keys())
refused = client.post("/api/accept", json={"session_id": pres["session_id"], "option_id": "s1"}, headers={"x-spike-key": "spike-secret"})
check("...and Approve refuses it even if the call is made directly", refused.status_code == 400 and "prohibited" in refused.json()["detail"], refused.text)

# ---- the Dashboard endpoint
class FakeTimingRunStore:
    def add(self, record):
        return record["id"]

    def list_run_timings(self, user_id):
        return [{"user_id": user_id, "request_type": "new_evaluation", "duration_seconds": 40.0}]


run_log.CosmosRunStore = FakeTimingRunStore
RECS2 = FakeRecs()
RECS2.list_all = lambda: [r for r in RECS.docs.values()]
recommendation_store.CosmosRecommendationStore = lambda: RECS2
r = client.get("/api/dashboard", headers={"x-spike-key": "spike-secret"})
check("the dashboard endpoint needs sign-in", client.get("/api/dashboard").status_code == 401)
check("the dashboard answers with stats, rows and the measured median time", r.status_code == 200 and r.json()["stats"]["median_seconds_to_result"] == 40.0 and "evaluations" in r.json(), r.text[:300])
check("the dashboard shows the recommendation this user approved earlier in the test", r.json()["stats"]["evaluations_saved"] == 1, r.json()["stats"])
RECS2.list_all = lambda: list(RECS.docs.values()) + [{"id": "x", "type": "accepted_recommendation", "company_id": "another-company", "user_id": "u", "accepted_at": "2026-10-01T00:00:00+00:00",
                                                       "chosen": {"label": "Not yours", "forecast": {}}, "feature": {}, "follow_ups": [], "outcomes": []}]
other = client.get("/api/dashboard", headers={"x-spike-key": "spike-secret"}).json()
check("another company's recommendation never appears on this user's dashboard", other["stats"]["evaluations_saved"] == 1 and all(e["model"] != "Not yours" for e in other["evaluations"]), other["stats"])

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
