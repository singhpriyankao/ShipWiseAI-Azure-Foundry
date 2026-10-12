"""
Offline tests that the Orchestrator routes an outcome report to the report step and continues a half-finished
one in code, without asking the parse model again. The parse step and the extraction model
are fakes; the report logic itself is tested in test_outcome_report.py.

Run: python test_report_flow.py
"""
import json
import os
import warnings
warnings.filterwarnings("ignore")

import orchestrator as orch
import recommendation_store as rs

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


TEXT = "We shipped the Haiku option last month. Accuracy is about 82% on 200 labelled tickets and p95 latency is 1.4 seconds."
EXTRACTED = {"accuracy": {"value_percent": 82, "quote": "Accuracy is about 82% on 200 labelled tickets", "how_measured": "200 labelled tickets"},
             "latency": {"value_ms": 1400, "quote": "p95 latency is 1.4 seconds", "kind": "unspecified", "statistic": "p95"},
             "model_used": {"name": "Haiku", "quote": "We shipped the Haiku option"}, "summary": "ok"}


class FakeModel:
    def __init__(self):
        self.calls = 0
        outer = self

        class _C:
            def create(self, **kw):
                outer.calls += 1
                msg = type("M", (), {"content": json.dumps(EXTRACTED)})
                return type("R", (), {"choices": [type("C", (), {"message": msg})]})
        self.chat = type("Chat", (), {"completions": _C()})


def record():
    return {"id": "rec1", "type": "accepted_recommendation", "user_id": "priyanka", "company_id": "acme", "accepted_at": "2026-10-20T12:00:00+00:00", "status": "awaiting_outcome",
            "feature": {"description": "A tool that tags tickets", "category": "classification_tagging"},
            "chosen": {"option_id": "o1", "label": "Anthropic Claude Haiku 4.5 (Non-reasoning)", "forecast": {"yearly_cost_usd": 540.0, "latency_ms": 472, "latency_means": "time until the first answer token arrives"}},
            "follow_ups": [{"day": d, "due": "2026-11-19T12:00:00+00:00", "status": "pending"} for d in (30, 60, 90)], "outcomes": []}


model = FakeModel()
clients = type("C", (), {"model_client": model})()
parse_calls = []


def fake_parse(request_type):
    def parse(model_client, message, session_state=None, max_attempts=2):
        parse_calls.append(message)
        return {"request_type": request_type, "feature_description": None, "requested_agents": None,
                "constraints": {"accuracy_target": None, "latency_target": None, "budget": None}, "clarifying_question": None, "chosen_option": None}
    return parse


path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_flow_test.json")
real_parse = orch.parse_request
try:
    if os.path.exists(path):
        os.remove(path)
    store = rs.LocalRecommendationStore(path)
    store.add(record())

    d = orch.decide({"request_type": "report_outcome", "feature_description": None, "requested_agents": None, "constraints": {"accuracy_target": None, "latency_target": None, "budget": None},
                     "clarifying_question": None, "chosen_option": None}, None)
    check("decide: a report needs no saved session and runs no agents", d["request_type"] == "report_outcome" and d["agents"] == [] and not d["ask_user"], d)
    check("parse validation: a report's figures are results, not targets, so no constraints are passed on", orch._validate_parse({"request_type": "report_outcome", "constraints": {"latency_target": "about 1.1s", "budget": "$400 a month"}})["constraints"] == {"accuracy_target": None, "latency_target": None, "budget": None})
    check("parse validation: other request types keep their constraints", orch._validate_parse({"request_type": "provide_constraints", "constraints": {"latency_target": "500ms"}})["constraints"]["latency_target"] == "500ms")
    check("the request type is known", "report_outcome" in orch.REQUEST_TYPES)

    orch.parse_request = fake_parse("report_outcome")
    out = orch.handle_message(TEXT, None, clients, on_event=lambda m: None, recommendation_store=store, user_id="priyanka")
    check("report: it is read back for confirmation", "Here is what I understood about Anthropic Claude Haiku 4.5" in out["reply"] and 'Reply "yes"' in out["reply"], out["reply"])
    check("report: nothing is saved yet", store.get("rec1")["outcomes"] == [])
    check("report: the half-finished report is kept in the session", out["session_state"]["pending_outcome"]["stage"] == "confirm")
    check("report: the parse model was asked once, to recognise the report", parse_calls == [TEXT])

    parse_calls.clear()
    orch.parse_request = fake_parse("out_of_scope")  # would route "yes" wrongly if the parse model were consulted
    done = orch.handle_message("yes", out["session_state"], clients, on_event=lambda m: None, recommendation_store=store, user_id="priyanka")
    check("confirm: 'yes' is handled in code, without asking the parse model", parse_calls == [] and "Saved your report" in done["reply"], (parse_calls, done["reply"]))
    check("confirm: the report is on the record, confirmed", len(store.get("rec1")["outcomes"]) == 1 and store.get("rec1")["outcomes"][0]["fields"]["accuracy"]["value_percent"] == 82)
    check("confirm: the session no longer carries a pending report", "pending_outcome" not in done["session_state"])
    check("confirm: the outcome is returned for the caller", done["saved_outcome"]["confirmed_by_pm"] is True)

    orch.parse_request = fake_parse("report_outcome")
    again = orch.handle_message(TEXT, None, clients, on_event=lambda m: None, recommendation_store=store, user_id="priyanka")
    cancelled = orch.handle_message("cancel", again["session_state"], clients, on_event=lambda m: None, recommendation_store=store, user_id="priyanka")
    check("cancel: nothing more is saved", "Nothing was saved" in cancelled["reply"] and len(store.get("rec1")["outcomes"]) == 1)

    corrected = orch.handle_message("no, the accuracy was 78%", again["session_state"], clients, on_event=lambda m: None, recommendation_store=store, user_id="priyanka")
    check("correct: a correction is read back again, not saved", "Here is what I understood" in corrected["reply"] and len(store.get("rec1")["outcomes"]) == 1)

    nostore = orch.handle_message(TEXT, None, clients, on_event=lambda m: None)
    check("no store: it says it can't save instead of pretending", nostore["reply"] == orch.NO_STORE_MSG)

    empty = rs.LocalRecommendationStore(path + ".empty")
    nothing = orch.handle_message(TEXT, None, clients, on_event=lambda m: None, recommendation_store=empty, user_id="priyanka")
    check("nothing waiting: a report gets a plain answer", "waiting for a report" in nothing["reply"], nothing["reply"])
    other = orch.handle_message(TEXT, None, clients, on_event=lambda m: None, recommendation_store=store, user_id="someone-else")
    check("another user's recommendations are never used for a report", "waiting for a report" in other["reply"])
finally:
    orch.parse_request = real_parse
    for suffix in ("", ".empty"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
