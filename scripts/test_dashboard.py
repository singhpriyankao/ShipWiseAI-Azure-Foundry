"""
Offline tests for dashboard.py (the numbers and rows the Dashboard shows). Synthetic records; no network.

Run: python test_dashboard.py
"""
import warnings
from datetime import datetime, timedelta, timezone
warnings.filterwarnings("ignore")

import dashboard

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


NOW = datetime(2026, 11, 20, tzinfo=timezone.utc)


def rec(rid, company="northwind-support", accepted_days_ago=10, outcomes=None, sample=False, latency_target="under 800ms", budget="$40K per year",
        label="Model One", misses=None, followups_done=()):
    accepted = NOW - timedelta(days=accepted_days_ago)
    return {"id": rid, "type": "accepted_recommendation", "user_id": "amara", "company_id": company, "accepted_at": accepted.isoformat(), "sample": sample,
            "feature": {"description": "A billing chatbot for SaaS customers", "source_text": "x"},
            "constraints_given": {"latency_target": latency_target, "budget": budget},
            "chosen": {"label": label, "accepted_misses": misses or [], "forecast": {"yearly_cost_usd": 1200, "latency_ms": 600, "latency_means": "time until the first answer token arrives (the reply starts appearing)"}},
            "legal": {"classification": "transparency_obligated"},
            "follow_ups": [{"day": d, "due": (accepted + timedelta(days=d)).isoformat(), "status": "reported" if d in followups_done else "pending"} for d in (30, 60, 90)],
            "outcomes": outcomes or []}


def outcome(latency_ms=None, yearly=None, when="2026-11-10T00:00:00+00:00"):
    fields = {}
    if latency_ms is not None:
        fields["latency"] = {"value_ms": latency_ms, "kind": "first_answer", "statistic": "median"}
    if yearly is not None:
        fields["cost"] = {"yearly_usd": yearly}
    return {"confirmed_by_pm": True, "reported_at": when, "fields": fields}


# ---- status of a row
fresh = dashboard.evaluation_row(rec("a", accepted_days_ago=10), NOW)
check("a recent approval is awaiting, with its next follow-up (day 30)", fresh["status"] == "awaiting" and fresh["next_follow_up"]["day"] == 30, fresh)
due = dashboard.evaluation_row(rec("b", accepted_days_ago=35), NOW)
check("an approval past its 30-day date with no report is feedback due", due["status"] == "feedback_due" and due["next_follow_up"]["day"] == 30, due)
rep = dashboard.evaluation_row(rec("c", accepted_days_ago=35, outcomes=[outcome(latency_ms=700, yearly=1000)]), NOW)
check("a confirmed report makes it reported, and it is no longer due", rep["status"] == "reported" and rep["outcome"] is not None, rep)
check("reported results that meet the PM's own targets read as met", rep["outcome"]["verdict"] == "met" and rep["outcome"]["targets"] == {"latency": "met", "cost": "met"}, rep["outcome"])
missed = dashboard.evaluation_row(rec("d", accepted_days_ago=35, outcomes=[outcome(latency_ms=950, yearly=1000)]), NOW)
check("a reported latency over the target reads as missed, with which target", missed["outcome"]["verdict"] == "missed" and missed["outcome"]["targets"]["latency"] == "missed" and missed["outcome"]["targets"]["cost"] == "met", missed["outcome"])
unconfirmed = rec("e", accepted_days_ago=35, outcomes=[{**outcome(latency_ms=700), "confirmed_by_pm": False}])
check("a report the PM never confirmed doesn't count", dashboard.evaluation_row(unconfirmed, NOW)["status"] == "feedback_due")
other_model = rec("f", accepted_days_ago=35, outcomes=[{**outcome(latency_ms=700), "model_differs": True}])
check("a report about a different model doesn't count either", dashboard.evaluation_row(other_model, NOW)["status"] == "feedback_due")
no_target = dashboard.evaluation_row(rec("g", accepted_days_ago=35, latency_target=None, budget=None, outcomes=[outcome(latency_ms=700)]), NOW)
check("a report with no target to compare to has no verdict (not 'met')", no_target["status"] == "reported" and no_target["outcome"]["verdict"] is None, no_target["outcome"])
check("an accepted miss is carried to the row", dashboard.evaluation_row(rec("h", misses=["latency"]), NOW)["accepted_misses"] == ["latency"])
check("latency kind is plain words", fresh["forecast"]["latency_kind"] == "to the first words")
check("a long feature description is shortened", len(dashboard.evaluation_row({**rec("i"), "feature": {"description": "x " * 200}}, NOW)["feature"]) <= 90)
check("sample data keeps its flag", dashboard.evaluation_row(rec("j", sample=True), NOW)["sample"] is True and fresh["sample"] is False)

# ---- the whole dashboard
records = [rec("a", accepted_days_ago=10, label="Model One"),
           rec("b", accepted_days_ago=35, label="Model Two"),
           rec("c", accepted_days_ago=40, outcomes=[outcome(latency_ms=700, yearly=1000)], label="Model One", sample=True),
           rec("d", accepted_days_ago=45, outcomes=[outcome(latency_ms=950, yearly=1000)], label="Model One", sample=True),
           rec("z", company="other-company", accepted_days_ago=5, label="Elsewhere"),
           {"id": "s", "type": "session", "company_id": "northwind-support"}]
runs = [{"user_id": "amara", "request_type": "new_evaluation", "duration_seconds": 20.0, "agent_results": {"regulations": {"status": "ok"}}},
        {"user_id": "amara", "request_type": "new_evaluation", "duration_seconds": 80.0, "agent_results": {"regulations": {"status": "ok"}}},
        {"user_id": "amara", "request_type": "new_evaluation", "duration_seconds": 50.0, "agent_results": {"regulations": {"status": "ok"}}},
        {"user_id": "amara", "request_type": "new_evaluation", "duration_seconds": 5.0, "error": "boom", "agent_results": {}},
        {"user_id": "amara", "request_type": "accept_option", "duration_seconds": 1.0, "agent_results": {}},
        {"user_id": "someone-else", "request_type": "new_evaluation", "duration_seconds": 999.0, "agent_results": {"regulations": {"status": "ok"}}}]
d = dashboard.build_dashboard(records, runs, "amara", "northwind-support", NOW)
s = d["stats"]
check("only this company's recommendations are counted (not other companies', not sessions)", s["evaluations_saved"] == 4 and all(r["model"] != "Elsewhere" for r in d["evaluations"]), s)
check("awaiting feedback counts only those due", s["awaiting_feedback"] == 1, s)
check("Decision Quality is the share that met their own targets (1 of 2 reported)", s["decision_quality"] == {"rate": 0.5, "met_all": 1, "comparable": 2}, s["decision_quality"])
check("median time to a result is measured from this user's successful evaluations only", s["median_seconds_to_result"] == 50.0 and s["runs_measured"] == 3, s)
check("newest approval first", [r["id"] for r in d["evaluations"]] == ["a", "b", "c", "d"], [r["id"] for r in d["evaluations"]])
check("the action needed is the first feedback-due item", d["action_needed"]["id"] == "b")
check("most used models are counted", d["top_models"][0] == {"model": "Model One", "count": 3}, d["top_models"])
check("the dashboard knows it holds sample data", d["has_sample_data"] is True)

empty = dashboard.build_dashboard([], [], "new-user", "new-co", NOW)
check("a brand-new user gets zeros and no invented figures",
      empty["stats"]["evaluations_saved"] == 0 and empty["stats"]["decision_quality"]["rate"] is None and empty["stats"]["median_seconds_to_result"] is None and empty["evaluations"] == [] and empty["action_needed"] is None and empty["has_sample_data"] is False, empty)
no_reports = dashboard.build_dashboard([rec("a")], [], "amara", "northwind-support", NOW)
check("Decision Quality is empty (not zero) until something has been reported", no_reports["stats"]["decision_quality"]["rate"] is None and no_reports["stats"]["decision_quality"]["comparable"] == 0)

light = [{"user_id": "amara", "request_type": "new_evaluation", "duration_seconds": 30.0}, {"user_id": "amara", "request_type": "new_evaluation", "duration_seconds": 60.0, "error": None}]
check("the light run listing (no agent outputs) is counted when it had no error", dashboard.build_dashboard([], light, "amara", "n", NOW)["stats"]["median_seconds_to_result"] == 45.0)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
