"""
Offline tests for recommendation_store.py: working out which option the PM meant, deciding whether there
is anything to accept, the saved record, and the stores. Local JSON file only; no Cosmos, no network.

Run: python test_recommendation_store.py
"""
import copy
import os
import warnings
from datetime import datetime, timedelta, timezone
warnings.filterwarnings("ignore")

import recommendation_store as rs

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def option(oid, label, role="shortlist", cost=1200.0, lat=764, pct=69.0, verdicts=None):
    return {"id": oid, "role": role, "model_stack": label, "yearly_cost_usd": cost, "latency_ms": lat,
            "latency_means": "time until the first answer token arrives", "first_answer_ms": lat, "full_response_ms": 3900,
            "quality_percentile": pct, "input_price_per_1m_tokens": 2.0, "output_price_per_1m_tokens": 12.0,
            "verdicts": verdicts or {"latency": "meets", "cost": "meets"}}


OPTIONS = [
    dict(option("opt57", "OpenAI gpt-5.6-terra (Non-reasoning)"), raw_forecast={"latency_ms": 700, "yearly_cost_usd": 1100.0}, learned=["Latency: 3 reports from your company put this setting at 1.09x the forecast."]),
    option("opt32", "Anthropic Claude Haiku 4.5 (Non-reasoning)", cost=540.0, lat=472, pct=59.1),
    option("opt39", "OpenAI gpt-6-luna (Non-reasoning)", role="near_miss", cost=54.0, lat=862, pct=64.5, verdicts={"latency": "misses", "cost": "meets"}),
    option("opt40", "Anthropic Claude Sonnet 5.5 (Adaptive Reasoning, Low Effort)", role="near_miss", cost=1100.0, lat=917, pct=None, verdicts={"latency": "misses", "cost": "meets"}),
]
SESSION = {
    "feature_description": "A chatbot that answers billing questions.",
    "source_text": "We're adding a chatbot that answers billing questions. 90% accurate, under 800ms, $40K a year.",
    "constraints": {"accuracy_target": "90%", "latency_target": "under 800ms", "budget": "$40K annual"},
    "results": {
        "regulations": {"status": "ok", "output": {"classification": "transparency_obligated", "applicability_status": "From 2 August 2026.",
                                                   "derogation_assessment": {"outcome": "not_relevant", "reason": ""}}},
        "model_stack_analysis": {"status": "ok", "output": {
            "scenarios": OPTIONS, "volume_assumption": {"requests_per_month": 20000, "volume_basis": "assumed"},
            "caveats": ["This feature needs the model to read images."],
            "interpretation": {"source": "model", "reply_style": "streamed_chat", "feature_category": "chat_assistant"}}},
    },
}

# ---- which option did the PM mean?
def pick(text):
    o, problem = rs.resolve_option(text, OPTIONS)
    return (o["id"] if o else None), problem


check("resolve: 'option 2' is the second shortlist option", pick("option 2")[0] == "opt32", pick("option 2"))
check("resolve: a bare '1' is the first shortlist option", pick("1")[0] == "opt57")
check("resolve: 'the second one' (a word) works", pick("the second one")[0] == "opt32")
check("resolve: 'go with the first' works", pick("go with the first")[0] == "opt57")
check("resolve: a model name picks it ('Haiku')", pick("Haiku")[0] == "opt32", pick("Haiku"))
check("resolve: 'accept the Terra one please' picks it", pick("accept the Terra one please")[0] == "opt57", pick("accept the Terra one please"))
check("resolve: a near miss can be chosen by name", pick("gpt-6 luna")[0] == "opt39", pick("gpt-6 luna"))
check("resolve: a number never picks a near miss (they aren't on the numbered shortlist)", pick("option 3")[0] is None and "only showed 2" in pick("option 3")[1], pick("option 3"))
check("resolve: option 0 is rejected", pick("option 0")[0] is None)
check("resolve: a version number like 4.5 is not mistaken for an option number", pick("Haiku 4.5")[0] == "opt32", pick("Haiku 4.5"))
check("resolve: an internal id works", pick("opt39")[0] == "opt39")
check("resolve: nothing to go on asks which one", pick("accept")[0] is None and "Which option" in pick("accept")[1], pick("accept"))
check("resolve: empty or None asks which one", pick("")[0] is None and pick(None)[0] is None)
check("resolve: an unknown name says it couldn't match", pick("Mistral")[0] is None and "couldn't match" in pick("Mistral")[1], pick("Mistral"))
check("resolve: a name that fits two options asks which", pick("non-reasoning")[0] is None and "more than one" in pick("non-reasoning")[1], pick("non-reasoning"))
check("resolve: the question names the options it could be", "terra" in pick("non-reasoning")[1] and "Haiku" in pick("non-reasoning")[1])
two_efforts = [option("a", "Acme X (Low)"), option("b", "Acme X (High)")]
check("resolve: two effort settings of one model are ambiguous by model name alone", rs.resolve_option("Acme X", two_efforts)[0] is None)
check("resolve: ...and distinct once the effort is named", rs.resolve_option("Acme X Low", two_efforts)[0]["id"] == "a")

# ---- is there anything to accept?
check("can_accept: a normal session is acceptable", rs.can_accept(SESSION) == (True, None))
check("can_accept: no session at all", rs.can_accept(None)[0] is False and "nothing to accept" in rs.can_accept(None)[1])
check("can_accept: no model options yet", rs.can_accept({"results": {}})[0] is False)
failed = copy.deepcopy(SESSION)
failed["results"]["model_stack_analysis"] = {"status": "error", "error": "x"}
check("can_accept: a failed model stack run leaves nothing to accept", rs.can_accept(failed)[0] is False)
prohibited = copy.deepcopy(SESSION)
prohibited["results"]["regulations"]["output"]["classification"] = "prohibited"
ok, why = rs.can_accept(prohibited)
check("can_accept: a prohibited feature can't be accepted, and the reason says why", ok is False and "prohibited" in why, why)
regs_failed = copy.deepcopy(SESSION)
regs_failed["results"]["regulations"] = {"status": "error", "error": "429"}
check("can_accept: a failed legal check doesn't block the choice (it is recorded instead)", rs.can_accept(regs_failed) == (True, None))

# ---- the saved record
rec = rs.build_acceptance_record(SESSION, OPTIONS[1], "priyanka", "acme", NOW)
check("record: it is typed so it can be told apart from older documents", rec["type"] == "accepted_recommendation" and rec["status"] == "awaiting_outcome")
check("record: who and when", rec["user_id"] == "priyanka" and rec["company_id"] == "acme" and rec["accepted_at"] == NOW.isoformat())
check("record: the PM's own words and the feature category are kept", "billing questions" in rec["feature"]["source_text"] and rec["feature"]["category"] == "chat_assistant" and rec["feature"]["reply_style"] == "streamed_chat")
check("record: the chosen option is a snapshot of what was forecast", rec["chosen"]["label"].startswith("Anthropic Claude Haiku") and rec["chosen"]["forecast"]["yearly_cost_usd"] == 540.0 and rec["chosen"]["forecast"]["latency_ms"] == 472 and rec["chosen"]["forecast"]["quality_percentile"] == 59.1)
check("record: the forecast says what its latency means", "first answer token" in rec["chosen"]["forecast"]["latency_means"])
check("record: the prices behind the cost are kept (they change)", rec["chosen"]["forecast"]["input_price_per_1m_tokens"] == 2.0)
check("record: the usage the cost assumed is kept", rec["usage_assumption"]["requests_per_month"] == 20000 and rec["usage_assumption"]["volume_basis"] == "assumed")
check("record: the legal result at the time is kept", rec["legal"]["status"] == "checked" and rec["legal"]["classification"] == "transparency_obligated")
check("record: the warnings shown are kept", rec["caveats"] == ["This feature needs the model to read images."])
check("record: the targets the PM gave are kept", rec["constraints_given"]["latency_target"] == "under 800ms")
check("record: the other options shown are listed by name, not including the chosen one", len(rec["alternatives_shown"]) == 3 and rec["chosen"]["label"] not in rec["alternatives_shown"], rec["alternatives_shown"])
check("record: it starts with no outcomes", rec["outcomes"] == [])
check("record: three follow-ups at 30, 60 and 90 days, all pending", [f["day"] for f in rec["follow_ups"]] == [30, 60, 90] and all(f["status"] == "pending" for f in rec["follow_ups"]))
check("record: the due dates are counted from the day of choosing", rec["follow_ups"][0]["due"] == (NOW + timedelta(days=30)).isoformat() and rec["follow_ups"][2]["due"] == (NOW + timedelta(days=90)).isoformat())
check("record: an option that met everything has no accepted misses", rec["chosen"]["accepted_misses"] == [])
risky = rs.build_acceptance_record(SESSION, OPTIONS[2], "p", "c", NOW)
check("record: choosing a near miss records exactly what it misses", risky["chosen"]["accepted_misses"] == ["latency"], risky["chosen"]["accepted_misses"])
check("record: ids are unique", rs.build_acceptance_record(SESSION, OPTIONS[1], "p", "c", NOW)["id"] != rec["id"])
no_regs = copy.deepcopy(SESSION)
del no_regs["results"]["regulations"]
check("record: no legal check run -> says not_run, not an invented result", rs.build_acceptance_record(no_regs, OPTIONS[1], "p", "c", NOW)["legal"] == {"status": "not_run", "classification": None})
check("record: a failed legal check is recorded as failed", rs.build_acceptance_record(regs_failed, OPTIONS[1], "p", "c", NOW)["legal"]["status"] == "check_failed")
check("record: building it doesn't change the session", SESSION["results"]["model_stack_analysis"]["output"]["scenarios"][0]["id"] == "opt57")

# ---- the confirmation the PM reads
msg = rs.confirmation_message(rec)
check("message: names the choice and the forecast", "you chose Anthropic Claude Haiku 4.5" in msg and "$540" in msg and "472ms" in msg and "59th percentile" in msg, msg)
check("message: gives the three follow-up dates", "30 days (2026-11-02)" in msg and "60 days (2026-12-02)" in msg and "90 days (2027-01-01)" in msg, msg)
check("message: says the legal result", "transparency_obligated" in msg)
check("message: tells the PM to report what happened once launched, since reporting now exists", "report what actually happened" in msg and "isn't available yet" not in msg)
check("message: a near miss says what was accepted", "knowing it misses your latency target" in rs.confirmation_message(risky))
check("message: a missing legal check is said plainly", "legal check did not complete" in rs.confirmation_message(rs.build_acceptance_record(regs_failed, OPTIONS[1], "p", "c", NOW)))

check("record: the benchmark forecast before any learned correction is kept, so corrections don't compound", rs.build_acceptance_record(SESSION, OPTIONS[0], "p", "c", NOW)["chosen"]["forecast_raw"] == {"latency_ms": 700, "yearly_cost_usd": 1100.0})
check("record: the learned sentences the PM was shown are kept", rs.build_acceptance_record(SESSION, OPTIONS[0], "p", "c", NOW)["chosen"]["learned_shown"][0].startswith("Latency: 3 reports"))
check("record: an option with no learning has an empty list, not a crash", rs.build_acceptance_record(SESSION, OPTIONS[1], "p", "c", NOW)["chosen"]["learned_shown"] == [] and rs.build_acceptance_record(SESSION, OPTIONS[1], "p", "c", NOW)["chosen"]["forecast_raw"] is None)

# ---- stores
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recommendations_test.json")
if os.path.exists(path):
    os.remove(path)
store = rs.LocalRecommendationStore(path)
try:
    check("store: an empty store has nothing in it", store.list_all() == [] and store.get("nope") is None)
    rid = store.add(rec)
    check("store: a saved record comes back unchanged", store.get(rid)["chosen"]["label"] == rec["chosen"]["label"] and store.get(rid) == rec)
    check("store: it is listed under its user and not another", len(store.for_user("priyanka")) == 1 and store.for_user("someone-else") == [])
    check("store: nothing is due the day it was chosen", store.due_follow_ups(NOW) == [])
    due = store.due_follow_ups(NOW + timedelta(days=31))
    check("store: the 30-day follow-up is due after 30 days", len(due) == 1 and due[0][1]["day"] == 30, due)
    check("store: after 91 days all three are due", len(store.due_follow_ups(NOW + timedelta(days=91))) == 3)
    rec["status"] = "outcome_reported"
    store.update(rec)
    check("store: a recommendation that is no longer waiting has no due follow-ups", store.due_follow_ups(NOW + timedelta(days=91)) == [])
    rec["status"] = "awaiting_outcome"
    rec["follow_ups"][0]["status"] = "done"
    store.update(rec)
    check("store: a follow-up marked done is not due again", [f["day"] for _, f in store.due_follow_ups(NOW + timedelta(days=91))] == [60, 90])

    # documents the first Model Stack script wrote have no type and must be ignored
    data = store._load()
    data["old-doc"] = {"id": "old-doc", "agent": "model_stack_analysis", "outcome": None}
    store._save(data)
    check("store: older documents with no type are ignored", all(r["id"] != "old-doc" for r in store.list_all()))

    # ---- the whole step
    store2 = rs.LocalRecommendationStore(path + ".2")
    result = rs.accept_option(SESSION, "option 2", "priyanka", "acme", store2, NOW)
    check("accept: a clear choice is recorded and confirmed", result["ok"] and result["record"]["chosen"]["option_id"] == "opt32" and "Recorded" in result["message"], result)
    check("accept: it is really in the store", store2.get(result["record"]["id"]) is not None)
    check("accept: the session is told what was accepted, so it can't be recorded twice", result["session_update"]["accepted"] == [{"option_id": "opt32", "record_id": result["record"]["id"]}], result["session_update"])
    again = copy.deepcopy(SESSION)
    again.update(result["session_update"])
    second = rs.accept_option(again, "Haiku", "priyanka", "acme", store2, NOW)
    check("accept: choosing the same option again changes nothing and says so", second["ok"] and "already recorded" in second["message"] and len(store2.list_all()) == 1 and second["record"]["id"] == result["record"]["id"], second)
    other = rs.accept_option(again, "option 1", "priyanka", "acme", store2, NOW)
    check("accept: a different option in the same conversation is a second record", other["ok"] and len(store2.list_all()) == 2 and len(other["session_update"]["accepted"]) == 2)
    unclear = rs.accept_option(SESSION, "non-reasoning", "p", "c", rs.LocalRecommendationStore(path + ".3"), NOW)
    check("accept: an ambiguous choice saves nothing and asks", unclear["ok"] is False and unclear["record"] is None and "more than one" in unclear["message"])
    blocked = rs.accept_option(prohibited, "option 1", "p", "c", rs.LocalRecommendationStore(path + ".4"), NOW)
    check("accept: a prohibited feature saves nothing", blocked["ok"] is False and "prohibited" in blocked["message"])
    nothing = rs.accept_option(None, "option 1", "p", "c", rs.LocalRecommendationStore(path + ".5"), NOW)
    check("accept: no conversation to accept from is handled, not a crash", nothing["ok"] is False)
finally:
    for suffix in ("", ".2", ".3", ".4", ".5"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
