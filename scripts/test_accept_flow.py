"""
Offline tests that the Orchestrator routes an "accept" message to the accept step and records the choice
. The parse step is replaced by a fake that returns what the real one should; the accept
logic itself is tested in test_recommendation_store.py.

Run: python test_accept_flow.py
"""
import copy
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


def option(oid, label, role="shortlist", verdicts=None):
    return {"id": oid, "role": role, "model_stack": label, "yearly_cost_usd": 540.0, "latency_ms": 472, "latency_means": "time until the first answer token arrives",
            "first_answer_ms": 472, "full_response_ms": 2400, "quality_percentile": 59.1, "input_price_per_1m_tokens": 1.0, "output_price_per_1m_tokens": 5.0,
            "verdicts": verdicts or {"latency": "meets", "cost": "meets"}}


SESSION = {
    "feature_description": "A chatbot that answers billing questions.", "source_text": "We want a billing chatbot.",
    "constraints": {"accuracy_target": None, "latency_target": "under 800ms", "budget": "$40K annual"},
    "results": {
        "regulations": {"status": "ok", "output": {"classification": "transparency_obligated", "applicability_status": "x", "derogation_assessment": None}},
        "model_stack_analysis": {"status": "ok", "output": {
            "scenarios": [option("o1", "OpenAI gpt-5.6-terra (Non-reasoning)"), option("o2", "Anthropic Claude Haiku 4.5 (Non-reasoning)")],
            "volume_assumption": {"requests_per_month": 20000}, "caveats": [],
            "interpretation": {"feature_category": "chat_assistant", "reply_style": "streamed_chat"}}},
    },
}


def parse_for(request_type, chosen=None):
    def fake(model_client, message, session_state=None, max_attempts=2):
        return {"request_type": request_type, "feature_description": None, "requested_agents": None,
                "constraints": {"accuracy_target": None, "latency_target": None, "budget": None},
                "clarifying_question": None, "chosen_option": chosen}
    return fake


clients = type("C", (), {"model_client": None})()
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accept_flow_test.json")
real_parse, real_run_agents = orch.parse_request, orch.run_agents
agent_calls = []
orch.run_agents = lambda *a, **k: agent_calls.append(a) or {}

try:
    # ---- the routing decision
    d = orch.decide({"request_type": "accept_option", "feature_description": None, "requested_agents": None, "constraints": {"accuracy_target": None, "latency_target": None, "budget": None},
                     "clarifying_question": None, "chosen_option": "option 2"}, SESSION)
    check("decide: an accept with a saved session runs no agents and carries the PM's words", d["request_type"] == "accept_option" and d["agents"] == [] and d["chosen_option"] == "option 2", d)
    d = orch.decide({"request_type": "accept_option", "feature_description": None, "requested_agents": None, "constraints": {"accuracy_target": None, "latency_target": None, "budget": None},
                     "clarifying_question": None, "chosen_option": "1"}, None)
    check("decide: an accept with no saved session becomes 'nothing to accept', not a crash", d["request_type"] == "unclear" and d["ask_user"] and "nothing to accept" in d["message_to_user"], d)
    check("decide: the new request type is known", "accept_option" in orch.REQUEST_TYPES)
    check("parse validation keeps the words naming the option", orch._validate_parse({"request_type": "accept_option", "chosen_option": "  Haiku  "})["chosen_option"] == "Haiku")
    check("parse validation: no words, no option", orch._validate_parse({"request_type": "accept_option", "chosen_option": ""})["chosen_option"] is None)

    check("words naming the option must be the PM's own: a copy of the session note is dropped", orch._confirm_chosen_words({"chosen_option": "a chatbot that answers billing questions"}, "yes, accept it")["chosen_option"] is None)
    check("words naming the option: a verbatim excerpt is kept, whatever the spacing and case", orch._confirm_chosen_words({"chosen_option": "Option  2"}, "I'll go with option 2")["chosen_option"] == "Option  2")
    check("words naming the option: none stays none", orch._confirm_chosen_words({"chosen_option": None}, "anything")["chosen_option"] is None)
    check("words naming the option: other parse fields are untouched", orch._confirm_chosen_words({"request_type": "accept_option", "chosen_option": "zzz"}, "option 1")["request_type"] == "accept_option")

    check("words naming the option: pronouns and filler ('it', 'that one') name nothing and are dropped", orch._confirm_chosen_words({"chosen_option": "it"}, "yes, accept it")["chosen_option"] is None and orch._confirm_chosen_words({"chosen_option": "that one"}, "take that one")["chosen_option"] is None)
    check("words naming the option: 'the first one' still names an option", orch._confirm_chosen_words({"chosen_option": "the first one"}, "take the first one")["chosen_option"] == "the first one")

    # ---- the whole message
    store = rs.LocalRecommendationStore(path)
    orch.parse_request = parse_for("accept_option", "option 2")
    out = orch.handle_message("I'll go with option 2", copy.deepcopy(SESSION), clients, on_event=lambda m: None, recommendation_store=store, user_id="priyanka", company_id="acme")
    check("handle: the choice is recorded and the reply confirms it", "Recorded: you chose Anthropic Claude Haiku 4.5" in out["reply"], out["reply"])
    check("handle: it is really in the store, under the right user and company", len(store.list_all()) == 1 and store.list_all()[0]["user_id"] == "priyanka" and store.list_all()[0]["company_id"] == "acme")
    check("handle: no sub-agent ran, and no model was asked to write anything", agent_calls == [])
    check("handle: the session remembers what was accepted", out["session_state"]["accepted"][0]["option_id"] == "o2")
    check("handle: the rest of the session is untouched", out["session_state"]["feature_description"] == SESSION["feature_description"] and "results" in out["session_state"])
    check("handle: the outcome carries the record", out["accepted"]["chosen"]["option_id"] == "o2")

    again = orch.handle_message("accept option 2 again", out["session_state"], clients, on_event=lambda m: None, recommendation_store=store)
    check("handle: accepting the same option twice records it once", "already recorded" in again["reply"] and len(store.list_all()) == 1)

    orch.parse_request = parse_for("accept_option", None)
    out = orch.handle_message("yes, accept", copy.deepcopy(SESSION), clients, on_event=lambda m: None, recommendation_store=rs.LocalRecommendationStore(path + ".b"))
    check("handle: an accept that names nothing asks which option, and saves nothing", "Which option" in out["reply"] and out["accepted"] is None)

    orch.parse_request = parse_for("accept_option", "option 9")
    out = orch.handle_message("option 9", copy.deepcopy(SESSION), clients, on_event=lambda m: None, recommendation_store=rs.LocalRecommendationStore(path + ".c"))
    check("handle: an option that wasn't shown is refused", "only showed 2" in out["reply"])

    orch.parse_request = parse_for("accept_option", "option 1")
    out = orch.handle_message("option 1", copy.deepcopy(SESSION), clients, on_event=lambda m: None)
    check("handle: with no store set up it says so instead of pretending it saved", out["reply"] == orch.NO_STORE_MSG, out["reply"])

    prohibited = copy.deepcopy(SESSION)
    prohibited["results"]["regulations"]["output"]["classification"] = "prohibited"
    out = orch.handle_message("option 1", prohibited, clients, on_event=lambda m: None, recommendation_store=rs.LocalRecommendationStore(path + ".d"))
    check("handle: a prohibited feature can't be accepted", "prohibited" in out["reply"] and out["accepted"] is None)

    orch.parse_request = parse_for("accept_option", "option 1")
    out = orch.handle_message("option 1", None, clients, on_event=lambda m: None, recommendation_store=rs.LocalRecommendationStore(path + ".e"))
    check("handle: with no conversation at all there is nothing to accept", "nothing to accept" in out["reply"])
finally:
    orch.parse_request, orch.run_agents = real_parse, real_run_agents
    for suffix in ("", ".b", ".c", ".d", ".e"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
