"""
The ShipWise AI Orchestrator (hybrid design).

Two language-model steps with code in between:

  1. parse_request     - a small model reads the PM's message and returns
                         structured fields (what feature, what constraints,
                         what kind of request).
  2. decide            - plain code turns those fields into which sub-agents
                         run, or a question for the PM. All routing rules live
                         here, not in a prompt.
  3. run_agents        - the chosen sub-agents run in parallel.
  4. apply_rules       - plain code applies the Regulations-precedence rules
                         (a prohibited feature never gets a model pick shown).
  5. synthesize        - a stronger model writes one reply from the results.
  6. ensure_regulations_block - plain code checks the Regulations finding and
                         its disclaimer survived into the reply, and puts them
                         back if the writing step dropped them.

First version: one request end to end, plus one continuation (the PM supplies
the missing constraints and only Model Stack runs, from saved session state).
General follow-ups are not supported yet. Nothing is written to the
`recommendations` container here; that write belongs to the PM's accept step,
which doesn't exist yet.
"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

from bounded_fanout_search import _strip_code_fence
from foundry_clients import budgeted_call, with_rate_limit_retry
from orchestrator_prompts import PARSE_PROMPT, SYNTHESIS_PROMPT
from run_competitive_analysis import call_competitive_analysis
from run_model_stack_analysis import call_model_stack_analysis
import run_log
from learning import Calibration
from outcome_report import handle_pending, start_report
from recommendation_store import _FILLER, _tokens, accept_option
from run_regulations_analysis import DISCLAIMER, run_regulations_analysis
from target_checks import add_target_checks

# Overridable so the routing eval can compare models on the same cases.
PARSE_DEPLOYMENT = os.environ.get("ORCHESTRATOR_PARSE_MODEL", "gpt-4.1-nano")
SYNTHESIS_DEPLOYMENT = "gpt-4o"

REQUEST_GATE = threading.Lock()

AGENT_IDS = ("regulations", "competitive_analysis", "model_stack_analysis")
REQUEST_TYPES = ("new_evaluation", "provide_constraints", "follow_up", "accept_option", "report_outcome", "out_of_scope", "unclear")
CONSTRAINT_KEYS = ("accuracy_target", "latency_target", "budget")

AGENT_LABELS = {
    "regulations": "the regulations check",
    "competitive_analysis": "the competitive analysis",
    "model_stack_analysis": "the model-stack analysis",
}

NEEDS_CONSTRAINTS_MSG = "To compare model options I need at least one target: accuracy, latency, or budget. What are yours?"
ASK_FEATURE_MSG = "What AI feature for customer support would you like me to evaluate? A sentence on what it does is enough."
OUT_OF_SCOPE_MSG = (
    "That's outside what I can help with. I evaluate AI features for customer support: "
    "the regulations that apply, what competitors do, and which model to use. "
    "Describe a feature and I'll start."
)
NOTHING_TO_ACCEPT_MSG = "There is nothing to accept yet. Describe the feature you want evaluated and I'll show you options to choose from."
NO_STORE_MSG = "I can't save a choice in this run because no place to store it was set up."
FOLLOW_UP_MSG = (
    "Follow-up questions about earlier results aren't supported yet. You can start a new "
    "evaluation, or give me accuracy, latency or budget targets for the saved feature."
)
PARSE_FAILED_MSG = "Sorry, I couldn't read that request. Could you describe the feature you want evaluated?"
FILTERED_MSG = (
    "That message was blocked by the platform's safety filter before I could read it. "
    "Please describe the feature you want evaluated in plain terms, without instructions aimed at the system."
)


def is_content_filter_error(e):
    # Azure OpenAI rejects some prompts outright (HTTP 400, code "content_filter").
    return isinstance(e, openai.BadRequestError) and "content_filter" in str(e)


# --------------------------------------------------------------------------
# Step 1: parse
# --------------------------------------------------------------------------

def _clean_str(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def _validate_parse(raw):
    """Turns the model's JSON into a clean, fully-populated dict, or raises
    ValueError. The model's formatting isn't trusted (same lesson as every
    other agent in this build)."""
    if not isinstance(raw, dict):
        raise ValueError("parse output is not a JSON object")
    request_type = raw.get("request_type")
    if request_type not in REQUEST_TYPES:
        raise ValueError(f"unknown request_type {request_type!r}")

    requested = raw.get("requested_agents")
    if isinstance(requested, list):
        requested = [a for a in requested if a in AGENT_IDS] or None
    else:
        requested = None

    raw_constraints = raw.get("constraints") if isinstance(raw.get("constraints"), dict) else {}
    if request_type == "report_outcome":
        # "replies take about 1.1s, $400 a month" are results the PM is reporting, not targets they want. On the
        # first live routing run the parse model filled the constraint fields from them.
        raw_constraints = {}
    return {
        "request_type": request_type,
        "feature_description": _clean_str(raw.get("feature_description")),
        "requested_agents": requested,
        "constraints": {k: _clean_str(raw_constraints.get(k)) for k in CONSTRAINT_KEYS},
        "clarifying_question": _clean_str(raw.get("clarifying_question")),
        "chosen_option": _clean_str(raw.get("chosen_option")),
    }


def _confirm_chosen_words(parse, user_message):
    """The words naming the chosen option must be the PM's own. On the first live routing run the model,
    given a message with no option named ("yes, accept it"), filled the field with the feature description
    it had been shown in the session note. Anything that isn't a verbatim excerpt of the message is dropped,
    so code asks which option was meant instead of guessing."""
    words = parse.get("chosen_option")
    if words and re.sub(r"\s+", " ", words).strip().lower() not in re.sub(r"\s+", " ", user_message).lower():
        parse = dict(parse, chosen_option=None)
    # "it" and "that one" are in the message but name nothing; code asks which option was meant.
    elif words and all(tok in _FILLER for tok in _tokens(words)):
        parse = dict(parse, chosen_option=None)
    return parse


def parse_request(model_client, user_message, session_state=None, max_attempts=2):
    session_note = "No saved session."
    if session_state and session_state.get("feature_description"):
        session_note = "A saved session exists for this feature: " + session_state["feature_description"]

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = with_rate_limit_retry(lambda: model_client.chat.completions.create(
                model=PARSE_DEPLOYMENT,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": PARSE_PROMPT},
                    {"role": "user", "content": f"{session_note}\n\n<user_message>\n{user_message}\n</user_message>"},
                ],
            ))
        except openai.BadRequestError as e:
            if not is_content_filter_error(e):
                raise
            # The platform's own filter refused the message. Retrying the same
            # text won't change that; say so and ask for a rephrase. Nothing
            # reaches any sub-agent.
            return {
                "request_type": "unclear", "feature_description": None, "requested_agents": None,
                "constraints": {k: None for k in CONSTRAINT_KEYS},
                "clarifying_question": FILTERED_MSG, "blocked_by_filter": True, "chosen_option": None,
            }
        try:
            return _confirm_chosen_words(_validate_parse(json.loads(_strip_code_fence(response.choices[0].message.content))), user_message)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            print(f"  (parse attempt {attempt} unusable: {e}; {'retrying' if attempt < max_attempts else 'giving up'})")

    # Honest failure: ask the PM to rephrase rather than guess a route.
    return {
        "request_type": "unclear", "feature_description": None, "requested_agents": None,
        "constraints": {k: None for k in CONSTRAINT_KEYS},
        "clarifying_question": PARSE_FAILED_MSG, "parse_failed": str(last_error), "chosen_option": None,
    }


# --------------------------------------------------------------------------
# Step 2: decide (all routing rules, in code)
# --------------------------------------------------------------------------

def has_numeric_constraint(constraints):
    """A target only counts if it contains a number. "fast" and "cheap" don't."""
    return any(re.search(r"\d", v) for v in constraints.values() if isinstance(v, str))


def _saved_constraint(session_state, key):
    constraints = (session_state or {}).get("constraints") or {}
    # Saved state may use the short names (accuracy/latency) or the full ones.
    return constraints.get(key) or constraints.get(key.replace("_target", ""))


def decide(parse, session_state=None):
    request_type = parse["request_type"]
    constraints = parse["constraints"]
    saved_feature = (session_state or {}).get("feature_description")

    decision = {
        "request_type": request_type, "agents": [], "ask_user": False,
        "feature_description": parse["feature_description"], "constraints": dict(constraints),
        "message_to_user": None, "chosen_option": parse.get("chosen_option"),
    }

    if request_type == "report_outcome":
        # A report usually arrives weeks later in a new conversation, so unlike accept_option it needs no saved session.
        return decision

    if request_type == "accept_option":
        if not saved_feature:
            # The parse prompt only allows this with a saved session; if it happens anyway there is nothing to accept.
            decision.update(request_type="unclear", ask_user=True, message_to_user=NOTHING_TO_ACCEPT_MSG)
        return decision

    if request_type == "provide_constraints":
        if not saved_feature:
            request_type = decision["request_type"] = "unclear"  # nothing to attach the targets to
        else:
            decision["feature_description"] = saved_feature
            decision["constraints"] = {k: constraints.get(k) or _saved_constraint(session_state, k) for k in CONSTRAINT_KEYS}
            if not has_numeric_constraint(constraints):
                decision.update(ask_user=True, message_to_user=NEEDS_CONSTRAINTS_MSG)
            else:
                decision["agents"] = ["model_stack_analysis"]
            return decision

    if request_type == "new_evaluation":
        if not parse["feature_description"]:
            request_type = decision["request_type"] = "unclear"
        else:
            requested = parse["requested_agents"]
            has_targets = has_numeric_constraint(constraints)
            if requested is None:
                agents = ["regulations", "competitive_analysis"] + (["model_stack_analysis"] if has_targets else [])
                ask = not has_targets
            else:
                agents = [a for a in AGENT_IDS if a in requested]
                ask = False
                if "model_stack_analysis" in agents and not has_targets:
                    agents.remove("model_stack_analysis")
                    ask = True
            decision.update(agents=agents, ask_user=ask, message_to_user=NEEDS_CONSTRAINTS_MSG if ask else None)
            return decision

    if request_type == "unclear":
        decision.update(ask_user=True, message_to_user=parse.get("clarifying_question") or ASK_FEATURE_MSG)
    elif request_type == "out_of_scope":
        decision["message_to_user"] = OUT_OF_SCOPE_MSG
    elif request_type == "follow_up":
        decision["message_to_user"] = FOLLOW_UP_MSG
    return decision


def plan_message(decision):
    """The plan the PM sees before anything runs. Plain text from code, no model call."""
    names = [AGENT_LABELS[a] for a in decision["agents"]]
    plan = "Plan: I'll run " + (" and ".join(names) if len(names) < 3 else ", ".join(names[:-1]) + " and " + names[-1]) + " now."
    if decision["ask_user"] and "model_stack_analysis" not in decision["agents"]:
        plan += " I'll hold the model-stack analysis until you give me at least one target (accuracy, latency, or budget)."
    return plan


# --------------------------------------------------------------------------
# Step 3: run the sub-agents in parallel
# --------------------------------------------------------------------------

def _run_one(name, feature, constraints, clients, source_text=None, calibration=None):
    try:
        if name == "regulations":
            _, output, guardrail_status = run_regulations_analysis(feature, clients)
            return {"status": "ok", "guardrail_status": guardrail_status, "output": output}
        if name == "competitive_analysis":
            _, output = call_competitive_analysis(feature, openai_client=clients.agent_client)
            return {"status": "ok", "output": output}
        _, output = call_model_stack_analysis(
            feature, constraints.get("accuracy_target"), constraints.get("latency_target"),
            constraints.get("budget"), model_client=clients.model_client, source_text=source_text, calibration=calibration,
        )
        return {"status": "ok", "output": output}
    except Exception as e:  # one sub-agent failing must not take the others down
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def _emit(progress, event):
    """Tell whoever is watching (the page's live plan panel) what just happened. A failing watcher must never
    break a request, so any error here is dropped."""
    if progress is None:
        return
    try:
        progress(event)
    except Exception:
        pass


def run_agents(agents, feature, constraints, clients, source_text=None, calibration=None, progress=None):
    """Runs the chosen sub-agents in parallel. `progress`, if given, is called with an event when each one
    starts and when each one finishes (with how long it took and whether it worked)."""
    with ThreadPoolExecutor(max_workers=max(1, len(agents))) as pool:
        futures, started = {}, {}
        for name in agents:
            started[name] = time.monotonic()
            _emit(progress, {"type": "agent_started", "agent": name, "label": AGENT_LABELS[name]})
            futures[name] = pool.submit(run_log.bind(_run_one, name, feature, constraints, clients, source_text, calibration))
        by_future = {f: name for name, f in futures.items()}
        results = {}
        for future in as_completed(by_future):
            name = by_future[future]
            results[name] = future.result()
            _emit(progress, {"type": "agent_finished", "agent": name, "label": AGENT_LABELS[name], "status": results[name]["status"],
                             "seconds": round(time.monotonic() - started[name], 1)})
        return {name: results[name] for name in agents}


# --------------------------------------------------------------------------
# Step 4: Regulations-precedence rules (in code)
# --------------------------------------------------------------------------

def apply_rules(results, prior_results=None, constraints=None):
    """
    Builds the facts the writing step is allowed to use, and enforces:
    - regulations `prohibited`  -> the model pick is withheld, never shown
    - regulations `high_risk`   -> the model pick is flagged for validation
    - regulations failed to run -> said plainly; no model pick presented as cleared
    A Regulations result from an earlier turn counts the same as a new one.
    """
    regs_result = results.get("regulations") or (prior_results or {}).get("regulations")
    regs_output = regs_result["output"] if regs_result and regs_result.get("status") == "ok" else None
    classification = regs_output.get("classification") if regs_output else None

    facts = {"flags": {}, "not_run_notes": []}
    if regs_output is not None:
        facts["regulations"] = regs_output
    elif regs_result is not None:
        facts["regulations"] = {"status": "error", "note": "The regulations check could not run: " + regs_result.get("error", "unknown error")}

    comp = results.get("competitive_analysis")
    if comp:
        facts["competitive_analysis"] = comp["output"] if comp["status"] == "ok" else {"status": "error", "note": comp.get("error")}

    model = results.get("model_stack_analysis")
    if model:
        if classification == "prohibited":
            facts["model_recommendation"] = "withheld"
            facts["flags"]["model_withheld_reason"] = "The regulations check found this feature prohibited under the EU AI Act, so a model recommendation would be moot."
            run_log.event("orchestrator", "prohibited_feature_model_withheld", True)
        else:
            # Each option gets a code-computed verdict against the PM's targets, so the
            # writing step copies it instead of judging "meets 90%" from free text.
            facts["model_stack_analysis"] = add_target_checks(model["output"], constraints) if model["status"] == "ok" else {"status": "error", "note": model.get("error")}
            if classification == "high_risk":
                facts["flags"]["model_pick_needs_validation"] = True
                run_log.event("orchestrator", "high_risk_pick_flagged_for_validation", True)
            if regs_result is not None and regs_output is None:
                facts["flags"]["regulations_unavailable_do_not_call_model_pick_cleared"] = True
                run_log.event("orchestrator", "regulations_unavailable_pick_not_cleared", True)
    return facts, classification


# --------------------------------------------------------------------------
# Steps 5 and 6: write the reply, then check it kept the legal finding
# --------------------------------------------------------------------------

def synthesize(model_client, facts):
    try:
        response = budgeted_call(lambda: model_client.chat.completions.create(
            model=SYNTHESIS_DEPLOYMENT,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYNTHESIS_PROMPT},
                {"role": "user", "content": json.dumps(facts, indent=2)},
            ],
        ), SYNTHESIS_PROMPT + json.dumps(facts, indent=2), key="synthesis")
    except openai.BadRequestError as e:
        if not is_content_filter_error(e):
            raise
        run_log.event("orchestrator", "writing_step_refused_by_content_filter", True, "findings assembled by code")
        return fallback_reply(facts)
    return response.choices[0].message.content.strip()


def fallback_reply(facts):
    """Used when the writing step is refused by the platform filter. Plain
    code assembles the findings so the PM still gets them, unpolished."""
    parts = ["(The summary could not be written because a safety filter rejected it; the findings follow as returned.)"]
    if facts.get("flags", {}).get("model_withheld_reason"):
        parts.append(facts["flags"]["model_withheld_reason"])
    for key, label in (("competitive_analysis", "Competitive analysis"), ("model_stack_analysis", "Model stack options")):
        if key in facts:
            parts.append(f"{label}:\n{json.dumps(facts[key], indent=2)}")
    if facts.get("question_for_user"):
        parts.append(facts["question_for_user"])
    return "\n\n".join(parts)


_QUESTION_SECTION = re.compile(r"\n+#{1,4}[ \t]*questions?\b[^\n]*(?:\n.*)?\Z", re.I | re.S)


def strip_invented_question(reply, question_for_user):
    """The writing step sometimes ends with a "Question for user" section when
    the routing asked nothing (seen in 2 of 3 live runs; a prompt rule alone
    didn't stop it). If code didn't ask a question, the section is removed.
    Returns (reply, changed)."""
    if question_for_user:
        return reply, False
    stripped = _QUESTION_SECTION.sub("", reply).rstrip()
    return stripped, stripped != reply.rstrip()


def ensure_regulations_block(reply, regs_output):
    """
    The Regulations finding and disclaimer must reach the PM word for word,
    however the writing step phrased things. If the header line is missing,
    put the whole finding in front; if only the disclaimer is missing, add it
    at the end. Returns (reply, changed).
    """
    if not regs_output:
        return reply, False
    classification = regs_output.get("classification")
    disclaimer = regs_output.get("disclaimer") or DISCLAIMER
    header = f"Regulatory classification: {classification}"

    if header not in reply:
        block = (
            f"{header}\n{regs_output.get('answer', '')}\n"
            f"Applicability: {regs_output.get('applicability_status')}\n{disclaimer}\n"
            "(Added by the system because the written summary left out the regulatory finding.)\n\n"
        )
        return block + reply, True
    reply, changed = ensure_applicability_status(reply, regs_output.get("applicability_status"), header)
    if disclaimer not in reply:
        return reply.rstrip() + "\n\n" + disclaimer, True
    return reply, changed


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_FULL_DATE = re.compile(rf"\b\d{{1,2}} (?:{_MONTHS}) \d{{4}}\b")
_SENTENCE_SPLIT = re.compile(r"((?<=[.!?])\s+)")
_TIMING_WORDS = ("applic", "article 113", "take effect", "takes effect", "come into", "comes into", "enter into", "in force")


def ensure_applicability_status(reply, status, header):
    """
    The applicability status is built in code from Article 113, but the
    writing step restates it in its own words, and in the first end-to-end run of
    2026-10-04 it turned "2 August 2026 (Article 50)" into "2 August 2025", a real date
    that belongs to other chapters. Two rules, both in code: a sentence that gives a full
    date about when something applies, where the date is not one in the code-built status,
    is removed; and the code-built status must appear in the reply word for word, and is
    added under the finding if it doesn't. Returns (reply, changed).
    """
    if not status:
        return reply, False
    allowed = set(_FULL_DATE.findall(status))
    changed = False
    lines = []
    for line in reply.split("\n"):
        parts = _SENTENCE_SPLIT.split(line)
        kept = []
        dropped = False
        for i in range(0, len(parts), 2):
            sentence = parts[i]
            lower = sentence.lower()
            wrong = [d for d in _FULL_DATE.findall(sentence) if d not in allowed]
            if wrong and any(w in lower for w in _TIMING_WORDS):
                dropped = True
                continue
            kept.append(sentence)
            if i + 1 < len(parts):
                kept.append(parts[i + 1])
        if dropped:
            changed = True
            line = "".join(kept).rstrip()
        lines.append(line)
    reply = "\n".join(lines)
    if status not in reply:
        at = reply.find(header)
        end = reply.find("\n\n", at)
        end = len(reply) if end == -1 else end
        reply = reply[:end] + f"\n\nApplicability: {status}" + reply[end:]
        changed = True
    return reply, changed


# --------------------------------------------------------------------------
# The whole turn
# --------------------------------------------------------------------------

def handle_message(user_message, session_state, clients, on_event=print, recommendation_store=None,
                   user_id="demo-user", company_id="demo-company", progress=None):
    """
    One PM message in, one reply out. `session_state` is the saved state (or
    None); the returned dict includes the new state to save.
    """
    # A report that is half-finished (waiting to be confirmed, corrected, or matched to a recommendation) is
    # continued here, in code, before any parsing: "yes", "cancel" and "no, it was 78%" need no model to route them.
    if recommendation_store is not None and (session_state or {}).get("pending_outcome"):
        continued = handle_pending(clients.model_client, user_message, session_state, recommendation_store)
        if continued:
            return {"parse": None, "decision": {"request_type": "report_outcome", "agents": [], "ask_user": False}, "results": {},
                    "reply": continued["reply"], "session_state": continued["session_state"], "saved_outcome": continued.get("saved_outcome")}

    _emit(progress, {"type": "reading"})
    parse = parse_request(clients.model_client, user_message, session_state)
    decision = decide(parse, session_state)
    outcome = {"parse": parse, "decision": decision, "results": {}, "reply": None, "session_state": session_state}

    if decision["request_type"] == "report_outcome":
        if recommendation_store is None:
            outcome["reply"] = NO_STORE_MSG
            return outcome
        started = start_report(clients.model_client, user_message, session_state, recommendation_store, user_id)
        outcome["reply"] = started["reply"]
        outcome["session_state"] = started["session_state"]
        return outcome

    if decision["request_type"] == "accept_option":
        # Recording the PM's choice is plain code (recommendation_store.py): no sub-agent and no model call.
        if recommendation_store is None:
            outcome["reply"] = NO_STORE_MSG
            return outcome
        accepted = accept_option(session_state, decision["chosen_option"], user_id, company_id, recommendation_store)
        outcome["reply"] = accepted["message"]
        outcome["accepted"] = accepted["record"] if accepted["ok"] else None
        if accepted["session_update"]:
            outcome["session_state"] = {**(session_state or {}), **accepted["session_update"]}
        return outcome

    if not decision["agents"]:
        outcome["reply"] = decision["message_to_user"]
        # Keep a described feature so a later "here are my targets" message can attach to it.
        if decision["feature_description"] and decision["request_type"] in ("new_evaluation", "provide_constraints"):
            outcome["session_state"] = {**(session_state or {}), "feature_description": decision["feature_description"],
                                        "source_text": user_message if decision["request_type"] == "new_evaluation" else (session_state or {}).get("source_text"),
                                        "constraints": decision["constraints"], "agents_already_run": (session_state or {}).get("agents_already_run", []),
                                        "results": (session_state or {}).get("results", {})}
        return outcome

    prior = (session_state or {}).get("results", {})
    prior_regs = (prior.get("regulations") or {}).get("output") or {}

    # Rule: a feature already found prohibited gets no model run at all.
    if decision["request_type"] == "provide_constraints" and prior_regs.get("classification") == "prohibited":
        run_log.event("orchestrator", "prohibited_feature_no_model_run", True)
        outcome["reply"] = ("The regulations check already found this feature prohibited under the EU AI Act, "
                            "so I won't produce a model recommendation for it.")
        return outcome

    # One request at a time through the sub-agents and the writing step. The gpt-4o deployment allows
    # 50,000 tokens a minute and one request uses about 37,000, so two requests overlapping can't both fit.
    # Letting them interleave made one request's small calls use up budget the other's large legal check
    # needed, and both finished late (135 and 195 seconds in a live test). Taking turns lets the first
    # finish at its normal speed. It only covers this process; separate processes still rely on the 429 retry.
    if not REQUEST_GATE.acquire(blocking=False):
        on_event("  (another request is using the model quota; yours will start when it finishes)")
        REQUEST_GATE.acquire()
    try:
        on_event(plan_message(decision))
        _emit(progress, {"type": "plan", "request_type": decision["request_type"], "agents": decision["agents"],
                         "agent_labels": {a: AGENT_LABELS[a] for a in decision["agents"]}, "constraints": decision["constraints"],
                         "feature": decision["feature_description"], "message": plan_message(decision)})
        # The Model Stack interpretation needs the PM's own words, not the parse step's one-sentence summary of
        # the feature, which drops details such as a stated request volume. A follow-up message that only gives
        # targets reuses the words saved from the original request.
        source_text = (session_state or {}).get("source_text") if decision["request_type"] == "provide_constraints" else None
        source_text = source_text or user_message
        # What the company's confirmed outcome reports have taught about each model setting (learning.py).
        calibration = Calibration(recommendation_store.list_all(), company_id) if recommendation_store is not None else None
        results = run_agents(decision["agents"], decision["feature_description"], decision["constraints"], clients, source_text, calibration, progress)
        facts, classification = apply_rules(results, prior, decision["constraints"])
        facts["feature_description"] = decision["feature_description"]
        facts["constraints"] = decision["constraints"]

        if decision["ask_user"] and classification == "prohibited":
            # No point asking for model targets for a feature that can't ship.
            facts["not_run_notes"].append("The model-stack analysis was not run and will not be: the feature is prohibited.")
        elif decision["ask_user"]:
            facts["not_run_notes"].append("The model-stack analysis has not run: no numeric accuracy, latency or budget target was given.")
            facts["question_for_user"] = decision["message_to_user"]

        _emit(progress, {"type": "writing"})
        reply = synthesize(clients.model_client, facts)
    finally:
        REQUEST_GATE.release()
    reply, removed = strip_invented_question(reply, facts.get("question_for_user"))
    run_log.event("orchestrator", "invented_question_removed", removed)
    if removed:
        on_event("  (the written summary ended with a question nobody asked; the system removed it)")
    regs_output = facts.get("regulations") if isinstance(facts.get("regulations"), dict) and "classification" in facts["regulations"] else None
    reply, restored = ensure_regulations_block(reply, regs_output)
    if regs_output:
        run_log.event("orchestrator", "regulatory_finding_kept_in_reply", restored, "finding or date line put back or corrected" if restored else None)
    if restored:
        on_event("  (the written summary left out part of the regulatory finding; the system put it back)")

    merged_results = {**prior, **{k: v for k, v in results.items() if v.get("status") == "ok"}}
    outcome.update(
        results=results, facts=facts, reply=reply,
        session_state={
            "feature_description": decision["feature_description"],
            "source_text": source_text,
            "constraints": decision["constraints"],
            "agents_already_run": sorted(set((session_state or {}).get("agents_already_run", [])) | {k for k, v in results.items() if v.get("status") == "ok"}),
            "results": merged_results,
        },
    )
    return outcome
