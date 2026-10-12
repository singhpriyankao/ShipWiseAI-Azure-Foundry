"""
Tests for the Orchestrator's code-only rules (routing decisions, the
Regulations-precedence rules, and the check that the legal finding survives
the writing step). No model calls, no sign-in: sub-agent outputs are
hand-built from the shapes the real agents return.

Run: python test_orchestrator_rules.py
"""
import warnings
warnings.filterwarnings("ignore")

from orchestrator import (
    DISCLAIMER, apply_rules, decide, ensure_applicability_status, ensure_regulations_block, has_numeric_constraint, plan_message, strip_invented_question,
)

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


def parse(request_type, feature="a feature", requested=None, **constraints):
    c = {"accuracy_target": None, "latency_target": None, "budget": None}
    c.update(constraints)
    return {"request_type": request_type, "feature_description": feature, "requested_agents": requested,
            "constraints": c, "clarifying_question": None}


def regs(classification):
    return {"status": "ok", "guardrail_status": "passed", "output": {
        "classification": classification, "answer": "Because of Article X.", "applicability_status": "From 2 August 2026.",
        "disclaimer": DISCLAIMER}}


MODEL = {"status": "ok", "output": {"scenarios": [{"model_stack": "gpt-4o"}]}}
COMP = {"status": "ok", "output": {"summary": "competitors"}}

# ---- numeric-constraint rule
check("'fast' is not a constraint", not has_numeric_constraint({"latency_target": "fast", "budget": None}))
check("'under 2 seconds' is a constraint", has_numeric_constraint({"latency_target": "under 2 seconds"}))

# ---- decide(): routing
d = decide(parse("new_evaluation"))
check("no constraints -> regulations + competitive, ask", d["agents"] == ["regulations", "competitive_analysis"] and d["ask_user"])
d = decide(parse("new_evaluation", latency_target="2s"))
check("one numeric constraint -> all three, no question", len(d["agents"]) == 3 and not d["ask_user"])
d = decide(parse("new_evaluation", latency_target="fast"))
check("vague adjective only -> model stack held", "model_stack_analysis" not in d["agents"] and d["ask_user"])
d = decide(parse("new_evaluation", requested=["regulations"]))
check("narrow regulations -> only regulations, no question", d["agents"] == ["regulations"] and not d["ask_user"])
d = decide(parse("new_evaluation", requested=["model_stack_analysis"]))
check("narrow model, no targets -> nothing runs, ask", d["agents"] == [] and d["ask_user"])
d = decide(parse("new_evaluation", requested=["model_stack_analysis"], budget="$15K"))
check("narrow model with targets -> only model stack", d["agents"] == ["model_stack_analysis"] and not d["ask_user"])
d = decide(parse("new_evaluation", feature=None))
check("new_evaluation with no feature -> unclear, ask", d["request_type"] == "unclear" and d["ask_user"])
d = decide(parse("unclear", feature=None))
check("unclear -> ask, nothing runs", d["agents"] == [] and d["ask_user"])
d = decide(parse("out_of_scope", feature=None))
check("out_of_scope -> declined, nothing runs", d["agents"] == [] and not d["ask_user"] and d["message_to_user"])
d = decide(parse("follow_up", feature=None), {"feature_description": "x"})
check("follow_up -> not supported yet, nothing runs", d["agents"] == [] and "aren't supported" in d["message_to_user"])

session = {"feature_description": "ticket summarizer", "constraints": {"accuracy": None, "latency": None, "budget": None}}
d = decide(parse("provide_constraints", feature=None, latency_target="500ms"), session)
check("targets with a saved feature -> model stack only, saved feature used",
      d["agents"] == ["model_stack_analysis"] and d["feature_description"] == "ticket summarizer" and d["constraints"]["latency_target"] == "500ms")
d = decide(parse("provide_constraints", feature=None, budget="cheap"), session)
check("provide_constraints with no number -> ask again, nothing runs", d["agents"] == [] and d["ask_user"])
d = decide(parse("provide_constraints", feature=None, latency_target="500ms"), None)
check("targets with no saved session -> unclear, ask what the feature is", d["request_type"] == "unclear" and d["agents"] == [])
check("plan names what runs and what is held", "regulations check" in plan_message(decide(parse("new_evaluation"))) and "hold" in plan_message(decide(parse("new_evaluation"))))

# ---- apply_rules(): Regulations precedence
facts, cls = apply_rules({"regulations": regs("prohibited"), "model_stack_analysis": MODEL, "competitive_analysis": COMP})
check("prohibited -> model pick withheld", "model_stack_analysis" not in facts and facts.get("model_recommendation") == "withheld")
check("prohibited -> competitive analysis still shown", "competitive_analysis" in facts)
check("prohibited -> regulations finding kept", facts["regulations"]["classification"] == "prohibited")
facts, cls = apply_rules({"regulations": regs("high_risk"), "model_stack_analysis": MODEL})
check("high_risk -> model pick shown but flagged for validation", "model_stack_analysis" in facts and facts["flags"].get("model_pick_needs_validation"))
facts, cls = apply_rules({"regulations": regs("transparency_obligated"), "model_stack_analysis": MODEL})
check("transparency_obligated -> model pick shown, not flagged", "model_stack_analysis" in facts and not facts["flags"])
facts, cls = apply_rules({"regulations": regs("analysis_failed")})
check("analysis_failed passes through as exactly that", facts["regulations"]["classification"] == "analysis_failed")
facts, cls = apply_rules({"regulations": {"status": "error", "error": "RuntimeError: boom"}, "model_stack_analysis": MODEL})
check("regulations crashed -> said plainly, model pick not presented as cleared",
      facts["regulations"]["status"] == "error" and facts["flags"].get("regulations_unavailable_do_not_call_model_pick_cleared"))
facts, cls = apply_rules({"model_stack_analysis": MODEL}, prior_results={"regulations": regs("prohibited")})
check("prohibited found in an EARLIER turn still withholds the pick", "model_stack_analysis" not in facts)
facts, cls = apply_rules({"model_stack_analysis": MODEL}, prior_results={"regulations": regs("high_risk")})
check("high_risk from an earlier turn still flags the pick", facts["flags"].get("model_pick_needs_validation"))

# ---- ensure_regulations_block(): the finding must survive the writing step
r = regs("high_risk")["output"]
good = f"Regulatory classification: high_risk\nBecause. {r['applicability_status']}\n{DISCLAIMER}\n\nCompetitors..."
text, changed = ensure_regulations_block(good, r)
check("complete reply is left alone", not changed and text == good)
text, changed = ensure_regulations_block("Here are some model options.", r)
check("reply missing the finding -> finding put in front", changed and text.startswith("Regulatory classification: high_risk") and DISCLAIMER in text)
text, changed = ensure_regulations_block("Regulatory classification: high_risk\nBecause.", r)
check("reply missing only the disclaimer -> disclaimer appended", changed and text.endswith(DISCLAIMER))
text, changed = ensure_regulations_block("Reply says Regulatory classification: no_specific_obligation_identified", r)
check("reply with the WRONG classification -> correct finding put in front", changed and text.startswith("Regulatory classification: high_risk"))
text, changed = ensure_regulations_block("No regulations ran.", None)
check("no regulations result -> nothing to enforce", not changed)

# ---- invented closing question
r, changed = strip_invented_question("Findings.\n\n### Question for user\nShould we add disclosure?", None)
check("invented question section removed when none was asked", r == "Findings." and changed, repr(r))
r, changed = strip_invented_question("Findings.\n\n### Question for user\nWhat is your budget?", "What is your budget?")
check("real question section kept", not changed and "budget" in r)
r, changed = strip_invented_question("Findings with no question section.", None)
check("reply without a question section untouched", not changed)
r, changed = strip_invented_question("Findings.\n\n### Competitive analysis\nQuestions of price are answered.", None)
check("other sections mentioning questions untouched", not changed, repr(r))

# ---- the applicability date must survive the writing step (found in the 2026-10-04 end-to-end run)
STATUS = "Under Article 113, the provisions cited here take effect on these dates: 2 August 2026 (Article 50; the general application date)."
HEADER = "Regulatory classification: transparency_obligated"
r50 = {"classification": "transparency_obligated", "answer": "Article 50 applies.", "disclaimer": DISCLAIMER, "applicability_status": STATUS}
# The wording of the live reply: the writing step restated the date, wrongly.
live_reply = (HEADER + "  \nArticle 50 paragraph 1 of the EU AI Act governs AI systems that interact directly with natural persons. "
              "This provision imposes a transparency obligation on providers. The applicability date for this transparency "
              "obligation is 2 August 2025, per Article 113 of the EU AI Act.  \n\n" + DISCLAIMER + "\n\n### Competitive Analysis\nX.")
text, changed = ensure_regulations_block(live_reply, r50)
check("a wrong applicability date is removed", "2 August 2025" not in text, text)
check("the code-built status is added in its place", STATUS in text and changed, text)
check("the rest of the finding is kept", "transparency obligation on providers" in text and "### Competitive Analysis" in text)
check("the status goes under the finding, before the next section", text.index(STATUS) < text.index("### Competitive Analysis"))
good50 = HEADER + "\nArticle 50 applies. " + STATUS + "\n\n" + DISCLAIMER
text, changed = ensure_regulations_block(good50, r50)
check("a reply that already has the status is untouched", text == good50 and not changed, text)
text, changed = ensure_applicability_status("Header\nIt takes effect on 2 August 2026.", STATUS, "Header")
check("the right date in the model's own words is kept", "It takes effect on 2 August 2026." in text)
text, changed = ensure_applicability_status("Header\nIntercom launched on 3 March 2024 and applies to billing.", STATUS, "Header")
check("a date that isn't about when a rule applies is not removed", "3 March 2024" in text, text)
text, changed = ensure_applicability_status("Header\nFine.", None, "Header")
check("no status, nothing to enforce", text == "Header\nFine." and not changed)

print(f"\n{'ALL PASSED' if not failures else str(len(failures)) + ' FAILED: ' + ', '.join(failures)}")
raise SystemExit(1 if failures else 0)
