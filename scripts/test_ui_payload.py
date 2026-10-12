"""
Offline tests for ui_payload.py (what the page shows for a finished request) and for the Orchestrator's progress
events. No network. Fake agents stand in for Azure.

Run: python test_ui_payload.py
"""
import warnings
warnings.filterwarnings("ignore")

import orchestrator as orch
import ui_payload
from session_store import OwnedSessionStore

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


# ---- citation labels
check("article with paragraph", ui_payload.citation_label({"article_number": 50, "paragraph_number": 1}) == "Article 50(1)")
check("article without paragraph", ui_payload.citation_label({"article_number": 6}) == "Article 6")
check("annex item with sub-point", ui_payload.citation_label({"annex_id": "III", "annex_item_number": 5, "annex_sub_point": "b"}) == "Annex III, item 5(b)")
check("annex alone", ui_payload.citation_label({"annex_id": "III"}) == "Annex III")
check("recital", ui_payload.citation_label({"recital_number": 53}) == "Recital 53")
check("an empty citation still gives something readable", ui_payload.citation_label({}) == "EU AI Act")


def opt(oid, role, quality, yearly, latency, verdicts=None):
    return {"id": oid, "role": role, "model_stack": f"Model {oid}", "quality_percentile": quality, "yearly_cost_usd": yearly, "latency_ms": latency,
            "latency_means": "time until the first answer token arrives (the reply starts appearing)",
            "verdicts": verdicts or {"latency": "meets", "cost": "meets"}, "trade_offs": "t", "comparison": "c", "learned": []}


MODEL = {"scenarios": [opt("s1", "shortlist", 90, 1200, 600), opt("s2", "shortlist", 70, 300, 700), opt("n1", "near_miss", 80, 500, 1000, {"latency": "misses", "cost": "meets"})],
         "volume_assumption": {"requests_per_month": 20000, "volume_basis": "assumed"}, "overview": "ov", "caveats": ["c1"],
         "accuracy_note": "No public benchmark measures accuracy on your task.", "attribution": "Artificial Analysis"}
REGS = {"classification": "transparency_obligated", "answer": "Article 50 applies.", "applicability_status": "2 August 2026",
        "citations": [{"article_number": 50, "paragraph_number": 1}], "missing_inputs": [], "disclaimer": "Not legal advice."}
COMP = {"competitors": [{"product": "Intercom Fin", "vendor": "Intercom", "pricing_notes": "$0.99 per resolution", "pricing_source_url": "https://x/y",
                         "accuracy_notes": "No verified source", "latency_notes": "No source found:", "source_checks": {"pricing": "verified", "accuracy": "page_does_not_support_claim"}}]}


def outcome(regs=REGS, regs_status="ok"):
    results = {"regulations": {"status": regs_status, "output": regs, "guardrail_status": "passed", **({"error": "boom"} if regs_status == "error" else {})},
               "competitive_analysis": {"status": "ok", "output": COMP},
               "model_stack_analysis": {"status": "ok", "output": MODEL}}
    return {"decision": {"request_type": "new_evaluation", "agents": list(results), "constraints": {"latency_target": "under 800ms", "budget": "$40K per year", "accuracy_target": "at least 90%"}},
            "results": results, "facts": {"model_stack_analysis": MODEL}, "reply": "The summary.",
            "session_state": {"feature_description": "a billing chatbot", "constraints": {"latency_target": "under 800ms", "budget": "$40K per year", "accuracy_target": "at least 90%"}, "results": results}}


r = ui_payload.build_ui_result(outcome(), "sess-1")
check("the page gets the summary, the feature and the session id", r["summary"] == "The summary." and r["feature"] == "a billing chatbot" and r["session_id"] == "sess-1")
check("chips carry latency and budget as the PM wrote them", r["chips"]["latency"] == "under 800ms" and r["chips"]["budget"] == "$40K per year", r["chips"])
check("the volume chip says whether the volume was assumed", r["chips"]["volume"] == {"requests_per_month": 20000, "basis": "assumed"}, r["chips"])
check("there is no accuracy chip, only a note that an accuracy target was given", "accuracy" not in r["chips"] and r["chips"]["accuracy_target_given"] is True)
check("the legal banner has a tone, a title, plain citations and the disclaimer", r["regulatory"]["tone"] == "notice" and r["regulatory"]["title"].startswith("Transparency") and r["regulatory"]["citations"] == ["Article 50(1)"] and r["regulatory"]["disclaimer"], r["regulatory"])
check("three cards are produced", [c["slot"] for c in r["cards"]] == ["A", "B", "C"], r["cards"])
check("the competitors come with a check result for each claim", r["competitors"][0]["claims"][0] == {"field": "pricing", "note": "$0.99 per resolution", "source_url": "https://x/y", "check": "verified"}, r["competitors"])
check("an unverified claim keeps its failed check so the page can say so", r["competitors"][0]["claims"][1]["check"] == "page_does_not_support_claim")
check("the page can accept an option when options exist", r["can_accept"] is True)
check("the accuracy note and attribution are passed through", r["accuracy_note"] and r["attribution"] == "Artificial Analysis")

# ---- a prohibited feature: no cards, no accept
p = ui_payload.build_ui_result(outcome(regs={"classification": "prohibited", "answer": "Article 5(1)(f).", "citations": [{"article_number": 5, "paragraph_number": 1}]}), "s")
check("a prohibited feature shows a blocked banner", p["regulatory"]["tone"] == "blocked" and p["withheld"] is True)
check("...with no cards and no way to accept", p["cards"] == [] and p["can_accept"] is False)

# ---- the legal check failed or didn't run
f = ui_payload.build_ui_result(outcome(regs_status="error"), "s")
check("a failed legal check is shown as failed, never as 'no obligation'", f["regulatory"]["status"] == "failed" and f["regulatory"]["tone"] == "unclear", f["regulatory"])
check("...and the cards still appear (the options are not blocked by a failed check)", len(f["cards"]) == 3)
n = ui_payload.build_ui_result({"decision": {"request_type": "new_evaluation", "agents": ["model_stack_analysis"], "constraints": {}},
                                "results": {"model_stack_analysis": {"status": "ok", "output": MODEL}}, "facts": {"model_stack_analysis": MODEL}, "reply": "r",
                                "session_state": {"results": {"model_stack_analysis": {"status": "ok", "output": MODEL}}}}, "s")
check("a request that didn't run the legal check says it wasn't run", n["regulatory"]["status"] == "not_run")
check("a request with no agents (a question back to the PM) yields an empty page document, not an error",
      ui_payload.build_ui_result({"decision": {"request_type": "unclear", "agents": []}, "results": {}, "reply": "What feature?"}, "s")["cards"] == [])
check("None is handled", ui_payload.build_ui_result(None, "s")["cards"] == [])

# ---- progress events from the Orchestrator
events = []
real_run_one = orch._run_one
orch._run_one = lambda name, *a, **k: {"status": "ok" if name != "competitive_analysis" else "error", "output": {}, "error": "x"}
try:
    orch.run_agents(["regulations", "competitive_analysis", "model_stack_analysis"], "f", {}, None, progress=events.append)
finally:
    orch._run_one = real_run_one
kinds = [(e["type"], e["agent"]) for e in events]
check("every agent reports a start and a finish", sum(1 for k, _ in kinds if k == "agent_started") == 3 and sum(1 for k, _ in kinds if k == "agent_finished") == 3, kinds)
check("each start comes before that agent's finish", all(kinds.index(("agent_started", a)) < kinds.index(("agent_finished", a)) for a in ("regulations", "competitive_analysis", "model_stack_analysis")), kinds)
fin = {e["agent"]: e for e in events if e["type"] == "agent_finished"}
check("a finish says whether it worked, and how long it took", fin["competitive_analysis"]["status"] == "error" and fin["regulations"]["status"] == "ok" and isinstance(fin["regulations"]["seconds"], float), fin)
check("labels are the plain names the PM sees", fin["regulations"]["label"] == "the regulations check")


def exploding(event):
    raise RuntimeError("watcher broke")


orch._run_one = lambda name, *a, **k: {"status": "ok", "output": {}}
try:
    res = orch.run_agents(["regulations"], "f", {}, None, progress=exploding)
finally:
    orch._run_one = real_run_one
check("a watcher that raises never breaks the request", res == {"regulations": {"status": "ok", "output": {}}}, res)

# ---- sessions belong to their user
store = OwnedSessionStore()
store.put("s1", "amara", {"x": 1})
check("a user reads back their own session", store.get("s1", "amara") == {"x": 1})
check("another user gets nothing for that session id", store.get("s1", "someone-else") is None)
try:
    store.put("s1", "someone-else", {"x": 2})
    overwritten = True
except PermissionError:
    overwritten = False
check("another user cannot overwrite it", overwritten is False and store.get("s1", "amara") == {"x": 1})
check("an unknown session is nothing, not an error", store.get("nope", "amara") is None)

# ---- the verdict tiles, how to read, coverage, guardrails
MODEL2 = {**MODEL, "interpretation": {"reply_style": "streamed_chat"}, "counts": {"considered": 102, "meet_all_given_targets": 5, "latency_unknown": 66},
          "latency_means": "time until the first answer token arrives", "volume_assumption": {"requests_per_month": 20000, "volume_basis": "assumed", "avg_input_tokens": 1500, "avg_output_tokens": 250, "token_basis": "estimated"}}
o2 = outcome()
o2["results"]["model_stack_analysis"]["output"] = MODEL2
o2["facts"]["model_stack_analysis"] = MODEL2
o2["session_state"]["results"]["model_stack_analysis"]["output"] = MODEL2
events = [{"agent": "regulations", "rule": "citation_check", "fired": True, "detail": "1 invalid"}, {"agent": "regulations", "rule": "recital_only_check", "fired": False},
          {"agent": "competitive_analysis", "rule": "source_check:page_does_not_support_claim", "fired": True, "detail": "9 claim(s)"},
          {"agent": "competitive_analysis", "rule": "source_check:verified", "fired": False, "detail": "2 claim(s)"},
          {"agent": "model_stack_analysis", "rule": "token_estimate_held_to_typical_range", "fired": True, "detail": "input tokens 200 -> 1,500"},
          {"agent": "orchestrator", "rule": "some_new_rule", "fired": True}]
v = ui_payload.build_ui_result(o2, "s", guardrail_events=events)
tiles = {t["key"]: t for t in v["verdicts"]}
check("three tiles in order: legal, recommended, market", [t["key"] for t in v["verdicts"]] == ["legal", "recommended", "market"])
check("the legal tile names the result, the act and the article, and when it applies", tiles["legal"]["title"] == "Transparency duty" and tiles["legal"]["detail"] == "EU AI Act, Article 50(1)" and tiles["legal"]["extra"] == "Applies from 2 August 2026" and tiles["legal"]["tone"] == "notice", tiles["legal"])
check("the recommended tile names the top option with monthly cost and latency", tiles["recommended"]["title"] == "Model s1" and tiles["recommended"]["detail"] == "$100/month · 600 ms to the first words" and tiles["recommended"]["extra"] == "2 other options below", tiles["recommended"])
check("the market tile counts the competitors and the verified claims", tiles["market"]["title"] == "1 competitor researched" and tiles["market"]["detail"] == "1 of 3 claims verified", tiles["market"])
check("partly verified is amber, not green, and makes no 'nothing verified' claim", tiles["market"]["tone"] == "caution" and tiles["market"]["extra"] is None, tiles["market"])
zero = ui_payload.verdict_tiles({"status": "checked", "short": "x", "tone": "clear", "citations": [], "dates": []}, [], False, True, {"total": 9, "verified": 0, "not_supported": 9, "could_not_check": 0, "no_source": 0, "unchecked": 0}, 3)[2]
check("with nothing verified (the case in the PDF) the tile says so plainly and is not green", zero["title"] == "3 competitors researched" and zero["detail"] == "0 of 9 claims verified" and zero["extra"] == "No claim could be backed by its source" and zero["tone"] == "caution", zero)
check("the legal block carries the dates found in the applicability text", v["regulatory"]["dates"] == ["2 August 2026"] and v["regulatory"]["quotes"] == [])
check("the chips include the reply style read from the description", v["chips"]["reply_style"] == "streamed chat")
check("'how to read' is laid out as separate facts", v["reading"]["compared"] == 102 and v["reading"]["meet_all"] == 5 and v["reading"]["latency_unknown"] == 66 and v["reading"]["assumptions"]["avg_input_tokens"] == 1500 and v["reading"]["accuracy_note"], v["reading"])
cov = v["competitor_coverage"]
check("coverage counts each claim by what its check found, and claims with no check recorded as unchecked", cov == {"total": 3, "verified": 1, "not_supported": 1, "could_not_check": 0, "no_source": 0, "unchecked": 1}, cov)
check("a competitor row says which pages it cited", v["competitors"][0]["cited_hosts"] == ["x"], v["competitors"][0])
g = v["guardrails"]
check("guardrail summary counts every check and those that fired", g["checked"] == 6 and g["fired"] == 4, g)
check("fired guardrails read as plain sentences", "A citation was not in the retrieved law, so the legal answer was rejected (1 invalid)" in g["lines"] and "9 competitor claims removed because the cited page did not support them" in g["lines"], g["lines"])
check("a guardrail that did not fire is not listed", not any("recital" in line.lower() for line in g["lines"]) and not any("verified" in line.lower() for line in g["lines"]), g["lines"])
check("the token-estimate guardrail carries what it changed", any("raised to the typical range (input tokens 200 -> 1,500)" in line for line in g["lines"]), g["lines"])
check("an unknown rule is still shown, readably", "some new rule" in g["lines"], g["lines"])
check("no events gives an empty summary, not an error", ui_payload.guardrail_summary(None) == {"checked": 0, "fired": 0, "lines": []})

# tiles for the other situations
pro = ui_payload.build_ui_result(outcome(regs={"classification": "prohibited", "answer": "x", "citations": [{"article_number": 5, "paragraph_number": 1}], "applicability_status": "2 February 2025"}), "s")
pt = {t["key"]: t for t in pro["verdicts"]}
check("a prohibited feature: a blocked legal tile and no recommendation", pt["legal"]["tone"] == "blocked" and pt["legal"]["title"] == "Prohibited" and pt["recommended"]["title"] == "No recommendation" and pt["recommended"]["tone"] == "blocked", pt)
nores = ui_payload.build_ui_result({"decision": {"request_type": "new_evaluation", "agents": ["regulations"], "constraints": {}}, "results": {"regulations": {"status": "ok", "output": REGS}}, "facts": {}, "reply": "r",
                                    "session_state": {"results": {"regulations": {"status": "ok", "output": REGS}}}}, "s")
nt = {t["key"]: t for t in nores["verdicts"]}
check("a request that compared no models says so, and one that researched no competitors says so", nt["recommended"]["title"] == "Not compared" and nt["market"]["title"] == "Not researched", nt)
failed = {t["key"]: t for t in ui_payload.build_ui_result(outcome(regs_status="error"), "s")["verdicts"]}
check("a failed legal check is never shown as a result", failed["legal"]["title"] == "Check did not finish" and failed["legal"]["tone"] == "unclear", failed["legal"])
allnear = dict(MODEL2, scenarios=[opt("n1", "near_miss", 80, 500, 1000, {"latency": "misses", "cost": "meets"})])
o3 = outcome(); o3["facts"]["model_stack_analysis"] = allnear; o3["session_state"]["results"]["model_stack_analysis"]["output"] = allnear
nt3 = {t["key"]: t for t in ui_payload.build_ui_result(o3, "s")["verdicts"]}
check("when nothing meets every target the tile says so and names the closest with its miss", nt3["recommended"]["title"] == "No option meets every target" and nt3["recommended"]["detail"] == "Closest: Model n1" and "Misses latency" in nt3["recommended"]["extra"] and nt3["recommended"]["tone"] == "caution", nt3["recommended"])
full_cov = ui_payload.verdict_tiles({"status": "checked", "short": "x", "tone": "clear", "citations": [], "dates": []}, [], False, True, {"total": 6, "verified": 6, "not_supported": 0, "could_not_check": 0, "no_source": 0}, 2)
check("all claims verified makes the market tile green", full_cov[2]["tone"] == "good" and full_cov[2]["title"] == "2 competitors researched", full_cov[2])
two_dates = ui_payload.regulatory_block({"regulations": {"status": "ok", "output": {**REGS, "applicability_status": "Dates: 2 February 2025 (Article 5); 2 August 2026 (Article 50)."}}})
check("more than one date is listed in order, each once", two_dates["dates"] == ["2 February 2025", "2 August 2026"], two_dates["dates"])

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
