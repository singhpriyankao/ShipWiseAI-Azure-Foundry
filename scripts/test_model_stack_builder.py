"""
Offline tests for model_stack_builder.py: candidate building, target checks, the shortlist,
and the option format. Synthetic price and benchmark data; no network, no sign-in.
Includes the cases the real data exposed (zero latencies, an unreachable latency target).

Run: python test_model_stack_builder.py
"""
import warnings
from datetime import date
warnings.filterwarnings("ignore")

import model_benchmarks as mb
import model_stack_builder as sb
from target_checks import check_scenario

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


TODAY = date(2026, 10, 1)
USAGE = dict(sb.DEFAULT_USAGE)  # 20,000 requests, 1,000 in, 250 out


def aa(name, index, tps, ttfat, p_in, p_out, release="2026-06-01"):
    return {"id": name, "name": name, "creator": "x", "intelligence_index": index, "output_tokens_per_second": tps,
            "first_token_seconds": ttfat, "first_answer_token_seconds": ttfat, "input_price_per_1m": p_in,
            "output_price_per_1m": p_out, "release_date": release}


def price(vendor, model, p_in, p_out):
    return {"vendor": vendor, "model": model, "input_per_1m": p_in, "output_per_1m": p_out, "source_url": "u"}


AA = [
    aa("Fast Mini", 40.0, 200.0, 0.3, 0.1, 0.4),
    aa("Smart Pro (Low)", 70.0, 100.0, 1.5, 3.0, 15.0),
    aa("Smart Pro (High)", 80.0, 80.0, 20.0, 3.0, 15.0),
    aa("Cheap Lite", 30.0, 150.0, 0.6, 0.05, 0.2),
    aa("Old Model", 20.0, 100.0, 0.5, 1.0, 2.0, release="2024-01-01"),
    aa("No Timing", 50.0, None, None, 1.0, 2.0),
]
PRICES = [price("acme", "Fast Mini", 0.1, 0.4), price("acme", "Smart Pro", 3.0, 15.0), price("beta", "Cheap Lite", 0.05, 0.2),
          price("beta", "Old Model", 1.0, 2.0), price("beta", "No Timing", 1.0, 2.0), price("beta", "Unlisted", 1.0, 2.0)]
cands = sb.build_candidates(PRICES, AA, USAGE, TODAY)
names = [(c["model"], c["configuration"]) for c in cands]

# ---- building
check("build: a model released over a year ago is not offered", not any(c["model"] == "Old Model" for c in cands), names)
check("build: a priced model with no benchmark entry is left out", not any(c["model"] == "Unlisted" for c in cands), names)
check("build: each reasoning-effort configuration is its own candidate", ("Smart Pro", "Smart Pro (Low)") in names and ("Smart Pro", "Smart Pro (High)") in names, names)
check("build: every candidate gets a unique id", len({c["id"] for c in cands}) == len(cands))
fast = next(c for c in cands if c["model"] == "Fast Mini")
# 20,000 x 12 x (1000 x 0.1 + 250 x 0.4) / 1e6 = 240,000 x 200 / 1e6 = 48
check("build: yearly cost is calculated from prices and the usage assumption", abs(fast["yearly_cost_usd"] - 48.0) < 1e-6, fast["yearly_cost_usd"])
check("build: first-answer time and full-response estimate are in milliseconds", fast["first_answer_ms"] == 300 and fast["full_response_ms"] == 1550, (fast["first_answer_ms"], fast["full_response_ms"]))
nt = next(c for c in cands if c["model"] == "No Timing")
check("build: a model with no timing data is kept but has no latency figure", nt["first_answer_ms"] is None and nt["full_response_ms"] is None)
nobench = sb.build_candidates(PRICES, [], USAGE, TODAY)
check("build: with no benchmark data at all, cost-only candidates are built", len(nobench) == 6 and all(not c["benchmarked"] for c in nobench))
check("build: a price entry with no price is skipped", sb.build_candidates([{"vendor": "a", "model": "x", "input_per_1m": None, "output_per_1m": 1}], AA, USAGE, TODAY) == [])
check("build: an unknown release date can't be shown to be recent, so it isn't offered",
      sb.build_candidates([price("a", "Dated", 1, 2)], [aa("Dated", 1, 1, 1, 1, 2, release=None)], USAGE, TODAY) == [])

# ---- targets
t = sb.parse_targets("under 800ms", "$40K annual")
check("targets: latency and budget are read", t["latency_ms"] == 800.0 and t["budget_usd"] == 40000.0, t)
t2 = sb.parse_targets("under 2 seconds", "$15K a month")
check("targets: seconds and monthly budgets are converted", t2["latency_ms"] == 2000.0 and t2["budget_usd"] == 180000.0, t2)
t3 = sb.parse_targets(None, "$15K")
check("targets: a budget with no period is given but unreadable", t3["budget_given"] and t3["budget_usd"] is None and not t3["latency_given"], t3)

# ---- evaluating
check("evaluate: first-answer time at or under the target meets", sb.evaluate(fast, t)["latency"] == "meets")
slow = next(c for c in cands if c["configuration"] == "Smart Pro (High)")
check("evaluate: over the target misses", sb.evaluate(slow, t)["latency"] == "misses")
check("evaluate: no timing data -> unknown, never 'meets'", sb.evaluate(nt, t)["latency"] == "unknown")
check("evaluate: the whole-reply time is NOT what is compared (a 2-second reply still meets 800ms if it starts in 300ms)", fast["full_response_ms"] > 800 and sb.evaluate(fast, t)["latency"] == "meets")
check("evaluate: no target given -> 'no target'", sb.evaluate(fast, sb.parse_targets(None, None)) == {"latency": "no target", "cost": "no target"})
check("evaluate: an unreadable budget -> cost unknown", sb.evaluate(fast, t3)["cost"] == "unknown")
check("evaluate: cost over budget misses", sb.evaluate(slow, sb.parse_targets(None, "$10 annual"))["cost"] == "misses")

# ---- shortlist
r = sb.shortlist(sb.build_candidates(PRICES, AA, USAGE, TODAY), t)
chosen = [(c["model"], c["configuration"]) for c in r["shortlist"]]
check("shortlist: only options meeting every given target", all(c["verdicts"]["latency"] == "meets" and c["verdicts"]["cost"] == "meets" for c in r["shortlist"]), chosen)
check("shortlist: an unmeasured model is not on it, and is counted", ("No Timing", None) not in chosen and r["counts"]["latency_unknown"] == 1, r["counts"])
check("shortlist: sorted by quality, highest first", [c["quality_percentile"] for c in r["shortlist"]] == sorted([c["quality_percentile"] for c in r["shortlist"]], reverse=True))
check("shortlist: the cheapest qualifying option is included", min(c["yearly_cost_usd"] for c in r["shortlist"]) == min(c["yearly_cost_usd"] for c in sb.build_candidates(PRICES, AA, USAGE, TODAY) if sb.evaluate(c, t) == {"latency": "meets", "cost": "meets"}))
check("shortlist: the slow high-effort configuration is a near miss, not on the list", ("Smart Pro", "Smart Pro (High)") not in chosen and any(c["configuration"] == "Smart Pro (High)" for c in r["near_misses"]), r["near_misses"])
check("shortlist: counts are reported", r["counts"]["considered"] == len(cands) and r["counts"]["meet_all_given_targets"] == len(r["shortlist"]) or r["counts"]["meet_all_given_targets"] >= len(r["shortlist"]), r["counts"])

many = [aa(f"Model {i}", 50.0 + i, 100.0, 0.2, 0.1, 0.1) for i in range(12)]
manyp = [price("acme", f"Model {i}", 0.1, 0.1) for i in range(12)]
rr = sb.shortlist(sb.build_candidates(manyp, many, USAGE, TODAY), sb.parse_targets("under 800ms", None))
check("shortlist: never more than five", len(rr["shortlist"]) <= sb.MAX_SHORTLIST, len(rr["shortlist"]))

two = [aa("Dup (Low)", 60.0, 100, 0.2, 1, 1), aa("Dup (Mid)", 65.0, 100, 0.2, 1, 1), aa("Dup (High)", 70.0, 100, 0.2, 1, 1)]
rd = sb.shortlist(sb.build_candidates([price("a", "Dup", 1, 1)], two, USAGE, TODAY), sb.parse_targets("under 800ms", None))
check("shortlist: at most two configurations of one model", len(rd["shortlist"]) == 2, len(rd["shortlist"]))

# near misses are the CLOSEST misses, not the highest-quality ones (a bug the real data exposed)
nm_aa = [aa("Close", 40.0, 100, 0.9, 0.1, 0.1), aa("Far", 99.0, 100, 50.0, 0.1, 0.1), aa("Near2", 45.0, 100, 1.2, 0.1, 0.1)]
nm = sb.shortlist(sb.build_candidates([price("a", "Close", .1, .1), price("a", "Far", .1, .1), price("a", "Near2", .1, .1)], nm_aa, USAGE, TODAY),
                  sb.parse_targets("under 800ms", None))
check("near misses: the closest to the target come first, not the highest quality", [c["model"] for c in nm["near_misses"]] == ["Close", "Near2"], [c["model"] for c in nm["near_misses"]])

none_fit = sb.shortlist(sb.build_candidates([price("a", "Far", .1, .1)], nm_aa, USAGE, TODAY), sb.parse_targets("under 10ms", None))
check("no option meets the targets -> empty shortlist, near misses still shown", none_fit["shortlist"] == [] and none_fit["counts"]["meet_all_given_targets"] == 0)

# ---- the option format the rest of the system reads
opt = sb.to_scenario(fast | {"verdicts": sb.evaluate(fast, t)}, "shortlist", USAGE)
check("scenario: cost is a yearly string the target check can read", opt["estimated_cost"] == "$48 per year", opt["estimated_cost"])
check("scenario: latency is the compared first-answer time, and says what it means", opt["estimated_latency"] == "300ms" and "first answer token" in opt["latency_means"], opt)
check("scenario: the whole-reply time is carried separately", opt["full_response_ms"] == 1550)
check("scenario: accuracy is labelled a proxy and says no benchmark measures it", opt["accuracy_is_proxy"] and "No benchmark measures accuracy" in opt["accuracy_notes"], opt["accuracy_notes"])
check("scenario: the price basis shows the calculation's inputs", "20,000 requests a month" in opt["price_basis"] and "$0.1 per 1M input tokens" in opt["price_basis"], opt["price_basis"])
verdict = check_scenario(opt, {"latency_target": "under 800ms", "budget": "$40K annual"})
check("scenario: the existing target check reads latency and cost from it", verdict["latency"].startswith("meets") and verdict["cost"].startswith("meets"), verdict)
unbench = sb.to_scenario(nobench[0] | {"verdicts": {"latency": "unknown", "cost": "no target"}}, "shortlist", USAGE)
check("scenario: with no benchmark data it says neither accuracy nor latency is known", "neither accuracy nor latency" in unbench["accuracy_notes"] and unbench["estimated_latency"] == "not available")


# ---- comparisons are written by code (an agent-written note once called a mid-priced option "higher cost than other options")
def opt(oid, name, pct, first, cost, lat_v="meets", cost_v="meets", role="shortlist"):
    return {"id": oid, "role": role, "model_stack": name, "quality_percentile": pct, "first_answer_ms": first, "latency_ms": first, "yearly_cost_usd": cost,
            "verdicts": {"latency": lat_v, "cost": cost_v}}


A, B, C = opt("opt1", "Alpha", 80.0, 700, 5000.0), opt("opt2", "Beta", 60.0, 300, 900.0), opt("opt3", "Gamma", 70.0, 500, 2000.0)
tg = sb.parse_targets("under 800ms", "$40K annual")
cmp = sb.compare_options([A, B, C], tg)
check("compare: the highest-quality option says so and by how much over the next", "ranks highest in general quality (80, against 70 for the next)" in cmp["opt1"], cmp["opt1"])
check("compare: the slowest option is placed against the fastest, with the gap", "starts replying 400ms later than the fastest (Beta, 300ms)" in cmp["opt1"], cmp["opt1"])
check("compare: the cheapest says so", "costs the least ($900 a year)" in cmp["opt2"], cmp["opt2"])
check("compare: a middle option is placed against the best and cheapest, not called 'higher' or 'lower' loosely",
      "ranks 10 percentile points below the highest (Alpha, 80) in general quality" in cmp["opt3"] and "costs $1.1K more a year than the cheapest (Beta, $900)" in cmp["opt3"], cmp["opt3"])
check("compare: the fastest says so", "starts replying soonest (300ms)" in cmp["opt2"], cmp["opt2"])
check("compare: every sentence starts the same way and names the others as the basis", all(v.startswith("Compared with the other options shown, it ") for v in cmp.values()))

tie = sb.compare_options([opt("a", "A", 70.0, 300, 100.0), opt("b", "B", 70.0, 300, 100.0)], tg)
check("compare: a tie is reported as a tie, not as one being better", "ties for the highest general quality" in tie["a"] and "starts replying soonest" in tie["a"] and "costs the least" in tie["a"], tie["a"])

alone = sb.compare_options([A], tg)
check("compare: a single option has nothing to be compared with", alone["opt1"] == "", alone)

noq = sb.compare_options([opt("a", "A", None, 300, 100.0), opt("b", "B", 60.0, 400, 200.0)], tg)
check("compare: an option with no quality ranking says so instead of inventing one", "has no general quality ranking to compare" in noq["a"], noq["a"])
check("compare: ...and is not used as the 'next' for another option", "ranks highest" not in noq["b"] and "ties" not in noq["b"], noq["b"])

miss = sb.compare_options([opt("a", "A", 60.0, 862, 54.0, lat_v="misses", role="near_miss"), opt("b", "B", 59.0, 472, 540.0)], tg)
check("compare: a missed latency target is stated with the exact overshoot", "misses the latency target by 62ms (862ms against 800ms)" in miss["a"], miss["a"])
over = sb.compare_options([opt("a", "A", 60.0, 100, 50000.0, cost_v="misses"), opt("b", "B", 59.0, 200, 540.0)], tg)
check("compare: a missed budget is stated with the exact overshoot", "is over budget by $10K a year" in over["a"], over["a"])
check("compare: nothing is said about a target that was met", "misses" not in cmp["opt1"] and "over budget" not in cmp["opt1"])

check("label: vendor, model and only the bracketed setting", sb.option_label({"vendor": "anthropic", "model": "Claude Haiku 4.5", "configuration": "Claude 4.5 Haiku (Non-reasoning)"}) == "Anthropic Claude Haiku 4.5 (Non-reasoning)")
check("label: a configuration with no bracket adds nothing", sb.option_label({"vendor": "openai", "model": "gpt-9", "configuration": "GPT 9"}) == "OpenAI gpt-9")
check("label: no configuration (cost-only) adds nothing", sb.option_label({"vendor": "google", "model": "gemini-9", "configuration": None}) == "Google gemini-9")


# ---- the latency that is compared depends on how the reply is delivered
chat = sb.parse_targets("under 800ms", None, "streamed_chat")
batch = sb.parse_targets("under 800ms", None, "single_response")
check("targets: a streamed chat compares the time until the reply starts", chat["latency_basis"] == "first_answer")
check("targets: a single-response feature compares the time until the reply is complete", batch["latency_basis"] == "full_response")
check("targets: the default is a streamed chat", sb.parse_targets("under 800ms", None)["latency_basis"] == "first_answer")
fast_start = next(c for c in cands if c["model"] == "Fast Mini")   # first answer 300ms, whole reply 1550ms
check("evaluate: the same option meets 800ms as a chat but not as a single response",
      sb.evaluate(fast_start, chat)["latency"] == "meets" and sb.evaluate(fast_start, batch)["latency"] == "misses", (sb.evaluate(fast_start, chat), sb.evaluate(fast_start, batch)))
check("evaluate: a single-response feature with no output-speed data is unknown, not met",
      sb.evaluate(dict(fast_start, full_response_ms=None), batch)["latency"] == "unknown")
sc_chat = sb.to_scenario(dict(fast_start, verdicts=sb.evaluate(fast_start, chat)), "shortlist", USAGE, chat)
sc_batch = sb.to_scenario(dict(fast_start, verdicts=sb.evaluate(fast_start, batch)), "shortlist", USAGE, batch)
check("scenario: the compared latency and what it means follow the reply style",
      sc_chat["estimated_latency"] == "300ms" and "reply starts" in sc_chat["latency_means"] and sc_batch["estimated_latency"] == "1550ms" and "whole reply is complete" in sc_batch["latency_means"], (sc_chat["estimated_latency"], sc_batch["estimated_latency"]))
batch_cmp = sb.compare_options([opt("a", "A", 60.0, 900, 100.0), opt("b", "B", 60.0, 400, 100.0)], batch)
check("compare: wording says 'finishes its reply' when the whole reply is what is compared", "finishes its reply 500ms later than the fastest" in batch_cmp["a"], batch_cmp["a"])
check("compare: wording says 'starts replying' for a chat", "starts replying 500ms later" in sb.compare_options([opt("a", "A", 60.0, 900, 100.0), opt("b", "B", 60.0, 400, 100.0)], chat)["a"])
shortlisted = sb.shortlist(sb.build_candidates(PRICES, AA, USAGE, TODAY), batch)
check("shortlist: with a single-response target of 800ms, nothing that takes longer to finish is on it", all(c["full_response_ms"] is not None and c["full_response_ms"] <= 800 for c in shortlisted["shortlist"]), [c["full_response_ms"] for c in shortlisted["shortlist"]])

# ---- a code-written description of one option (replaces the agent's note)
d1 = sb.describe_option({"model_stack": "Anthropic Claude Haiku 4.5 (Non-reasoning)", "verdicts": {"latency": "meets", "cost": "meets"}})
check("describe: a non-reasoning setting is described as answering directly", d1.startswith("Answers directly") and "meets the latency target and fits the budget" in d1, d1)
d2 = sb.describe_option({"model_stack": "OpenAI gpt-6.1-sol (Low)", "verdicts": {"latency": "misses", "cost": "meets"}})
check("describe: an effort setting is described, naming the effort", "Thinks before it answers; the low effort setting controls how long it may think." in d2 and "misses the latency target and fits the budget" in d2, d2)
d3 = sb.describe_option({"model_stack": "Anthropic Claude Sonnet 5.5 (Adaptive Reasoning, Xhigh Effort)", "verdicts": {"latency": "unknown", "cost": "misses"}})
check("describe: the highest effort word wins and unknown/over-budget verdicts are stated", "xhigh effort" in d3 and "could not be compared on latency and is over budget" in d3, d3)
check("describe: a model with no setting gets only the verdicts", sb.describe_option({"model_stack": "Acme Fast Mini", "verdicts": {"latency": "meets", "cost": "no target"}}) == "It meets the latency target.")
check("describe: no targets and no setting gives an empty description, not an error", sb.describe_option({"model_stack": "Acme X", "verdicts": {"latency": "no target", "cost": "no target"}}) == "")
check("describe: it never compares options", not any(w in d1 + d2 + d3 for w in (" than ", "faster", "cheaper", "higher", "better")))

# ---- ordinals ("83th" was printed before)
check("ordinal: 1, 2, 3 and 4", [sb.ordinal(n) for n in (1, 2, 3, 4)] == ["1st", "2nd", "3rd", "4th"])
check("ordinal: the teens are all 'th'", [sb.ordinal(n) for n in (11, 12, 13, 14)] == ["11th", "12th", "13th", "14th"])
check("ordinal: 21, 22, 23, 83 and 100", [sb.ordinal(n) for n in (21, 22, 23, 83, 100)] == ["21st", "22nd", "23rd", "83rd", "100th"])
check("ordinal: a float is rounded first", sb.ordinal(69.0) == "69th" and sb.ordinal(82.6) == "83rd")
check("scenario: the accuracy note uses the right ordinal", "83rd percentile" in sb.to_scenario(dict(fast_start, quality_percentile=82.6, verdicts=sb.evaluate(fast_start, chat)), "shortlist", USAGE, chat)["accuracy_notes"])

# ---- what the company's own reports have taught is applied to the forecast
import learning as L


def reported(rid, ms, company="acme", label=None, category="classification_tagging", cost=None, volume_basis="assumed"):
    o = {"id": "o", "reported_at": "2026-11-20T00:00:00+00:00", "model_differs": False, "confirmed_by_pm": True,
         "fields": {"latency": {"value_ms": ms, "kind": "first_answer", "statistic": "median", "quote": "x"}}}
    if cost is not None:
        o["fields"]["cost"] = {"yearly_usd": cost, "amount_usd": cost / 12, "period": "month", "quote": "x"}
    o["fields"]["accuracy"] = {"value_percent": 80, "how_measured": "a sample", "quote": "x"}
    return {"id": rid, "type": "accepted_recommendation", "company_id": company, "user_id": "u", "feature": {"category": category, "description": "d"},
            "constraints_given": {}, "usage_assumption": {"requests_per_month": 20000, "volume_basis": volume_basis},
            "chosen": {"label": label, "forecast": {"latency_ms": 300, "latency_means": "time until the first answer token arrives (the reply starts appearing)", "yearly_cost_usd": 48.0}},
            "follow_ups": [], "outcomes": [o]}


fast_label = sb.option_label(next(c for c in cands if c["model"] == "Fast Mini"))
three_slow = [reported(str(i), 450, label=fast_label) for i in range(3)]          # 450 reported against a 300 forecast: 1.5x
base = sb.build_candidates(PRICES, AA, USAGE, TODAY)
sb.apply_calibration(base, L.Calibration(three_slow, "acme"), "classification_tagging", "first_answer")
fast_c = next(c for c in base if c["model"] == "Fast Mini")
check("learning: three agreeing reports move the compared latency by their median ratio (300ms becomes 450ms)", fast_c["first_answer_ms"] == 450, fast_c["first_answer_ms"])
check("learning: the original forecast is kept beside it, so corrections can't compound", fast_c["raw_forecast"]["latency_ms"] == 300)
check("learning: the sentence says what changed and from how many reports", any("3 reports from your company" in l and "300ms to 450ms" in l for l in fast_c["learned"]), fast_c["learned"])
other = next(c for c in base if c["model"] == "Smart Pro")
smart_before = next(c for c in sb.build_candidates(PRICES, AA, USAGE, TODAY) if c["model"] == "Smart Pro")["first_answer_ms"]
check("learning: another model setting is untouched", other["first_answer_ms"] == smart_before and other["learned"] == [], (other["first_answer_ms"], smart_before, other["learned"]))
check("learning: the other latency (whole reply) is left alone, since the reports were about the first words", fast_c["full_response_ms"] == 1550)
t_800 = sb.parse_targets("under 400ms", None)
checked = sb.shortlist(base, t_800)
check("learning: a corrected figure can take an option off the shortlist (450ms no longer meets under 400ms)", all(c["model"] != "Fast Mini" for c in checked["shortlist"]), [(c["model"], c["first_answer_ms"]) for c in checked["shortlist"]])
uncorrected = sb.build_candidates(PRICES, AA, USAGE, TODAY)
sb.apply_calibration(uncorrected, None, "classification_tagging", "first_answer")
check("learning: with no calibration nothing changes, and the raw forecast is still recorded", next(c for c in uncorrected if c["model"] == "Fast Mini")["first_answer_ms"] == 300 and next(c for c in uncorrected if c["model"] == "Fast Mini")["raw_forecast"]["latency_ms"] == 300)

thin = sb.build_candidates(PRICES, AA, USAGE, TODAY)
sb.apply_calibration(thin, L.Calibration(three_slow[:2], "acme"), "classification_tagging", "first_answer")
thin_c = next(c for c in thin if c["model"] == "Fast Mini")
check("learning: two reports change nothing, but the PM is told what has been reported and why it wasn't applied", thin_c["first_answer_ms"] == 300 and any("2 of the 3 separate features needed" in l for l in thin_c["learned"]), thin_c["learned"])

disagree = [reported("a", 150, label=fast_label), reported("b", 450, label=fast_label), reported("c", 1200, label=fast_label)]
dis = sb.build_candidates(PRICES, AA, USAGE, TODAY)
sb.apply_calibration(dis, L.Calibration(disagree, "acme"), "classification_tagging", "first_answer")
check("learning: three reports that disagree don't move the forecast", next(c for c in dis if c["model"] == "Fast Mini")["first_answer_ms"] == 300)

others = sb.build_candidates(PRICES, AA, USAGE, TODAY)
sb.apply_calibration(others, L.Calibration(three_slow, "beta"), "classification_tagging", "first_answer")
check("learning: another company's reports are not applied", next(c for c in others if c["model"] == "Fast Mini")["first_answer_ms"] == 300)

whole = sb.build_candidates(PRICES, AA, USAGE, TODAY)
sb.apply_calibration(whole, L.Calibration(three_slow, "acme"), "classification_tagging", "full_response")
check("learning: reports about the first words aren't applied to a whole-reply latency", next(c for c in whole if c["model"] == "Fast Mini")["full_response_ms"] == 1550)

priced = [reported(str(i), 450, label=fast_label, cost=48.0 * 1.25, volume_basis="stated") for i in range(3)]
costed = sb.build_candidates(PRICES, AA, USAGE, TODAY)
sb.apply_calibration(costed, L.Calibration(priced, "acme"), "classification_tagging", "first_answer")
cc = next(c for c in costed if c["model"] == "Fast Mini")
check("learning: three agreeing cost reports correct the yearly cost, with the original kept", abs(cc["yearly_cost_usd"] - 60.0) < 1e-6 and cc["raw_forecast"]["yearly_cost_usd"] == 48.0, (cc["yearly_cost_usd"], cc["raw_forecast"]))
check("learning: accuracy evidence appears as a sentence and corrects nothing", any("80%" in l and "not by ShipWise" in l for l in cc["learned"]), cc["learned"])
sc = sb.to_scenario(dict(fast_c, verdicts=sb.evaluate(fast_c, sb.parse_targets("under 800ms", None))), "shortlist", USAGE, sb.parse_targets("under 800ms", None))
check("learning: the option carries its learned sentences and the raw forecast", sc["learned"] == fast_c["learned"] and sc["raw_forecast"]["latency_ms"] == 300)

# ---- an option that moves on or off the shortlist because of the reports is explained
t_400 = sb.parse_targets("under 400ms", None)
bc = sb.build_candidates(PRICES, AA, USAGE, TODAY)
sb.apply_calibration(bc, L.Calibration(three_slow, "acme"), "classification_tagging", "first_answer")
sb.shortlist(bc, t_400)
changes = sb.calibration_changes(bc, t_400)
check("changes: an option the reports pushed over the PM's target is named, with before and after", any("Fast Mini" in c and "met your target on the benchmark forecast and now misses it" in c for c in changes), changes)
t_900 = sb.parse_targets("under 900ms", None)
bc2 = sb.build_candidates(PRICES, AA, USAGE, TODAY)
sb.apply_calibration(bc2, L.Calibration(three_slow, "acme"), "classification_tagging", "first_answer")
sb.shortlist(bc2, t_900)
check("changes: a correction that doesn't change any verdict is not reported (450ms still meets under 900ms)", not any("Fast Mini" in c for c in sb.calibration_changes(bc2, t_900)), sb.calibration_changes(bc2, t_900))
bc3 = sb.build_candidates(PRICES, AA, USAGE, TODAY)
sb.apply_calibration(bc3, None, "classification_tagging", "first_answer")
sb.shortlist(bc3, t_400)
check("changes: with no calibration there is nothing to report", sb.calibration_changes(bc3, t_400) == [])
check("changes: never more than the limit", len(sb.calibration_changes(bc, t_400, limit=0)) == 0)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
