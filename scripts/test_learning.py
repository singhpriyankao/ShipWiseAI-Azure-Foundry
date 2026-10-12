"""
Offline tests for learning.py: one observation per feature, like-for-like comparisons, the three-features rule,
the agreement rule, scope, the plain sentences, and "did it meet your own targets". Synthetic records; no model,
no network.

Run: python test_learning.py
"""
import warnings
warnings.filterwarnings("ignore")

import learning as L

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


LABEL = "Anthropic Claude Haiku 4.5 (Non-reasoning)"
FIRST = "time until the first answer token arrives (the reply starts appearing)"
FULL = "time until the whole reply is complete"


def outcome(latency=None, cost=None, volume=None, accuracy=None, when="2026-11-20T00:00:00+00:00", differs=False, confirmed=True):
    fields = {}
    if latency is not None:
        ms, kind, stat = latency
        fields["latency"] = {"value_ms": ms, "kind": kind, "statistic": stat, "quote": "x"}
    if cost is not None:
        fields["cost"] = {"yearly_usd": cost, "amount_usd": cost / 12, "period": "month", "quote": "x"}
    if volume is not None:
        fields["volume"] = {"requests_per_month": volume, "quote": "x"}
    if accuracy is not None:
        value, how = accuracy
        fields["accuracy"] = {"value_percent": value, "how_measured": how, "quote": "x"}
    return {"id": "o", "reported_at": when, "fields": fields, "model_differs": differs, "confirmed_by_pm": confirmed}


def record(rid="r", company="acme", label=LABEL, category="classification_tagging", means=FIRST, lat=500, cost=1000.0,
           assumed_volume=20000, volume_basis="assumed", outcomes=(), constraints=None):
    return {"id": rid, "type": "accepted_recommendation", "company_id": company, "user_id": "u", "status": "awaiting_outcome",
            "feature": {"category": category, "description": "d"},
            "constraints_given": constraints or {},
            "usage_assumption": {"requests_per_month": assumed_volume, "volume_basis": volume_basis},
            "chosen": {"label": label, "forecast": {"latency_ms": lat, "latency_means": means, "yearly_cost_usd": cost}},
            "follow_ups": [], "outcomes": list(outcomes)}


# ---- one observation per feature
three = record(outcomes=[outcome(latency=(600, "first_answer", "median"), when="2026-11-01T00:00:00+00:00"),
                         outcome(latency=(700, "first_answer", "median"), when="2026-12-01T00:00:00+00:00"),
                         outcome(latency=(650, "first_answer", "median"), when="2026-10-15T00:00:00+00:00")])
obs = L.latest_observations([three])
check("three reports on one project are one observation", len(obs) == 1)
check("...and it is the latest by date, not by position in the list", obs[0][1]["fields"]["latency"]["value_ms"] == 700, obs[0][1]["fields"])
check("a report about a different model is never used", L.latest_observations([record(outcomes=[outcome(latency=(600, "first_answer", "median"), differs=True)])]) == [])
check("a report the PM did not confirm is never used", L.latest_observations([record(outcomes=[outcome(latency=(600, "first_answer", "median"), confirmed=False)])]) == [])
check("a recommendation with no reports gives nothing", L.latest_observations([record()]) == [])
mixed = record(outcomes=[outcome(latency=(900, "first_answer", "median"), differs=True, when="2026-12-01T00:00:00+00:00"), outcome(latency=(600, "first_answer", "median"), when="2026-11-01T00:00:00+00:00")])
check("a later report about another model doesn't hide an earlier usable one", L.latest_observations([mixed])[0][1]["fields"]["latency"]["value_ms"] == 600)
check("the company filter keeps only that company's records", len(L.latest_observations([three, record("r2", "beta", outcomes=three["outcomes"])], "acme")) == 1)

# ---- like-for-like latency
r1 = record(outcomes=[outcome(latency=(600, "first_answer", "median"))])
obs, why = L.latency_observation(r1, r1["outcomes"][0])
check("latency: same kind and a median compare as a ratio of reported to forecast", obs["ratio"] == 1.2 and obs["basis"] == "first_answer", (obs, why))
r2 = record(outcomes=[outcome(latency=(1400, "full_response", "median"))])
check("latency: the whole reply reported against a first-words forecast is not compared", L.latency_observation(r2, r2["outcomes"][0])[0] is None and "kind" in L.latency_observation(r2, r2["outcomes"][0])[1])
r3 = record(outcomes=[outcome(latency=(900, "first_answer", "p95"))])
check("latency: a 95th-percentile figure is not compared with a median forecast", L.latency_observation(r3, r3["outcomes"][0])[0] is None and "median" in L.latency_observation(r3, r3["outcomes"][0])[1])
r4 = record(outcomes=[outcome(latency=(600, "unspecified", "median"))])
check("latency: a report that doesn't say which kind is not guessed at", L.latency_observation(r4, r4["outcomes"][0])[0] is None)
r5 = record(outcomes=[outcome(latency=(600, "first_answer", "average"))])
check("latency: an average is accepted", L.latency_observation(r5, r5["outcomes"][0])[0]["ratio"] == 1.2)
r6 = record(means=FULL, lat=2000, outcomes=[outcome(latency=(2600, "full_response", "median"))])
check("latency: a whole-reply forecast is compared with a whole-reply report", L.latency_observation(r6, r6["outcomes"][0])[0]["basis"] == "full_response" and L.latency_observation(r6, r6["outcomes"][0])[0]["ratio"] == 1.3)
r7 = record(outcomes=[outcome(cost=500.0)])
check("latency: no latency reported means nothing to compare", L.latency_observation(r7, r7["outcomes"][0])[0] is None)
r8 = record(outcomes=[outcome(latency=(600, "first_answer", "median"))])
r8["chosen"]["forecast_raw"] = {"latency_ms": 400, "yearly_cost_usd": 1000.0}
r8["chosen"]["forecast"]["latency_ms"] = 560   # what was shown after an earlier correction
check("latency: the ratio is against the benchmark forecast, not an already-corrected one (corrections must not compound)", L.latency_observation(r8, r8["outcomes"][0])[0]["ratio"] == 1.5)

# ---- cost, scaled to the volume that really happened
c1 = record(cost=1000.0, assumed_volume=20000, outcomes=[outcome(cost=2400.0, volume=40000)])
obs, why = L.cost_observation(c1, c1["outcomes"][0])
check("cost: the forecast is first scaled to the reported volume (double the volume, double the forecast), so volume error is not mistaken for price error",
      obs["forecast_usd"] == 2000.0 and abs(obs["ratio"] - 1.2) < 1e-9, (obs, why))
c2 = record(cost=1000.0, volume_basis="stated", outcomes=[outcome(cost=1500.0)])
check("cost: a volume the PM stated is already in the forecast, so no report of it is needed", L.cost_observation(c2, c2["outcomes"][0])[0]["ratio"] == 1.5)
c3 = record(cost=1000.0, volume_basis="assumed", outcomes=[outcome(cost=1500.0)])
obs, why = L.cost_observation(c3, c3["outcomes"][0])
check("cost: an assumed volume and no reported one can't separate price from volume, so it is not used", obs is None and "volume" in why, why)
check("cost: no cost reported means nothing to compare", L.cost_observation(c1, outcome(volume=40000)) == (None, "no cost figure to compare"))

# ---- the rule for applying a correction
check("stats: no ratios", L.ratio_stats([])["applies"] is False and L.ratio_stats([])["n"] == 0)
s = L.ratio_stats([1.4, 1.5])
check("stats: two separate features is not enough, and says how many are needed", s["applies"] is False and "2 of the 3" in s["why"], s)
s = L.ratio_stats([1.3, 1.4, 1.5])
check("stats: three agreeing features apply, at the median", s["applies"] and s["median"] == 1.4 and s["min"] == 1.3 and s["max"] == 1.5, s)
s = L.ratio_stats([0.4, 1.0, 2.6])
check("stats: three that disagree widely do not apply", s["applies"] is False and "disagree" in s["why"], s)
check("stats: the edge of the agreement band still counts (half to double the median)", L.ratio_stats([0.5, 1.0, 2.0])["applies"] is True)
check("stats: just outside the band does not", L.ratio_stats([0.49, 1.0, 2.0])["applies"] is False)
check("stats: an even number takes the middle of the two", abs(L.ratio_stats([1.0, 1.2, 1.4, 1.6])["median"] - 1.3) < 1e-9)

# ---- per model setting, per company
def with_latency(rid, company, ms, label=LABEL, **kw):
    return record(rid, company, label=label, outcomes=[outcome(latency=(ms, "first_answer", "median"))], **kw)


records = [with_latency("a", "acme", 650), with_latency("b", "acme", 700), with_latency("c", "acme", 600), with_latency("d", "beta", 5000),
           with_latency("e", "acme", 900, label="OpenAI gpt-6-sol (Non-reasoning)")]
cal = L.Calibration(records, "acme")
info = cal.for_option(LABEL, "classification_tagging")
check("calibration: three of the company's features on this setting give a correction at the median ratio", info["latency"]["first_answer"]["applies"] and info["latency"]["first_answer"]["median"] == 1.3, info["latency"])
check("calibration: another company's report is never mixed in", info["latency"]["first_answer"]["n"] == 3)
check("calibration: another model setting is never mixed in", cal.for_option("OpenAI gpt-6-sol (Non-reasoning)")["latency"]["first_answer"]["n"] == 1)
check("calibration: a setting with no reports says so", cal.for_option("Nobody (Low)")["reports"] == 0 and cal.for_option("Nobody (Low)")["latency"] == {})
mixed_basis = L.Calibration([with_latency("a", "acme", 650), record("b", "acme", means=FULL, lat=2000, outcomes=[outcome(latency=(2400, "full_response", "median"))])], "acme").for_option(LABEL)
check("calibration: first-words and whole-reply ratios are kept apart", set(mixed_basis["latency"]) == {"first_answer", "full_response"})
skipped = L.Calibration([record("a", "acme", outcomes=[outcome(latency=(900, "first_answer", "p95"))])], "acme").for_option(LABEL)
check("calibration: a report that couldn't be compared is counted as unused, not lost", skipped["unused"]["latency"] == 1 and skipped["latency"] == {})
cost_cal = L.Calibration([record(str(i), "acme", volume_basis="stated", cost=1000.0, outcomes=[outcome(cost=1300.0 + i * 20)]) for i in range(3)], "acme").for_option(LABEL)
check("calibration: three agreeing cost reports give a cost correction", cost_cal["cost"]["applies"] and 1.2 < cost_cal["cost"]["median"] < 1.4, cost_cal["cost"])

# ---- accuracy is evidence, never a correction
acc = [record(str(i), "acme", outcomes=[outcome(accuracy=(80 + i, f"{100 + i} labelled tickets"))]) for i in range(3)]
acc.append(record("x", "acme", category="summarization", outcomes=[outcome(accuracy=(55, "a sample"))]))
a = L.Calibration(acc, "acme").for_option(LABEL, "classification_tagging")["accuracy"]
check("accuracy: reports are a range with a count, from the same kind of feature only", a["n"] == 3 and a["min"] == 80 and a["max"] == 82 and a["measured_by_how"] == 3, a)
check("accuracy: nothing about it is a correction factor", "applies" not in a and "median" in a)
check("accuracy: with no reports for this kind of feature there is nothing", L.Calibration(acc, "acme").for_option(LABEL, "chat_assistant")["accuracy"] == {"n": 0})

# ---- pooled evidence across companies
pooled2 = [record(str(i), f"c{i}", outcomes=[outcome(accuracy=(70 + i, "t"))]) for i in range(2)]
check("pooled: two companies are too few to show anonymously", L.Calibration(pooled2, "acme").for_option(LABEL, "classification_tagging")["pooled_accuracy"] == {"n": 0})
pooled3 = [record(str(i), f"c{i}", outcomes=[outcome(accuracy=(70 + i, "t"))]) for i in range(3)]
p = L.Calibration(pooled3, "acme").for_option(LABEL, "classification_tagging")["pooled_accuracy"]
check("pooled: three different companies show an anonymous range", p["n"] == 3 and p["min"] == 70 and p["max"] == 72, p)
same_company = [record(str(i), "c0", outcomes=[outcome(accuracy=(70 + i, "t"))]) for i in range(3)]
check("pooled: three reports from ONE company are not 'across companies' and are not shown", L.Calibration(same_company, "acme").for_option(LABEL, "classification_tagging")["pooled_accuracy"] == {"n": 0})
check("pooled: it carries no company names or text, only numbers", set(p) == {"n", "min", "max", "median", "measured_by_how"})

# ---- the sentences
money = lambda d: f"${d:,.0f}"
applied = L.ratio_stats([1.3, 1.4, 1.5])
line = L.latency_line(applied, 472, 661)
check("sentence: an applied latency correction names the count, the ratio, the range and both figures", "3 reports" in line and "1.40x" in line and "1.30x to 1.50x" in line and "472ms to 661ms" in line, line)
thin = L.latency_line(L.ratio_stats([1.4]), 472, 472)
check("sentence: a single report says it changed nothing and why", "1 report " in thin and "1 of the 3 separate features needed" in thin and "not changed" in thin, thin)
check("sentence: no reports gives no sentence", L.latency_line(L.ratio_stats([]), 1, 1) is None and L.cost_line(L.ratio_stats([]), 1, 1, money) is None)
cl = L.cost_line(L.ratio_stats([1.2, 1.3, 1.25]), 1000, 1250, money)
check("sentence: an applied cost correction says model cost and both figures", "model cost" in cl and "$1,000 to $1,250 a year" in cl, cl)
disagree = L.latency_line(L.ratio_stats([0.4, 1.0, 2.6]), 472, 472)
check("sentence: reports that disagree are shown, not applied", "disagree" in disagree and "not changed" in disagree, disagree)
al = L.accuracy_line({"n": 3, "min": 80, "max": 82, "median": 81, "measured_by_how": 3}, {"n": 0}, "classification_tagging")
check("sentence: accuracy is a range with a count and says who measured it", len(al) == 1 and "80% to 82%" in al[0] and "3 reports" in al[0] and "not by ShipWise" in al[0] and "classification tagging" in al[0], al)
al1 = L.accuracy_line({"n": 1, "min": 82, "max": 82, "median": 82, "measured_by_how": 1}, {"n": 0}, "summarization")
check("sentence: one accuracy report is a single figure", "82%" in al1[0] and "1 report," in al1[0] and "to" not in al1[0].split("features:")[1].split("(")[0], al1)
alp = L.accuracy_line({"n": 0}, {"n": 3, "min": 70, "max": 72, "median": 71, "measured_by_how": 3}, "summarization")
check("sentence: pooled accuracy says it is anonymous evidence and adjusts nothing", "anonymously" in alp[0] and "nothing is adjusted" in alp[0], alp)
check("sentence: no accuracy reports gives no lines", L.accuracy_line({"n": 0}, {"n": 0}, "x") == [])

# ---- did the shipped feature meet the PM's own targets?
targets = {"accuracy_target": "at least 85%", "latency_target": "under 800ms", "budget": "$40K annual"}
good = record("g", constraints=targets, outcomes=[outcome(accuracy=(88, "t"), latency=(600, "first_answer", "median"), cost=12000.0)])
bad_acc = record("b", constraints=targets, outcomes=[outcome(accuracy=(80, "t"), latency=(600, "first_answer", "median"), cost=12000.0)])
none_set = record("n", constraints={}, outcomes=[outcome(accuracy=(80, "t"))])
nothing_reported = record("z", constraints=targets, outcomes=[outcome(volume=1000)])
dq = L.decision_quality([good, bad_acc, none_set, nothing_reported])
check("quality: counts every feature with a report, and only those with something comparable", dq["features_with_reports"] == 4 and dq["comparable"] == 2, dq)
check("quality: one met everything the PM set, one missed accuracy, so half", dq["met_all"] == 1 and dq["rate"] == 0.5, dq)
check("quality: it breaks down by target", dq["by_target"]["accuracy"] == {"met": 1, "of": 2} and dq["by_target"]["latency"] == {"met": 2, "of": 2} and dq["by_target"]["cost"] == {"met": 2, "of": 2}, dq["by_target"])
check("quality: with nothing comparable the rate is None, not zero", L.decision_quality([none_set, nothing_reported])["rate"] is None)
slow = record("s", constraints=targets, outcomes=[outcome(latency=(900, "first_answer", "median"))])
check("quality: a latency over the PM's own target is a miss", L.decision_quality([slow])["by_target"]["latency"] == {"met": 0, "of": 1})
wrong_kind = record("w", constraints=targets, outcomes=[outcome(latency=(900, "full_response", "median"))])
check("quality: a latency of a different kind than the one compared is not used", L.decision_quality([wrong_kind])["comparable"] == 0)
unspec = record("u", constraints=targets, outcomes=[outcome(latency=(700, "unspecified", "median"))])
check("quality: against the PM's own target, an unspecified kind is given the benefit of the doubt", L.decision_quality([unspec])["by_target"]["latency"] == {"met": 1, "of": 1})
over = record("o", constraints=targets, outcomes=[outcome(cost=60000.0)])
check("quality: a cost over the PM's own budget is a miss", L.decision_quality([over])["by_target"]["cost"] == {"met": 0, "of": 1})
check("quality: filtered to one company", L.decision_quality([good, record("g2", "beta", constraints=targets, outcomes=good["outcomes"])], "acme")["features_with_reports"] == 1)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
