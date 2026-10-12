"""
Offline tests for target_checks.py. The first group uses the exact figures the
Model Stack agent returned in the 2026-10-01 live run, where the writing step
wrongly said Claude 2 met a 90% accuracy target. No model calls.

Run: python test_target_checks.py
"""
from target_checks import add_target_checks, apply_computed_costs, check_scenario, parse_accuracy, parse_cost, parse_latency

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


TARGETS = {"accuracy_target": "90", "latency_target": "800ms", "budget": "$40K annual"}


def verdicts(scenario, targets=TARGETS):
    return {k: v.split(":")[0] for k, v in check_scenario(scenario, targets).items()}


# ---- the live case
gpt4 = {"estimated_cost": "$35K annually", "estimated_latency": "~500ms", "accuracy_notes": "~90-92%"}
claude2 = {"estimated_cost": "$25K annually", "estimated_latency": "300-700ms", "accuracy_notes": "~85-88%"}
llama = {"estimated_cost": "~$15K annually", "estimated_latency": "~800ms", "accuracy_notes": "~75-85%"}
check("GPT-4 meets accuracy, latency, cost", verdicts(gpt4) == {"accuracy": "meets", "latency": "meets", "cost": "meets"}, verdicts(gpt4))
check("Claude 2 MISSES the 90% accuracy target", verdicts(claude2)["accuracy"] == "misses", verdicts(claude2))
check("Claude 2 range 300-700ms meets 800ms", verdicts(claude2)["latency"] == "meets")
check("Llama 800ms meets 'under 800ms' boundary", verdicts(llama)["latency"] == "meets")
check("Llama misses accuracy", verdicts(llama)["accuracy"] == "misses")

# ---- ranges that straddle the target are not called a pass
check("accuracy range 88-92% vs 90 -> cannot be compared", verdicts({"accuracy_notes": "88-92%"}, {"accuracy_target": "90%"})["accuracy"] == "cannot be compared")
check("latency range 600-1000ms vs 800ms -> cannot be compared", verdicts({"estimated_latency": "600-1000ms"}, {"latency_target": "800ms"})["latency"] == "cannot be compared")
check("latency 1.2 seconds misses 800ms", verdicts({"estimated_latency": "1.2 seconds"}, {"latency_target": "800ms"})["latency"] == "misses")
check("latency 500ms vs 'under 2 seconds' meets", verdicts({"estimated_latency": "500ms"}, {"latency_target": "under 2 seconds"})["latency"] == "meets")

# ---- cost periods
check("monthly option vs annual budget is converted", parse_cost("$2K per month") == (24000.0, 24000.0), parse_cost("$2K per month"))
check("monthly $5K (=$60K/yr) misses $40K annual", verdicts({"estimated_cost": "$5K per month"}, {"budget": "$40K annual"})["cost"] == "misses")
check("option cost with no period -> cannot be compared", verdicts({"estimated_cost": "$35K"}, {"budget": "$40K annual"})["cost"] == "cannot be compared")
check("target budget with no period -> cannot be compared", verdicts({"estimated_cost": "$35K annually"}, {"budget": "$40K"})["cost"] == "cannot be compared")
check("per-token price is never compared to a budget", parse_cost("$5 per 1M tokens, annually") is None)
check("range $15K-$20K annually reads as a range", parse_cost("$15K-$20K annually") == (15000.0, 20000.0), parse_cost("$15K-$20K annually"))
check("range $15-20K annually shares the K", parse_cost("$15-20K annually") == (15000.0, 20000.0), parse_cost("$15-20K annually"))

# ---- ambiguity and missing data always resolve to 'cannot be compared'
check("two accuracy figures -> not parsed", parse_accuracy("90% on benchmark A, 78% on benchmark B") is None)
check("no percent sign -> not parsed", parse_accuracy("high accuracy") is None)
check("two latency figures -> not parsed", parse_latency("500ms typical, 2s worst case") is None)
check("'sessions' is not a seconds unit", parse_latency("handles 100 sessions") is None)
check("missing field -> cannot be compared", verdicts({}, {"accuracy_target": "90%"})["accuracy"] == "cannot be compared")
check("only targets the PM gave are checked", set(check_scenario(gpt4, {"latency_target": "800ms"})) == {"latency"})
check("no targets -> no checks", check_scenario(gpt4, {}) == {})

# ---- plumbing
original = {"scenarios": [dict(gpt4, model_stack="gpt-4")], "grounding_status": "grounded"}
with_checks = add_target_checks(original, TARGETS)
check("target_check added to each scenario", "target_check" in with_checks["scenarios"][0])
check("saved original is not mutated", "target_check" not in original["scenarios"][0])
check("malformed output passes through untouched", add_target_checks({"error": "x"}, TARGETS) == {"error": "x"})

# ---- assumed request volume
base = {"scenarios": [dict(gpt4, model_stack="gpt-4")]}
assumed = add_target_checks(dict(base, volume_assumption={"requests_per_month": 20000, "basis": "assumed"}), TARGETS)
check("assumed volume is noted on a cost verdict", "assumes 20000 requests per month" in assumed["scenarios"][0]["target_check"]["cost"], assumed["scenarios"][0]["target_check"]["cost"])
check("assumed-volume note does not change the verdict", assumed["scenarios"][0]["target_check"]["cost"].startswith("meets"))
stated = add_target_checks(dict(base, volume_assumption={"requests_per_month": 20000, "basis": "stated"}), TARGETS)
check("stated volume adds no note", "assumes" not in stated["scenarios"][0]["target_check"]["cost"])
none_given = add_target_checks(base, TARGETS)
check("no volume_assumption field: verdict unchanged, no crash", none_given["scenarios"][0]["target_check"]["cost"].startswith("meets"))
unclear = add_target_checks({"scenarios": [{"estimated_cost": "$5 per 1M tokens"}], "volume_assumption": {"requests_per_month": 1, "basis": "assumed"}}, TARGETS)
check("cannot-be-compared cost gets no volume note", "assumes" not in unclear["scenarios"][0]["target_check"]["cost"])
new_style = {"estimated_cost": "$28K per year", "estimated_latency": "300-700ms", "accuracy_notes": "85-88% on MMLU-Pro; no task-specific benchmark found"}
check("new-format figures all parse", all(v.startswith(("meets", "misses")) for v in check_scenario(new_style, TARGETS).values()), check_scenario(new_style, TARGETS))


# ---- yearly cost calculated in code (figures from the v6 test run, where the agent's own arithmetic was wrong)
VOL = {"requests_per_month": 20000, "avg_input_tokens": 500, "avg_output_tokens": 1000, "basis": "assumed"}
gpt = {"model_stack": "gpt-4-turbo", "estimated_cost": "$36K per year", "input_price_per_1m_tokens": 30, "output_price_per_1m_tokens": 60}
claude = {"model_stack": "claude-2", "estimated_cost": "$27K per year", "input_price_per_1m_tokens": 12, "output_price_per_1m_tokens": 45}
llama = {"model_stack": "llama", "estimated_cost": "$18K per year", "input_price_per_1m_tokens": 50, "output_price_per_1m_tokens": 80}
fixed = apply_computed_costs({"scenarios": [gpt, claude, llama], "volume_assumption": VOL})
costs = [s["estimated_cost"] for s in fixed["scenarios"]]
check("GPT-4 Turbo recalculated to $18K, not the agent's $36K", costs[0] == "$18K per year", costs)
check("Claude 2 recalculated to $12.2K, not $27K", costs[1] == "$12.2K per year", costs)
check("Llama recalculated to $25.2K, not $18K (ranking flips)", costs[2] == "$25.2K per year", costs)
check("agent's own figure kept for the record", fixed["scenarios"][0]["agent_stated_cost"] == "$36K per year")
check("calculated cost is marked as such", fixed["scenarios"][0].get("cost_calculated_in_code") is True)
check("calculated cost is readable by the target check", parse_cost(costs[1]) is not None and abs(parse_cost(costs[1])[0] - 12200) < 1, parse_cost(costs[1]))
no_prices = apply_computed_costs({"scenarios": [{"estimated_cost": "$36K per year"}], "volume_assumption": VOL})
check("missing prices -> 'not calculated', never the unverified agent figure", no_prices["scenarios"][0]["estimated_cost"].startswith("not calculated"), no_prices["scenarios"][0]["estimated_cost"])
check("'not calculated' reports as cannot be compared", check_scenario(no_prices["scenarios"][0], {"budget": "$40K annual"})["cost"].startswith("cannot be compared"))
no_volume = apply_computed_costs({"scenarios": [gpt]})
check("missing volume -> 'not calculated'", no_volume["scenarios"][0]["estimated_cost"].startswith("not calculated"))
bad_type = apply_computed_costs({"scenarios": [dict(gpt, input_price_per_1m_tokens="thirty")], "volume_assumption": VOL})
check("non-numeric price -> 'not calculated'", bad_type["scenarios"][0]["estimated_cost"].startswith("not calculated"))
free = apply_computed_costs({"scenarios": [dict(gpt, input_price_per_1m_tokens=0, output_price_per_1m_tokens=0)], "volume_assumption": VOL})
check("a genuinely free option calculates to $0", free["scenarios"][0]["estimated_cost"] == "$0 per year", free["scenarios"][0]["estimated_cost"])
check("original output is not mutated", gpt["estimated_cost"] == "$36K per year")
check("malformed output passes through untouched", apply_computed_costs({"error": "x"}) == {"error": "x"})
check("cost target verdict uses the calculated figure: Llama $25.2K vs $20K budget misses", check_scenario(fixed["scenarios"][2], {"budget": "$20K annual"})["cost"].startswith("misses"))


print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
