"""
Offline tests for cards.py (which three options the PM Dashboard shows, and how a near miss is described).

Run: python test_cards.py
"""
import warnings
warnings.filterwarnings("ignore")

import cards

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


def opt(oid, role, quality, yearly, latency, verdicts=None, means="time until the first answer token arrives (the reply starts appearing)"):
    return {"id": oid, "role": role, "model_stack": f"Model {oid}", "quality_percentile": quality, "yearly_cost_usd": yearly,
            "latency_ms": latency, "latency_means": means, "verdicts": verdicts or ({"latency": "meets", "cost": "meets"} if role == "shortlist" else {}),
            "trade_offs": "t", "comparison": "c", "learned": []}


C = {"latency_target": "under 800ms", "budget": "$40K per year"}

# ---- the standard case: five on the shortlist, two near misses
scen = [
    opt("s1", "shortlist", 92.2, 1680, 669),
    opt("s2", "shortlist", 88.7, 1680, 719),
    opt("s3", "shortlist", 68.8, 816, 686),
    opt("s4", "shortlist", 64.3, 66, 773),
    opt("s5", "shortlist", 59.0, 660, 496),
    opt("n1", "near_miss", 83.7, 816, 1196, {"latency": "misses", "cost": "meets"}),
    opt("n2", "near_miss", 80.7, 816, 1247, {"latency": "misses", "cost": "meets"}),
]
out = cards.build_cards({"scenarios": scen}, C)
check("three cards, in the order A, B, C", [c["slot"] for c in out] == ["A", "B", "C"], [c["slot"] for c in out])
check("A is the highest general quality on the shortlist and is called Recommended", out[0]["option_id"] == "s1" and out[0]["badge"] == "Recommended" and out[0]["tone"] == "good", out[0])
check("B is the cheapest of the rest and is called Lowest cost", out[1]["option_id"] == "s4" and out[1]["badge"] == "Lowest cost", out[1])
check("C is the closest near miss, called Caution", out[2]["option_id"] == "n1" and out[2]["badge"] == "Caution" and out[2]["tone"] == "caution", out[2])
check("the near miss says what it misses, with the amounts, underneath", out[2]["misses"] == ["Misses latency by 396 ms (1,196 ms against 800 ms)"], out[2]["misses"])
check("a card that meets everything lists no misses", out[0]["misses"] == [])
check("cost is shown per month as well as per year (yearly divided by 12)", out[0]["monthly_cost_usd"] == 140.0 and out[0]["yearly_cost_usd"] == 1680, out[0])
check("latency says it is time to the first words when that is what was measured", out[0]["latency_kind"] == "to the first words")
check("quality is always marked as a proxy", all(c["quality_is_proxy"] is True for c in out))
check("each card carries the id the Approve button sends", all(c["option_id"] for c in out))
check("the same option is never on two cards", len({c["option_id"] for c in out}) == 3)

full = cards.build_cards({"scenarios": [opt("s1", "shortlist", 70, 100, 900, means="time to complete the whole reply")]}, C)
check("latency for a whole-reply feature is labelled as the full reply", full[0]["latency_kind"] == "for the full reply", full)

# ---- fewer options than three
only_short = cards.build_cards({"scenarios": [opt("s1", "shortlist", 70, 100, 500)]}, C)
check("one shortlist option and no near miss gives one card", [c["slot"] for c in only_short] == ["A"] and only_short[0]["badge"] == "Recommended")
short_near = cards.build_cards({"scenarios": [opt("s1", "shortlist", 70, 100, 500), opt("n1", "near_miss", 60, 100, 900, {"latency": "misses"})]}, C)
check("one shortlist option plus a near miss gives A then B (Caution), with no gap in the letters", [(c["slot"], c["badge"]) for c in short_near] == [("A", "Recommended"), ("B", "Caution")], short_near)
two_short = cards.build_cards({"scenarios": [opt("s1", "shortlist", 70, 500, 500), opt("s2", "shortlist", 60, 100, 300)]}, C)
check("two shortlist options and no near miss give A and B only", [c["badge"] for c in two_short] == ["Recommended", "Lowest cost"], two_short)
three_short = cards.build_cards({"scenarios": [opt("s1", "shortlist", 70, 500, 500), opt("s2", "shortlist", 60, 100, 400), opt("s3", "shortlist", 50, 300, 200)]}, C)
check("three shortlist options and no near miss: the third card is the fastest of what is left", [(c["badge"], c["option_id"]) for c in three_short] == [("Recommended", "s1"), ("Lowest cost", "s2"), ("Fastest", "s3")], three_short)

# ---- nothing meets the targets
none_meet = cards.build_cards({"scenarios": [
    opt("n2", "near_miss", 80, 800, 1247, {"latency": "misses", "cost": "meets"}),
    opt("n1", "near_miss", 83, 800, 1196, {"latency": "misses", "cost": "meets"}),
    opt("n3", "near_miss", 70, 90000, 700, {"latency": "meets", "cost": "misses"})]}, C)
check("with no option meeting every target, every card is Caution and none is Recommended", all(c["badge"] == "Caution" for c in none_meet) and len(none_meet) == 3, none_meet)
check("...ordered by how little they miss by (a 396 ms latency miss beats a 125% cost miss)", [c["option_id"] for c in none_meet] == ["n1", "n2", "n3"], [c["option_id"] for c in none_meet])
check("a cost miss is described in dollars a year", none_meet[2]["misses"] == ["Over budget by $50,000 a year ($90,000 against $40,000)"], none_meet[2]["misses"])

# ---- an option missing both
both = cards.misses(opt("x", "near_miss", 50, 90000, 1600, {"latency": "misses", "cost": "misses"}), C)
check("an option that misses both targets lists both", [m["target"] for m in both] == ["latency", "cost"], both)
check("accuracy is never listed as a miss, whatever the verdict says", cards.misses(opt("x", "near_miss", 50, 1, 1, {"accuracy": "misses"}), C) == [])

# ---- missing or odd data
check("no options gives no cards", cards.build_cards({"scenarios": []}, C) == [] and cards.build_cards({}, C) == [] and cards.build_cards(None, C) == [])
odd = cards.build_cards({"scenarios": [opt("s1", "shortlist", None, None, None), opt("s2", "shortlist", 50, 100, 300)]}, C)
check("an option with no quality, cost or latency figure doesn't crash, and ranks last", odd[0]["option_id"] == "s2" and odd[1]["option_id"] == "s1", odd)
check("a missing cost gives a missing monthly cost, not a made-up one", odd[1]["monthly_cost_usd"] is None)
check("a near miss whose target can't be parsed has no amounts, but is still shown", cards.build_cards({"scenarios": [opt("n1", "near_miss", 50, 1, 900, {"latency": "misses"})]}, {"latency_target": "fast"})[0]["misses"] == [])
check("ties on quality go to the cheaper option", cards.build_cards({"scenarios": [opt("a", "shortlist", 80, 900, 500), opt("b", "shortlist", 80, 100, 500)]}, C)[0]["option_id"] == "b")

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
