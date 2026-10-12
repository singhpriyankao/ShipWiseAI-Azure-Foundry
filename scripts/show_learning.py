"""
Prints what the confirmed outcome reports have taught so far (learning.py): per model setting, the latency and cost
ratios against the forecast, whether a correction is being applied, the accuracy PMs reported, and how many shipped
features met the targets their PM set. No model calls and no sign-in.

Usage: python show_learning.py [--company COMPANY_ID] [--store-file PATH | --cosmos]
"""
import os
import sys

from learning import Calibration, MIN_FEATURES, decision_quality, latest_observations
from recommendation_store import CosmosRecommendationStore, LocalRecommendationStore


def main():
    args = sys.argv[1:]
    company = args[args.index("--company") + 1] if "--company" in args else "demo-company"
    if "--cosmos" in args:
        store = CosmosRecommendationStore()
    else:
        path = args[args.index("--store-file") + 1] if "--store-file" in args else os.path.join(os.path.dirname(os.path.abspath(__file__)), "accepted_recommendations.json")
        store = LocalRecommendationStore(path)
    records = store.list_all()
    cal = Calibration(records, company)
    print(f"company {company}: {len(records)} recommendations recorded, {len(cal.company)} with a confirmed report\n")

    dq = decision_quality(records, company)
    print("Met the targets the PM set themselves:")
    if dq["rate"] is None:
        print("   nothing comparable yet\n")
    else:
        print(f"   {dq['met_all']} of {dq['comparable']} features ({100 * dq['rate']:.0f}%). A rate over {dq['comparable']} feature{'s' if dq['comparable'] != 1 else ''} says little.")
        print("   by target: " + ", ".join(f"{k} {v['met']}/{v['of']}" for k, v in dq["by_target"].items() if v["of"]) + "\n")

    labels = sorted({r["chosen"]["label"] for r, _ in cal.company})
    for label in labels:
        categories = sorted({r["feature"].get("category") for r, _ in cal.company if r["chosen"]["label"] == label})
        info = cal.for_option(label, categories[0] if categories else None)
        print(label)
        for basis, stats in info["latency"].items():
            print(f"   latency ({basis.replace('_', ' ')}): {stats['n']} feature(s), median {stats['median']:.2f}x the forecast "
                  f"({stats['min']:.2f}x to {stats['max']:.2f}x) -> {stats['why']}")
        if info["cost"]["n"]:
            c = info["cost"]
            print(f"   model cost: {c['n']} feature(s), median {c['median']:.2f}x the forecast ({c['min']:.2f}x to {c['max']:.2f}x) -> {c['why']}")
        if info["unused"]["latency"] or info["unused"]["cost"]:
            print(f"   reports that couldn't be compared like for like: latency {info['unused']['latency']}, cost {info['unused']['cost']}")
        for category in categories:
            acc = cal.for_option(label, category)["accuracy"]
            if acc["n"]:
                print(f"   accuracy reported ({category}): {acc['min']:g}% to {acc['max']:g}% from {acc['n']} report(s), each measured on the reporter's own test set")
        print()
    print(f"(A correction is applied only when {MIN_FEATURES} separate features agree. Accuracy is shown, never applied.)")


if __name__ == "__main__":
    main()
