"""
Extraction eval for the outcome report: runs every hand-labelled report in
evals/outcome_extraction_cases.json through the real extraction (the small model plus the code checks)
and scores it field by field against the target of 90% or better.

A field counts as correct if it is empty where the label says empty, or present and matching the label
within a small tolerance. Wrongly filled fields (a "false positive") are reported separately from missed
ones, because a figure invented where the PM gave none is the worse error: a missed field costs the PM a
question, a wrong one would be saved after a distracted "yes".

Usage: python run_outcome_extraction_eval.py [runs]
One browser sign-in; one small model call per case per run (a fraction of a cent).
"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from foundry_clients import Clients
from outcome_report import extract_outcome

CASES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evals", "outcome_extraction_cases.json")


def got(fields):
    """The extraction in the same flat shape as the labels."""
    f = fields
    return {
        "accuracy": f["accuracy"]["value_percent"] if "accuracy" in f else None,
        "latency": f["latency"]["value_ms"] if "latency" in f else None,
        "latency_statistic": f["latency"]["statistic"] if "latency" in f else None,
        "latency_kind": f["latency"]["kind"] if "latency" in f else None,
        "volume": f["volume"]["requests_per_month"] if "volume" in f else None,
        "cost": f["cost"]["yearly_usd"] if "cost" in f else None,
        "model": f["model_used"]["name"] if "model_used" in f else None,
        "overall": f["overall"]["verdict"] if "overall" in f else None,
    }


def close(a, b, kind):
    if a is None or b is None:
        return a is None and b is None
    if kind == "accuracy":
        return abs(a - b) <= 0.5
    if kind in ("latency", "volume", "cost"):
        return abs(a - b) <= 0.01 * max(abs(b), 1)
    if kind == "model":
        return str(b).lower() in str(a).lower() or str(a).lower() in str(b).lower()
    return a == b


def score(case, extraction):
    expected, actual = case["expected"], got(extraction["fields"])
    results = []
    for field in ("accuracy", "latency", "volume", "cost", "model", "overall"):
        want, have = expected.get(field), actual[field]
        ok = close(have, want, field)
        results.append((field, ok, want, have))
    for extra in ("latency_statistic", "latency_kind"):
        if extra in expected:
            results.append((extra, actual[extra] == expected[extra], expected[extra], actual[extra]))
    return results


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cases = json.load(open(CASES_PATH, encoding="utf-8"))["cases"]
    client = Clients().model_client
    total = right = false_pos = missed = 0
    problems = {}
    for run in range(1, runs + 1):
        for case in cases:
            extraction = extract_outcome(client, case["text"])
            for field, ok, want, have in score(case, extraction):
                total += 1
                right += ok
                if not ok:
                    if want is None:
                        false_pos += 1
                    elif have is None:
                        missed += 1
                    problems.setdefault(case["id"], []).append((field, want, have, [r for r in extraction["rejected"] if r[0] != "report"][:2]))
        print(f"run {run}/{runs} done")
    print(f"\n{right}/{total} fields correct ({100 * right / total:.1f}%)  [target: >= 90%]")
    print(f"wrongly filled (figure where none belongs, or wrong figure): {false_pos + (total - right - false_pos - missed)};  missed (left empty): {missed}")
    for case_id, items in problems.items():
        print(f"\n{case_id}:")
        seen = set()
        for field, want, have, rejected in items:
            if (field, want, have) in seen:
                continue
            seen.add((field, want, have))
            print(f"   {field}: expected {want!r}, got {have!r}" + (f"   (rejected by checks: {rejected})" if rejected else ""))


if __name__ == "__main__":
    main()
