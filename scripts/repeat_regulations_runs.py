"""
Runs the full Regulations pipeline N times on the same question and prints a
compact per-run summary, to see which behaviors are consistent and which are
one-offs. A single passing run hid real problems before.

Usage: python repeat_regulations_runs.py [N] [results.json]
Makes real agent calls (N to 2N) and opens a browser login per call.
"""
import json, re, sys, warnings
warnings.filterwarnings("ignore")

from run_regulations_analysis import run_regulations_analysis

QUESTION = "our support bot reads a customer's message and guesses if they sound frustrated"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
results = []

for i in range(1, n + 1):
    try:
        user_input, out, status = run_regulations_analysis(QUESTION)
    except Exception as e:
        print(f"\nRUN {i}: EXCEPTION {type(e).__name__}: {e}")
        continue

    chunks = user_input["retrieved_chunks"]
    ids = [c.get("id") for c in chunks]
    da = out.get("derogation_assessment") or {}
    text_all = (out.get("answer") or "") + " " + (da.get("reason") or "")
    cites = []
    for c in out.get("citations", []):
        if c.get("type") == "annex":
            cites.append(f"Annex {c.get('annex_id')} {c.get('annex_item_number')}({c.get('annex_sub_point')})")
        elif c.get("type") == "recital":
            cites.append(f"Recital {c.get('recital_number')}")
        else:
            cites.append(f"Art {c.get('article_number')}({c.get('paragraph_number')})")

    print(f"\nRUN {i}: status={status} class={out.get('classification')} grounding={out.get('grounding_status')}")
    print(f"  chunks={len(chunks)} art_113={'art_113' in ids} art_6_para_3={'art_6_para_3' in ids}")
    print(f"  cites={cites}")
    print(f"  derogation_assessment={da}")
    print(f"  mentions 'profiling' in answer/reason: {bool(re.search('profil', text_all, re.I))}")
    print(f"  applicability_status={out.get('applicability_status')!r}")
    print(f"  answer={out.get('answer')!r}")
    results.append({"run": i, "status": status, "output": out, "chunk_ids": ids})

if len(sys.argv) > 2:
    json.dump(results, open(sys.argv[2], "w"), indent=2)
