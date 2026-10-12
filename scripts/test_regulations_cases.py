"""
Runs the full Regulations pipeline once per feature description, each chosen
to exercise a different outcome path, and prints result vs. expected.

Complements repeat_regulations_runs.py: that one runs a single question many
times (consistency); this one runs many different questions (coverage). All
earlier testing used one emotion-recognition prompt, so nothing had checked
the other classifications or the paths where Article 6(3) isn't involved.

"Expected" is the author's own reading of the Act, not ground truth - a mismatch
is a prompt to check the actual text, not an automatic failure of the agent.

Usage: python test_regulations_cases.py [results.json] [case_name ...]
With no case names, runs all of them.
Makes real agent calls (one to two per case) and opens a browser login per call.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")

from run_regulations_analysis import run_regulations_analysis

CASES = [
    ("faq_chatbot",
     "a chatbot on our help page that answers customers' common questions about billing and how to use the product",
     "transparency_obligated", "interacts directly with people -> Article 50(1) disclosure"),
    ("ticket_routing",
     "an internal model that reads incoming support tickets and tags them by topic so they go to the right team; customers never see it",
     "no_specific_obligation_identified", "internal, no direct interaction, no Annex III use"),
    ("spam_filter",
     "a filter that detects spam and abusive messages in our inbound support inbox and sets them aside",
     "no_specific_obligation_identified", "spam filtering is not an Annex III use"),
    ("credit_eligibility",
     "an AI that looks at a customer's payment history and decides whether to offer them a payment plan or extend credit on their invoice",
     "high_risk", "creditworthiness of natural persons -> Annex III item 5(b), a different Annex III item than 1(c)"),
    ("agent_emotion_monitoring",
     "software that listens to our support agents' voices during calls and infers whether they are stressed or frustrated, so managers can flag low performers",
     "prohibited", "emotion inference in the workplace -> Article 5(1)(f)"),
    ("too_vague",
     "add some AI to our product",
     "missing_inputs", "too vague to reason about -> should list missing_inputs, not guess"),
    ("out_of_scope",
     "which is cheaper for our support bot, GPT-4 or Claude?",
     "out_of_scope", "model-stack question -> not this agent's job"),
]

args = sys.argv[1:]
out_path = args.pop(0) if args and args[0].endswith(".json") else None
wanted = set(args)
results = []
for name, description, expected, why in CASES:
    if wanted and name not in wanted:
        continue
    print(f"\n=== {name} | expected: {expected} ({why})")
    try:
        user_input, out, status = run_regulations_analysis(description)
    except Exception as e:
        print(f"  EXCEPTION {type(e).__name__}: {e}")
        results.append({"case": name, "error": str(e)})
        continue

    chunk_ids = [c.get("id") for c in user_input["retrieved_chunks"]]
    cites = []
    for c in out.get("citations", []):
        if c.get("type") == "annex":
            cites.append(f"Annex {c.get('annex_id')} {c.get('annex_item_number')}({c.get('annex_sub_point')})")
        elif c.get("type") == "recital":
            cites.append(f"Recital {c.get('recital_number')}")
        else:
            cites.append(f"Art {c.get('article_number')}({c.get('paragraph_number')})")

    print(f"  status={status} classification={out.get('classification')!r} grounding={out.get('grounding_status')}")
    print(f"  chunks={len(chunk_ids)} art_6_para_3_retrieved={'art_6_para_3' in chunk_ids}")
    print(f"  missing_inputs={out.get('missing_inputs')}")
    print(f"  cites={cites}")
    print(f"  derogation_assessment={out.get('derogation_assessment')}")
    print(f"  applicability_status={out.get('applicability_status')!r}")
    print(f"  answer={out.get('answer')!r}")
    results.append({"case": name, "expected": expected, "status": status, "output": out, "chunk_ids": chunk_ids})

if out_path:
    json.dump(results, open(out_path, "w"), indent=2)
