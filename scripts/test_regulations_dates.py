"""
Offline tests for derive_applicability_status, using the real ingested
Article 113 text. No model calls, no sign-in.

Run: python test_regulations_dates.py
"""
import json
import os

from regulations_guardrails import derive_applicability_status

CHUNKS = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "eu_ai_act_chunks.json")
data = json.load(open(CHUNKS, encoding="utf-8"))
records = data if isinstance(data, list) else data.get("chunks", data)
art_113 = [r for r in records if r["id"] == "art_113"]

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


def art(n, p=None):
    return {"type": "article", "article_number": n, "paragraph_number": p, "recital_number": None,
            "annex_id": None, "annex_item_number": None, "annex_sub_point": None, "regulation": "EU_AI_Act"}


def annex(i):
    return {"type": "annex", "article_number": None, "paragraph_number": None, "recital_number": None,
            "annex_id": i, "annex_item_number": None, "annex_sub_point": None, "regulation": "EU_AI_Act"}


def out(*cits):
    return {"citations": list(cits)}


# the live-run case: Art 50(3) must get the general date, not the 2025 one
s = derive_applicability_status(out(annex("III"), art(6, 2), art(50, 3)), art_113)
check("Art 50 + Annex III + Art 6(2) -> only the general date", s.count("2 August 2026") == 1 and "2025" not in s, s)
check("Art 50 is not given the 2 August 2025 date", "2 August 2025" not in s, s)

s = derive_applicability_status(out(art(5, 1)), art_113)
check("Art 5 -> 2 February 2025 via 113(a)", "2 February 2025" in s and "113(a)" in s, s)

s = derive_applicability_status(out(art(5, 1), art(50, 3)), art_113)
check("two chapters -> both dates listed", "2 February 2025" in s and "2 August 2026" in s, s)

s = derive_applicability_status(out(art(31)), art_113)
check("Art 31 (Ch III Section 4) -> 2 August 2025 via 113(b)", "2 August 2025" in s and "113(b)" in s, s)

s = derive_applicability_status(out(art(101)), art_113)
check("Art 101 is excepted from 113(b) -> general date", "2 August 2026" in s and "2025" not in s, s)

s = derive_applicability_status(out(art(6, 1)), art_113)
check("Art 6(1) -> 2 August 2027 via 113(c)", "2 August 2027" in s, s)

s = derive_applicability_status(out(art("50", "3")), art_113)
check("article numbers given as text still map", "2 August 2026" in s, s)

check("a recital-only citation list -> nothing to date", derive_applicability_status(out({"type": "recital", "recital_number": 18, "article_number": None, "regulation": "EU_AI_Act"}), art_113) == "")
check("no citations -> nothing to date", derive_applicability_status(out(), art_113) == "")
check("a GDPR citation is ignored", derive_applicability_status(out({**art(6), "regulation": "GDPR"}), art_113) == "")
check("unmapped article (Art 111) -> None", derive_applicability_status(out(art(111)), art_113) is None)
check("Annex other than III -> None", derive_applicability_status(out(annex("I")), art_113) is None)
check("Article 113 not retrieved -> None", derive_applicability_status(out(art(50, 3)), []) is None)

print(f"\n{'ALL PASSED' if not failures else str(len(failures)) + ' FAILED: ' + ', '.join(failures)}")
raise SystemExit(1 if failures else 0)
