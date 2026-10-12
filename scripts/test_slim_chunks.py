"""
Offline tests for slim_for_model in run_regulations_analysis.py (the lossless trimming of the
chunks sent to the Regulations agent). No network, no sign-in.

Run: python test_slim_chunks.py
"""
import copy
import json
import warnings
warnings.filterwarnings("ignore")

from run_regulations_analysis import slim_for_model

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


ART50 = "Article 50 full text: 1. Providers shall inform people... 2. ... 3. ... 4. ..."
ART6 = "Article 6 full text: 1. ... 2. ... 3. ..."


def chunk(cid, article, parent, text, **extra):
    base = {"@search.score": 1.2, "@search.reranker_score": None, "@search.highlights": None, "@search.captions": None,
            "@search.document_debug_info": None, "id": cid, "regulation": "AI Act", "celex_id": "32024R1689",
            "oj_reference": "OJ L", "article_number": article, "paragraph_number": 1, "recital_number": None,
            "is_recital": False, "is_annex": False, "annex_id": None, "text": text, "parent_article_text": parent,
            "source_url": "https://eur-lex.example/x"}
    base.update(extra)
    return base


chunks = [chunk("art_50_para_1", 50, ART50, "1. Providers shall inform people..."),
          chunk("art_6_para_2", 6, ART6, "2. ..."),
          chunk("art_50_para_2", 50, ART50, "2. ..."),
          chunk("art_50_para_4", 50, ART50, "4. ...")]
original = copy.deepcopy(chunks)
slim = slim_for_model(chunks)

check("the first chunk of an article keeps the full parent text", slim[0]["parent_article_text"] == ART50)
check("a later chunk of the same article points back to the first by id", slim[2]["parent_article_text"] == "[Same article text as chunk art_50_para_1, given above.]", slim[2]["parent_article_text"])
check("every later chunk of that article points to the same first chunk", slim[3]["parent_article_text"] == slim[2]["parent_article_text"])
check("a different article keeps its own full text", slim[1]["parent_article_text"] == ART6)
check("the order of chunks is unchanged", [c["id"] for c in slim] == ["art_50_para_1", "art_6_para_2", "art_50_para_2", "art_50_para_4"])
check("the chunk's own text is never shortened", [c["text"] for c in slim] == [c["text"] for c in chunks])
check("citation fields are all still there", all(k in slim[0] for k in ("regulation", "article_number", "paragraph_number", "is_recital", "is_annex", "id")), slim[0].keys())
check("search scores, highlights, captions and debug info are removed", not any(k.startswith("@search") for c in slim for k in c))
check("urls and publication references the model doesn't use are removed", not any(k in c for c in slim for k in ("source_url", "celex_id", "oj_reference")))
check("empty fields are removed", "recital_number" not in slim[0] and "annex_id" not in slim[0])
check("the originals are untouched, so the guardrails still see everything", chunks == original)
check("the slim copy is clearly smaller", len(json.dumps(slim)) < 0.7 * len(json.dumps(chunks)), (len(json.dumps(slim)), len(json.dumps(chunks))))
check("a chunk with no parent text is handled", "parent_article_text" not in slim_for_model([chunk("rct_1", None, None, "Recital text", is_recital=True)])[0])
check("no chunks gives no chunks", slim_for_model([]) == [])
check("a boolean False is kept (it is information, not an empty field)", slim[0]["is_recital"] is False)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
