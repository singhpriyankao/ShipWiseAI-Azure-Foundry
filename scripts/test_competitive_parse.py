"""
Offline tests for competitive_parse.py (the v4 reply: plain text with the search tool's own citations) and for the
source check handling several cited pages. The first group uses three REAL v4 responses saved from the agent on
2026-10-06 (fixtures/competitive_v4_*.json); the rest are synthetic edge cases. No network.

Run: python test_competitive_parse.py
"""
import json
import os
import warnings
warnings.filterwarnings("ignore")

import competitive_checks as cc
import competitive_parse as cp

HERE = os.path.dirname(os.path.abspath(__file__))
failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {str(detail)[:300]}"))
    if not condition:
        failures.append(name)


def fixture(name):
    return json.load(open(os.path.join(HERE, "fixtures", f"competitive_v4_{name}.json"), encoding="utf-8"))


def by_product(out, product):
    return next(c for c in out["competitors"] if c["product"] == product)


# ---- real response 1: billing questions
billing = cp.parse_v4(fixture("billing"))
check("the billing reply parses into three competitors", billing is not None and [c["product"] for c in billing["competitors"]] == ["Maven AGI", "Ada", "Fin AI Agent"], billing and [c["product"] for c in billing["competitors"]])
check("header fields are read", billing["grounding_status"] == "grounded" and billing["missing_inputs"] == [])
fin = by_product(billing, "Fin AI Agent")
check("a cited line gets the address its citation carries (Intercom's own pricing page)", fin["pricing_source_url"] == "https://www.intercom.com/pricing" and fin["pricing_source_urls"] == ["https://www.intercom.com/pricing"], fin["pricing_source_urls"])
check("each line gets its own citation, not its neighbour's", fin["accuracy_source_url"] == "https://www.kommunicate.io/blog/intercom-pricing-breakdown/" and fin["latency_source_url"] == "https://www.intercom.com/pricing", (fin["accuracy_source_url"], fin["latency_source_url"]))
check("the figures survive and the citation markers are gone", fin["pricing_notes"].startswith("$0.99 per resolution") and "【" not in fin["pricing_notes"] and "$132/seat/month" in fin["pricing_notes"], fin["pricing_notes"])
check("vendor and comparison are read", fin["vendor"] == "Intercom" and fin["comparison_to_feature"].startswith("Fin AI Agent offers a resolution-based cost model"))
ada = by_product(billing, "Ada")
check("lines the agent wrote without any citation have no address, whatever they say", all(ada[f"{f}_source_url"] is None and ada[f"{f}_source_urls"] == [] for f in ("pricing", "accuracy", "latency")), ada)
check("their text is kept (the source check decides what the page shows)", ada["pricing_notes"].startswith("Ada's pricing is customized"))
maven = by_product(billing, "Maven AGI")
check("a line marked 'No source found.' keeps no address and gets the prefix the checks recognise", maven["pricing_notes"].startswith(cp.NO_SOURCE_PREFIX) and maven["pricing_source_url"] is None, maven["pricing_notes"])

# ---- real response 2: voice (the stray ' .' left where no citation was)
voice = cp.parse_v4(fixture("voice"))
netomi = by_product(voice, "Netomi AI")
check("an uncited claim with a figure ('85%') gets no address, so it can't be shown as sourced", "85%" in netomi["accuracy_notes"] and netomi["accuracy_source_url"] is None, netomi["accuracy_notes"])
check("the space left before the full stop is tidied", not netomi["accuracy_notes"].endswith(" .") and netomi["accuracy_notes"].endswith("."), repr(netomi["accuracy_notes"][-20:]))
kore = by_product(voice, "Kore.ai")
check("cited lines on other hosts get their own addresses", kore["pricing_source_url"] == "https://costbench.com/software/ai-chatbot-platforms/kore-ai/" and kore["latency_source_url"] == "https://www.cloudtalk.io/blog/kore-ai-pricing/", (kore["pricing_source_url"], kore["latency_source_url"]))

# ---- real response 3: agent assist
assist = cp.parse_v4(fixture("agent_assist"))
zen = by_product(assist, "Zendesk AI")
check("Zendesk's own page is attached to its pricing line", (zen["pricing_source_url"] or "").startswith("https://www.zendesk.com/"), zen["pricing_source_url"])
check("a 'No source found.' latency line is marked as such", zen["latency_notes"].startswith(cp.NO_SOURCE_PREFIX) and zen["latency_source_url"] is None)

# ---- the whole pipeline on a real reply: parse, then the source check (with fake pages)
def fake_fetch(url):
    pages = {"https://www.intercom.com/pricing": ("ok", "Intercom Fin AI Agent: $0.99 per resolution. Essential, Advanced and Expert seats from $29 a month up to $132 per seat per month. " * 30),
             "https://www.kommunicate.io/blog/intercom-pricing-breakdown/": ("ok", "A breakdown of Intercom pricing. " * 80)}
    return pages.get(url, ("page not found", ""))


import copy
checked = cc.check_competitor_sources(copy.deepcopy(billing), fetch=fake_fetch)
fin2 = by_product(checked, "Fin AI Agent")
check("the cited vendor page that contains the figures is verified", fin2["source_checks"]["pricing"] == "verified", fin2["source_checks"])
check("uncited claims are replaced as having no source, never shown", by_product(checked, "Ada")["source_checks"]["pricing"] == "no_source_note_replaced" and "not shown" in by_product(checked, "Ada")["pricing_notes"], by_product(checked, "Ada")["source_checks"])
check("a claim the agent itself marked 'No source found.' is left as stated", by_product(checked, "Maven AGI")["source_checks"]["pricing"] == "no_source_stated")

# ---- layout edge cases
def reply(text, annotations=None):
    return {"output": [{"type": "bing_grounding_call"}, {"type": "message", "content": [{"type": "output_text", "text": text, "annotations": annotations or []}]}]}


def cite(text, url, marker="【1:0†source】", kind="url_citation"):
    return {"type": kind, "url": url, "title": "t", "start_index": text.index(marker), "end_index": text.index(marker) + len(marker)}


t = "GROUNDING: grounded\nMISSING INPUTS: none\n\nCOMPETITOR 1\nProduct: P\nVendor: V\nPricing: $10 a month【1:0†source】【1:1†source】\nAccuracy: x\nLatency: No source found.\nComparison: c"
anns = [{"type": "url_citation", "url": "https://a.example/p", "start_index": t.index("【1:0"), "end_index": t.index("【1:0") + 13}, {"type": "url_citation", "url": "https://b.example/q", "start_index": t.index("【1:1"), "end_index": t.index("【1:1") + 13}]
two = cp.parse_v4(reply(t, anns))["competitors"][0]
check("a line citing two pages keeps both, in order, and the first is the main address", two["pricing_source_urls"] == ["https://a.example/p", "https://b.example/q"] and two["pricing_source_url"] == "https://a.example/p", two)
check("a citation in one line is never attached to another", two["accuracy_source_urls"] == [] and two["latency_source_urls"] == [])
check("other annotation types and non-web addresses are ignored", cp.parse_v4(reply(t, [{"type": "file_citation", "url": "https://a.example", "start_index": t.index("【1:0"), "end_index": 1}, {"type": "url_citation", "url": "javascript:alert(1)", "start_index": t.index("【1:0"), "end_index": 1}]))["competitors"][0]["pricing_source_urls"] == [])
check("Windows line endings don't break the layout", cp.parse_v4(reply(t.replace("\n", "\r\n"), []))["competitors"][0]["product"] == "P")
check("labels in any capitalisation are accepted", cp.parse_v4(reply("grounding: grounded\ncompetitor 1\nproduct: P\nvendor: V\npricing: no source found.\naccuracy: no source found.\nlatency: no source found.\ncomparison: c"))["competitors"][0]["vendor"] == "V")

check("an out-of-scope reply gives no competitors and says why", cp.parse_v4(reply("OUT OF SCOPE: this asks which model to use")) == {"competitors": [], "grounding_status": "ungrounded_fallback", "missing_inputs": ["out of scope: this asks which model to use"]})
miss = cp.parse_v4(reply("GROUNDING: grounded\nMISSING INPUTS: the kind of product; who the users are"))
check("missing inputs are listed, split on semicolons", miss["missing_inputs"] == ["the kind of product", "who the users are"] and miss["competitors"] == [], miss)
check("an ungrounded reply is reported as ungrounded", cp.parse_v4(reply("GROUNDING: ungrounded_fallback"))["grounding_status"] == "ungrounded_fallback")
check("a reply that follows none of the layout is unreadable (None), so the caller can ask again", cp.parse_v4(reply("Here are three great products you should look at!")) is None)
check("an empty reply, no message and no output are all unreadable", cp.parse_v4(reply("")) is None and cp.parse_v4({"output": []}) is None and cp.parse_v4({}) is None)
check("a competitor block with no product name is dropped, not invented", cp.parse_v4(reply("GROUNDING: grounded\nCOMPETITOR 1\nVendor: V\nPricing: p"))["competitors"] == [])
check("a missing grounding line is inferred from whether competitors were found", cp.parse_v4(reply("COMPETITOR 1\nProduct: P\nVendor: V"))["grounding_status"] == "grounded" and cp.parse_v4(reply("MISSING INPUTS: a"))["grounding_status"] == "ungrounded_fallback")

class FakeResponse:
    def __init__(self, d):
        self.d = d

    def model_dump(self):
        return self.d


check("the SDK response object is accepted as well as a dict", cp.parse_v4(FakeResponse(reply(t, anns)))["competitors"][0]["product"] == "P")

# ---- the source check with several cited pages
def check_urls(urls, pages, note="$10 a month"):
    out = {"competitors": [{"product": "P", "vendor": "V", "pricing_notes": note, "pricing_source_urls": urls, "pricing_source_url": urls[0] if urls else None,
                            "accuracy_notes": "No source found: x", "accuracy_source_url": None, "latency_notes": "No source found: x", "latency_source_url": None}]}
    cc.check_competitor_sources(out, fetch=lambda u: pages[u])
    return out["competitors"][0]


GOOD = ("ok", "V pricing: $10 a month. " * 200)
OTHER = ("ok", "A page about something else entirely. " * 200)
GONE = ("page not found", "")
BLOCKED = ("page returned 403", "")
r = check_urls(["https://a", "https://b"], {"https://a": OTHER, "https://b": GOOD})
check("a claim is verified if ANY cited page supports it, and the supporting address becomes its source", r["source_checks"]["pricing"] == "verified" and r["pricing_source_url"] == "https://b", r)
r = check_urls(["https://a", "https://b"], {"https://a": OTHER, "https://b": OTHER})
check("if every cited page loads and none supports it, the claim is removed", r["source_checks"]["pricing"] == "page_does_not_support_claim" and "not shown" in r["pricing_notes"], r)
r = check_urls(["https://a", "https://b"], {"https://a": GONE, "https://b": GONE})
check("if every cited page is missing, the claim is removed", r["source_checks"]["pricing"] == "page_does_not_support_claim" and "does not exist" in r["pricing_notes"], r)
r = check_urls(["https://a", "https://b"], {"https://a": GONE, "https://b": BLOCKED})
check("a missing page and a blocked page together prove nothing: could not be checked, claim kept", r["source_checks"]["pricing"].startswith("could_not_be_checked") and r["pricing_notes"] == "$10 a month", r)
r = check_urls(["https://a", "https://b"], {"https://a": GONE, "https://b": OTHER})
check("a missing page plus a page that loads and doesn't support it: removed", r["source_checks"]["pricing"] == "page_does_not_support_claim")
r = check_urls(["https://a"], {"https://a": GOOD})
check("one cited page works exactly as before", r["source_checks"]["pricing"] == "verified" and r["pricing_source_url"] == "https://a")
r = check_urls([], {})
check("no cited page: removed as having no source", r["source_checks"]["pricing"] == "no_source_note_replaced")

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
