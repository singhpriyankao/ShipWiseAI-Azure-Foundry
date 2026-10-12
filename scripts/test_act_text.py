"""
Offline tests for act_text.py (the cited provisions' own wording). A fake search client stands in for Azure.

Run: python test_act_text.py
"""
import warnings
warnings.filterwarnings("ignore")

import act_text

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


class FakeSearch:
    def __init__(self, table):
        self.table, self.filters = table, []

    def search(self, search_text, filter, top):
        self.filters.append(filter)
        if isinstance(self.table, Exception):
            raise self.table
        return iter(self.table.get(filter, []))


ART50 = {"article_number": 50, "paragraph_number": 1, "article_title": "Transparency obligations", "text": "1. Providers shall ensure that AI systems ... are informed.", "source_url": "https://eur-lex/x"}

# ---- filters
check("article and paragraph", act_text.passage_filter({"article_number": 50, "paragraph_number": 1}) == "article_number eq 50 and paragraph_number eq 1")
check("article alone", act_text.passage_filter({"article_number": 6}) == "article_number eq 6")
check("annex item with sub-point", act_text.passage_filter({"annex_id": "III", "annex_item_number": 5, "annex_sub_point": "b"}) == "annex_id eq 'III' and annex_item_number eq 5 and annex_sub_point eq 'b'")
check("recital", act_text.passage_filter({"recital_number": 53}) == "recital_number eq 53")
check("an empty citation has no filter", act_text.passage_filter({}) is None)
check("a quote or operator smuggled into an annex id is refused, not passed to the search", act_text.passage_filter({"annex_id": "III' or 1 eq 1 or annex_id eq '"}) is None)
check("a non-number article is refused", act_text.passage_filter({"article_number": "50 or 1 eq 1"}) is None)
check("a boolean is not a number", act_text.passage_filter({"article_number": True}) is None)

# ---- quotes
client = FakeSearch({"article_number eq 50 and paragraph_number eq 1": [ART50]})
q = act_text.quote_citations([{"article_number": 50, "paragraph_number": 1}], client)
check("a citation returns its label, title, exact text and source link", q == [{"label": "Article 50(1)", "title": "Transparency obligations", "text": ART50["text"], "source_url": "https://eur-lex/x"}], q)
check("the same citation twice is quoted once", len(act_text.quote_citations([{"article_number": 50, "paragraph_number": 1}] * 2, client)) == 1)
check("a passage that isn't in the index is left out, not invented", act_text.quote_citations([{"article_number": 99, "paragraph_number": 9}], client) == [])
check("a failing search leaves the quote out and raises nothing", act_text.quote_citations([{"article_number": 50, "paragraph_number": 1}], FakeSearch(RuntimeError("down"))) == [])
check("no citations gives no quotes", act_text.quote_citations([], client) == [] and act_text.quote_citations(None, client) == [])

many = {"article_number eq 6": [{"article_number": 6, "paragraph_number": 2, "article_title": "T", "text": "Second.", "source_url": "u"}, {"article_number": 6, "paragraph_number": 1, "article_title": "T", "text": "First.", "source_url": "u"}]}
whole = act_text.quote_citations([{"article_number": 6}], FakeSearch(many))
check("a whole-article citation returns its paragraphs in order, numbered", whole[0]["text"] == "1. First.\n\n2. Second.", whole)

long_text = "word " * 1000
cut = act_text.quote_citations([{"article_number": 50, "paragraph_number": 1}], FakeSearch({"article_number eq 50 and paragraph_number eq 1": [{**ART50, "text": long_text}]}), max_chars=100)
check("a long passage is cut at a word boundary with an ellipsis", len(cut[0]["text"]) <= 102 and cut[0]["text"].endswith("…"), cut[0]["text"][-12:])
check("at most four distinct citations are quoted", len(act_text.quote_citations([{"article_number": n, "paragraph_number": 1} for n in range(1, 9)], FakeSearch({f"article_number eq {n} and paragraph_number eq 1": [{**ART50, "article_number": n}] for n in range(1, 9)}))) == 4)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
