"""
Offline tests for competitive_checks.py. A fake page fetcher stands in for the
network, so nothing here makes a web request (the safety checks on addresses
are tested directly). The first group reproduces the 2026-10-04 v3 run: three
competitors, latency URLs null, and notes that did not say "No source found:".

Run: python test_competitive_checks.py
"""
import copy

import competitive_checks as cc
from competitive_checks import (NO_SOURCE_NOTE, check_competitor_sources, fetch_page_text, figures_in, judge_page)

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


def comp(**over):
    base = {
        "product": "Zendesk AI for Support", "vendor": "Zendesk",
        "accuracy_notes": "Precise answers from verified knowledge bases.", "accuracy_source_url": "https://a.example/acc",
        "latency_notes": "Immediate responses.", "latency_source_url": None,
        "pricing_notes": "Subscription-based, bundled with support plans.", "pricing_source_url": "https://a.example/price",
        "comparison_to_feature": "Close to the feature.",
    }
    base.update(over)
    return {"competitors": [base], "grounding_status": "grounded", "missing_inputs": []}


PAGE = "Zendesk makes support software. " * 80  # long enough to count as readable


def fetcher(pages):
    return lambda url: pages.get(url, ("page returned 404", ""))


# 1. Null URL without the required prefix: replaced; original kept.
out = check_competitor_sources(comp(), fetch=fetcher({"https://a.example/acc": ("ok", PAGE), "https://a.example/price": ("ok", PAGE)}))
c = out["competitors"][0]
check("null URL without prefix: note replaced", c["latency_notes"] == NO_SOURCE_NOTE, c["latency_notes"])
check("null URL without prefix: original kept", c["original_latency_notes"] == "Immediate responses.")
check("null URL without prefix: status", c["source_checks"]["latency"] == "no_source_note_replaced")

# 2. Null URL with the prefix: untouched.
out = check_competitor_sources(comp(latency_notes="No source found: nothing published."), fetch=fetcher({}))
c = out["competitors"][0]
check("null URL with prefix: untouched", c["latency_notes"] == "No source found: nothing published." and c["source_checks"]["latency"] == "no_source_stated")

# 3. Page loads and mentions the vendor: verified, note unchanged.
out = check_competitor_sources(comp(), fetch=fetcher({"https://a.example/acc": ("ok", PAGE), "https://a.example/price": ("ok", PAGE)}))
c = out["competitors"][0]
check("page mentions vendor: verified", c["source_checks"]["accuracy"] == "verified" and c["accuracy_notes"].startswith("Precise"))

# 4. Page loads, never mentions the vendor: replaced.
out = check_competitor_sources(comp(), fetch=fetcher({"https://a.example/acc": ("ok", "Unrelated text about gardening. " * 80), "https://a.example/price": ("ok", PAGE)}))
c = out["competitors"][0]
check("page without vendor: replaced", c["source_checks"]["accuracy"] == "page_does_not_support_claim" and c["accuracy_notes"].startswith("No verified source"), c["accuracy_notes"])
check("page without vendor: original kept", c["original_accuracy_notes"].startswith("Precise"))
check("a replacement in one field leaves the others alone", c["source_checks"]["pricing"] == "verified")

# 5. A figure in the note that is not on the page: replaced. The Intercom case ($74, 90%).
out = check_competitor_sources(comp(pricing_notes="Starts at $74/month.", accuracy_notes="Resolves up to 90% of queries."),
                               fetch=fetcher({"https://a.example/acc": ("ok", PAGE + " Resolves 90% of queries."), "https://a.example/price": ("ok", PAGE)}))
c = out["competitors"][0]
check("price not on page: replaced", c["source_checks"]["pricing"] == "page_does_not_support_claim" and "$74" in c["pricing_notes"], c["pricing_notes"])
check("percentage on page: verified", c["source_checks"]["accuracy"] == "verified", c["source_checks"]["accuracy"])

# 6. Figure written differently on the page ($1,200 vs $1200, "90 %" vs "90%") still matches.
check("figures normalised", judge_page({"vendor": "Zendesk"}, "From $1200 a month, 90% accurate.", "Zendesk plans from $1,200 a month, 90 % accurate. " + "x " * 800) is None)
check("figures_in finds prices and percentages", figures_in("$74/month, up to 90% and $1,000.50") == ["$74", "90%", "$1000.50"], figures_in("$74/month, up to 90% and $1,000.50"))

# 7. Page can't be read: note kept, marked could not be checked.
out = check_competitor_sources(comp(), fetch=fetcher({"https://a.example/acc": ("page returned 403", ""), "https://a.example/price": ("page has too little readable text", "")}))
c = out["competitors"][0]
check("blocked page: note kept", c["accuracy_notes"].startswith("Precise") and "original_accuracy_notes" not in c)
check("blocked page: status", c["source_checks"]["accuracy"].startswith("could_not_be_checked"), c["source_checks"]["accuracy"])

# 7b. A page that doesn't exist (404) is different from a blocked one: the claim is replaced.
out = check_competitor_sources(comp(), fetch=fetcher({"https://a.example/acc": (cc.PAGE_NOT_FOUND, ""), "https://a.example/price": ("ok", PAGE)}))
c = out["competitors"][0]
check("missing page: replaced", c["source_checks"]["accuracy"] == "page_does_not_support_claim" and "does not exist" in c["accuracy_notes"], c["accuracy_notes"])
check("missing page: original kept", c["original_accuracy_notes"].startswith("Precise"))

# 8. The same URL cited twice is fetched once.
calls = []
def counting(url):
    calls.append(url)
    return "ok", PAGE
check_competitor_sources(comp(latency_source_url="https://a.example/acc", pricing_source_url="https://a.example/acc"), fetch=counting)
check("one fetch per distinct URL", calls.count("https://a.example/acc") == 1 and len(calls) == 1, calls)

# 9. Odd agent outputs don't crash.
check("no competitors: unchanged", check_competitor_sources({"competitors": [], "grounding_status": "ungrounded_fallback"}, fetch=fetcher({})) == {"competitors": [], "grounding_status": "ungrounded_fallback"})
check("not a dict: unchanged", check_competitor_sources(None, fetch=fetcher({})) is None)

# 10. Safety: the fetcher refuses addresses a model could use to reach inside our network.
for url, label in (("http://localhost/x", "localhost"), ("http://127.0.0.1/x", "loopback"), ("http://169.254.169.254/metadata", "cloud metadata address"),
                   ("http://10.0.0.5/x", "private range"), ("file:///etc/passwd", "file scheme"), ("ftp://example.com/x", "ftp scheme")):
    status, _ = fetch_page_text(url)
    check(f"fetch refuses {label}", status in ("address not allowed", "not a web address"), status)

print()
print("ALL PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
