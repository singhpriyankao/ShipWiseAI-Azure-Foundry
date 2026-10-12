"""
Offline tests for vendor_prices.py. The page text below is copied from the layout
the live Anthropic and OpenAI pricing pages had on 2026-10-01, so these tests
check the parsers against the real structure without needing the network.
A live check is a separate step: python -c "import vendor_prices; print(vendor_prices.fetch_prices())"

Run: python test_vendor_prices.py
"""
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import vendor_prices as vp

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


ANTHROPIC = """Model pricing

Model | Base tokens | Prompt caching |
Name | Input | Output | 5m writes | 1h writes | Hits and refreshes |

Claude Fable 5.1For demanding reasoning and long-horizon agentic work


 | $10 / MTok
 | $50 / MTok
 | $12.50 / MTok
 | $20 / MTok
 | $0.25 / MTok
 |

Claude Haiku 4.5The fastest model with near-frontier intelligence


 | $1 / MTok
 | $5 / MTok
 | $1.25 / MTok
 | $2 / MTok
 | $0.10 / MTok
 |
Additional models
 |
Claude Opus 4.1
 | $15 / MTok
 | $75 / MTok
 | $18.75 / MTok
 | $30 / MTok
 | $1.50 / MTok
 |
Claude Haiku 3.5
 | $0.80 / MTok
 | $4 / MTok
 | $1 / MTok
 | $1.60 / MTok
 | $0.08 / MTok
 |
"""

OPENAI = """Flagship models
Our latest models
Prices per 1M tokens.
StandardBatchFlexFast
Standard
| Short context | Long context |
Model | Input | Cached input | Cache writes | Output | Input | Cached input | Cache writes | Output |
gpt-6-astra | $10.00 | $1.00 | $12.50 | $50.00 | $20.00 | $2.00 | $25.00 | $75.00 |
gpt-6-luna | $0.10 | $0.01 | $0.125 | $0.50 | $0.20 | $0.02 | $0.25 | $0.75 |
gpt-6-nolong | $1.00 | $0.10 | $1.25 | $5.00 | - | - | - | - |
All models
Batch
Model | Input | Cached input | Cache writes | Output | Input | Cached input | Cache writes | Output |
gpt-6-astra | $5.00 | $0.50 | $6.25 | $25.00 | $10.00 | $1.00 | $12.50 | $37.50 |
All models
Standard
Model | Modality | Input | Output | Cost |
gpt-image-9 | Image | $8.00 | $30.00 | - |
"""

# ---- Anthropic
a = vp.parse_anthropic(ANTHROPIC)
by = {m["model"]: m for m in a}
check("Anthropic: all four models read, name separated from the glued description", set(by) == {"Claude Fable 5.1", "Claude Haiku 4.5", "Claude Opus 4.1", "Claude Haiku 3.5"}, set(by))
check("Anthropic: input and output prices are the first two columns", (by["Claude Fable 5.1"]["input_per_1m"], by["Claude Fable 5.1"]["output_per_1m"]) == (10.0, 50.0))
check("Anthropic: decimal prices read", (by["Claude Haiku 3.5"]["input_per_1m"], by["Claude Haiku 3.5"]["output_per_1m"]) == (0.8, 4.0))
check("Anthropic: main-table models are 'current'", by["Claude Haiku 4.5"]["listing"] == "current")
check("Anthropic: models under 'Additional models' are 'additional'", by["Claude Opus 4.1"]["listing"] == "additional" and by["Claude Haiku 3.5"]["listing"] == "additional")
check("Anthropic: caching prices are not mistaken for output", by["Claude Haiku 4.5"]["output_per_1m"] == 5.0)

# ---- OpenAI
o = vp.parse_openai(OPENAI)
names = [m["model"] for m in o]
check("OpenAI: only the Standard flagship table is read", names == ["gpt-6-astra", "gpt-6-luna", "gpt-6-nolong"], names)
check("OpenAI: short-context input and output columns used", (o[0]["input_per_1m"], o[0]["output_per_1m"]) == (10.0, 50.0), o[0])
check("OpenAI: a row with no long-context price still reads its short-context price", o[2]["output_per_1m"] == 5.0)
check("OpenAI: Batch prices are not mixed in", all(m["input_per_1m"] != 5.0 for m in o))
check("OpenAI: image table after 'All models' is ignored", "gpt-image-9" not in names)

# ---- guards
check("an implausible price is dropped", vp.parse_anthropic("Claude Test 1\n | $99999 / MTok\n | $5 / MTok\n") == [])
check("an unrelated page parses to nothing", vp.parse_anthropic("nothing here") == [] and vp.parse_openai("nothing here") == [])

# ---- fetch: a vendor whose page doesn't parse is reported, the other still works
original = vp._page_text
vp._page_text = lambda url: ANTHROPIC.replace("Claude Haiku 3.5", "x") if url == vp.ANTHROPIC_URL else "page changed completely"
try:
    result = vp.fetch_prices()
finally:
    vp._page_text = original
check("a vendor whose layout broke is reported in problems, not guessed", any(p.startswith("openai") for p in result["problems"]), result["problems"])
vp._page_text = lambda url: ANTHROPIC.split("Additional models")[0]
try:
    two_models = vp.fetch_prices()
finally:
    vp._page_text = original
check("fewer than the minimum models counts as a layout change", any(p.startswith("anthropic") and "layout may have changed" in p for p in two_models["problems"]), two_models["problems"])

def boom(url):
    raise OSError("network down")
vp._page_text = boom
try:
    down = vp.fetch_prices()
finally:
    vp._page_text = original
check("network failure -> no models, both vendors reported", down["models"] == [] and len(down["problems"]) == 2, down)

# ---- cache: stale fallback when a fresh fetch fails
real_cache = vp.CACHE_PATH
vp.CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_cache_test.json")
try:
    with open(vp.CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"models": [{"vendor": "anthropic", "model": "Claude Old 1", "input_per_1m": 1, "output_per_1m": 2}],
                   "problems": [], "as_of": "old", "fetched_epoch": time.time() - 3 * 24 * 3600}, f)
    vp._page_text = boom
    stale = vp.get_prices()
    check("old cache used when a fresh fetch fails, and marked stale", stale.get("stale") is True and stale["models"][0]["model"] == "Claude Old 1", stale)
    check("the stale result says so in its problems", any("older saved price list" in p for p in stale["problems"]))
    vp._page_text = original
    with open(vp.CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"models": [{"vendor": "x", "model": "Fresh"}], "problems": [], "as_of": "now", "fetched_epoch": time.time()}, f)
    vp._page_text = boom  # would fail if it tried to fetch
    check("a fresh cache is used without fetching", vp.get_prices()["models"][0]["model"] == "Fresh")
finally:
    vp._page_text = original
    if os.path.exists(vp.CACHE_PATH):
        os.remove(vp.CACHE_PATH)
    vp.CACHE_PATH = real_cache

# ---- generic path: model extraction checked against the page
MD_OPENAI = """### Standard pricing data

| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input |
| --- | --- | --- | --- | --- | --- |
| gpt-6-astra | $10.00 | $1.00 | $12.50 | $50.00 | $20.00 |
| gpt-6.1-sol | $2.00 | $0.10 | $2.50 | $10.00 | $4.00 |
| gpt-6-luna | $0.10 | $0.01 | $0.125 | $0.50 | $0.20 |

### Batch pricing data

| Model | Short context input | Short context output |
| --- | --- | --- |
| gpt-6-astra | $5.00 | $25.00 |
"""
MD_GOOGLE = """Gemini 3.8 Flash

### Standard

|   | Free Tier | Paid Tier, per 1M tokens in USD |
|---|---|---|
| Input price | Free of charge | $0.75 through December 31, 2026. $1.50 starting January 1, 2027. |
| Output price (including thinking tokens) | Free of charge | $3.75 through December 31, 2026. $7.50 starting January 1, 2027. |

"""
ROW_ASTRA = "| gpt-6-astra | $10.00 | $1.00 | $12.50 | $50.00 | $20.00 |"


def good(model="gpt-6-astra", a=10.0, b=50.0, ev=ROW_ASTRA):
    return {"model": model, "input_per_1m": a, "output_per_1m": b, "input_evidence": ev, "output_evidence": ev}


check("verify: a claim backed by the page is accepted", vp.verify_extracted(good(), MD_OPENAI) == (True, ""))
check("verify: an invented price is rejected", vp.verify_extracted(good(a=3.0), MD_OPENAI)[0] is False)
check("verify: a price from outside its evidence is rejected", vp.verify_extracted(good(b=25.0), MD_OPENAI)[0] is False)
check("verify: a price that is on the page but not with this model is rejected", vp.verify_extracted(good(a=1.25, b=50.0), MD_OPENAI)[0] is False)
check("verify: a model not on the page is rejected", vp.verify_extracted(good(model="gpt-9-fake"), MD_OPENAI)[1] == "model name not on the page")
check("verify: a text price is rejected", vp.verify_extracted(good(a="ten"), MD_OPENAI)[0] is False)
check("verify: a boolean price is rejected", vp.verify_extracted(good(a=True), MD_OPENAI)[0] is False)
check("verify: missing fields are rejected, not crashed on", vp.verify_extracted({"model": "gpt-6-astra"}, MD_OPENAI)[0] is False)
check("verify: non-dict item is rejected, not crashed on", vp.verify_extracted("gpt-6-astra", MD_OPENAI)[0] is False)
check("verify: prices on the lines under a model's heading are accepted (Google-style layout)",
      vp.verify_extracted({"model": "Gemini 3.8 Flash", "input_per_1m": 0.75, "output_per_1m": 3.75}, MD_GOOGLE)[0] is True)
# Known limit, written down as a test so nobody assumes it is covered: a number from the
# wrong column of the right row passes, because it really is on that row.
check("KNOWN LIMIT: a cached-input price read as the input price passes verification",
      vp.verify_extracted(good(a=1.00, b=50.0), MD_OPENAI)[0] is True)
check("a batch price for a model that also has a standard row is rejected (the first table wins)",
      vp.verify_extracted(good(a=5.0, b=25.0), MD_OPENAI)[0] is False)


class FakeClient:
    def __init__(self, models):
        self.calls = 0
        payload = json.dumps({"models": models})
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls += 1
                outer.last_kwargs = kwargs
                msg = type("M", (), {"content": payload})
                return type("R", (), {"choices": [type("C", (), {"message": msg})]})
        self.chat = type("Chat", (), {"completions": _Completions()})


three = [good("gpt-6-astra", 10.0, 50.0, ROW_ASTRA),
         good("gpt-6.1-sol", 2.0, 10.0, "| gpt-6.1-sol | $2.00 | $0.10 | $2.50 | $10.00 | $4.00 |"),
         good("gpt-6-luna", 0.1, 0.5, "| gpt-6-luna | $0.10 | $0.01 | $0.125 | $0.50 | $0.20 |")]
fake_one = FakeClient(three + [good("gpt-9-fake", 1.0, 2.0, "| gpt-9-fake | $1.00 | $2.00 |")])
found, notes = vp.extract_with_llm(fake_one, "openai", MD_OPENAI)
check("extraction: verified models kept, an invented one dropped", [m["model"] for m in found] == ["gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna"], [m["model"] for m in found])
check("extraction: the rejection is reported", len(notes) == 1 and "gpt-9-fake" in notes[0], notes)
check("extraction: kept models are marked as verified", all(m["extraction"] == "llm_verified" for m in found))
check("extraction: today's date is given to the model", __import__("datetime").date.today().isoformat() in fake_one.last_kwargs["messages"][0]["content"])
check("extraction: the model's JSON mode is on", fake_one.last_kwargs["response_format"] == {"type": "json_object"})
swapped = vp.extract_with_llm(FakeClient([dict(good(), input_per_1m=50.0, output_per_1m=10.0)]), "openai", MD_OPENAI)[0]
check("extraction: swapped input and output columns are flagged", len(swapped) == 1 and swapped[0]["flag"] is not None, swapped)
check("extraction: a reply with no 'models' list gives nothing, no crash", vp.extract_with_llm(FakeClient("not a list"), "openai", MD_OPENAI)[0] == [])


# ---- discovery through llms.txt
real_fetch = vp._fetch
vp._fetch = lambda url: "# Docs\n- [Messages](https://x.example/messages.md)\n- [Pricing](https://x.example/about/pricing.md)\n"
check("discovery: the pricing link is taken from llms.txt", vp.discover_pricing_url({"llms_txt": "u", "fallback": "FB"}) == "https://x.example/about/pricing.md")
vp._fetch = lambda url: "# Docs\n- [Messages](https://x.example/messages.md)\n"
check("discovery: no pricing link in llms.txt -> the known fallback", vp.discover_pricing_url({"llms_txt": "u", "fallback": "FB"}) == "FB")
def _down(url):
    raise OSError("down")
vp._fetch = _down
check("discovery: llms.txt unreachable -> the known fallback", vp.discover_pricing_url({"llms_txt": "u", "fallback": "FB"}) == "FB")
check("discovery: a vendor with no llms.txt -> the known fallback", vp.discover_pricing_url({"llms_txt": None, "fallback": "FB"}) == "FB")

# ---- whole generic fetch, with every page faked
vp._fetch = lambda url: MD_OPENAI
result = vp.fetch_prices(FakeClient(three))
check("generic fetch: every vendor that verifies >= 3 models is included", len(result["models"]) == 9 and all("source_url" in m for m in result["models"]), len(result["models"]))
result = vp.fetch_prices(FakeClient(three[:2]))
check("generic fetch: a vendor with too few verified models is reported, not guessed", len(result["models"]) == 0 and len(result["problems"]) == 3, result["problems"])
vp._fetch = real_fetch


check("verify: a text-to-speech or embedding model is filtered out by name even though its price is on the page",
      vp.verify_extracted(good("gpt-6-astra-tts"), MD_OPENAI.replace("gpt-6-astra", "gpt-6-astra-tts"))[0] is False)

# ---- retry: the same page gave 18 verified models on one run and 0 on the next
class FakeSequence:
    """Replies with each payload in turn (the last one repeats)."""
    def __init__(self, *payloads):
        self.calls = 0
        outer = self

        class _Completions:
            def create(self, **kwargs):
                body = json.dumps({"models": payloads[min(outer.calls, len(payloads) - 1)]})
                outer.calls += 1
                msg = type("M", (), {"content": body})
                return type("R", (), {"choices": [type("C", (), {"message": msg})]})
        self.chat = type("Chat", (), {"completions": _Completions()})


bad_copy = [dict(m, input_per_1m=0.001) for m in three]
vp._fetch = lambda url: MD_OPENAI
seq = FakeSequence(bad_copy, three)
result = vp.fetch_prices(seq)
check("retry: a failed first read is followed by another and the good read is used", len(result["models"]) >= 3 and seq.calls == 7, (len(result["models"]), seq.calls))
clean = FakeSequence(three)
vp.fetch_prices(clean)
check("retry: a second read confirms a clean first one, then stops (two calls per vendor)", clean.calls == 6, clean.calls)
always_bad = FakeSequence(bad_copy)
result = vp.fetch_prices(always_bad)
check("retry: stops when a read adds nothing, and nothing unverified is kept", always_bad.calls == 6 and result["models"] == [], (always_bad.calls, len(result["models"])))
vp._fetch = real_fetch


# ---- merging several reads of one page
extra = good("gpt-6-sol", 2.0, 10.0)
real_sources = vp.SOURCES
vp.SOURCES = tuple(x for x in vp.SOURCES if x["vendor"] == "openai")
vp._fetch = lambda url: MD_OPENAI + "\n| gpt-6-sol | $2.00 | $0.20 | $2.50 | $10.00 | $4.00 |\n"
merged = vp.fetch_prices(FakeSequence(three[:2], [three[0], dict(extra, input_evidence=None)]))
openai_models = sorted(m["model"] for m in merged["models"] if m["vendor"] == "openai")
check("merge: a model found on only one of the reads is kept", openai_models == ["gpt-6-astra", "gpt-6-sol", "gpt-6.1-sol"], openai_models)
conflict = vp.fetch_prices(FakeSequence(three, [dict(three[0], input_per_1m=1.0, output_per_1m=50.0)]))
flagged = [m for m in conflict["models"] if m["model"] == "gpt-6-astra" and m["vendor"] == "openai"]
check("merge: two reads that disagree on a price are flagged for a hand check, first read kept",
      len(flagged) == 1 and flagged[0]["flag"] is not None and flagged[0]["input_per_1m"] == 10.0, flagged)
vp._fetch, vp.SOURCES = real_fetch, real_sources


# ---- completeness checks: names, diff against the last list, vendor's models page
check("normalize: case, '.md' links, parentheticals and dated snapshots are removed",
      vp.normalize_model_name("GPT-5.5 (<272K context length)") == "gpt-5.5"
      and vp.normalize_model_name("gpt-4.1-mini.md") == "gpt-4.1-mini"
      and vp.normalize_model_name("gpt-4o-2024-05-13") == "gpt-4o"
      and vp.normalize_model_name("Claude Opus 4.5") == "claude opus 4.5")
check("normalize: a trailing sentence period is not part of the name", vp.normalize_model_name("gpt-5.") == "gpt-5")

ANTHROPIC_MODELS_PAGE = "Our models: Claude Fable 5.1, Claude Opus 5.5 and Claude Haiku 4.5. Older: Claude Opus 4.1."
check("models page: Anthropic names read by shape", vp.parse_listed_models("anthropic", ANTHROPIC_MODELS_PAGE) == {"claude fable 5.1", "claude opus 5.5", "claude haiku 4.5", "claude opus 4.1"})
OPENAI_MODELS_PAGE = "[gpt-6-astra](/docs/models/gpt-6-astra.md) [gpt-4o-mini-tts](x) [o4-mini](y)"
listed_openai = vp.parse_listed_models("openai", OPENAI_MODELS_PAGE)
check("models page: OpenAI names read, '.md' removed, text-to-speech filtered", "gpt-6-astra" in listed_openai and "o4-mini" in listed_openai and not any("tts" in n for n in listed_openai), listed_openai)
check("models page: 'gpt-4.1' is not confused with 'gpt-4.1-mini'", vp.parse_listed_models("openai", "gpt-4.1 and gpt-4.1-mini") == {"gpt-4.1", "gpt-4.1-mini"})

priced = [{"vendor": "anthropic", "model": "Claude Opus 5.5"}, {"vendor": "anthropic", "model": "Claude Haiku 4.5"}]
check("unpriced: a listed model with no price is reported", vp.unpriced_models({"anthropic": {"claude opus 5.5", "claude fable 5.1"}}, priced) == {"anthropic": ["claude fable 5.1"]})
check("unpriced: nothing reported when every listed model has a price", vp.unpriced_models({"anthropic": {"claude opus 5.5"}}, priced) == {})
check("unpriced: a priced model the page doesn't list is not a problem", vp.unpriced_models({"anthropic": set()}, priced) == {})

prev = {"listed_models": {"anthropic": ["claude haiku 4.5", "claude opus 5.5"], "openai": ["gpt-6-astra"]}}
listed_now = {"anthropic": {"claude opus 5.5", "claude sonnet 6"}, "openai": {"gpt-6-astra"}}
d = vp.diff_listed_models(prev, listed_now)
check("diff: a model newly listed on the vendor's page is reported", d["new_on_models_page"] == ["anthropic: claude sonnet 6"], d)
check("diff: a model no longer listed is reported", d["no_longer_listed"] == ["anthropic: claude haiku 4.5"], d)
check("diff: no previous refresh -> baseline, nothing flagged", vp.diff_listed_models(None, listed_now) == {"new_on_models_page": [], "no_longer_listed": [], "baseline": True})
check("diff: a models page that couldn't be read isn't reported as everything removed",
      vp.diff_listed_models(prev, {"anthropic": {"claude opus 5.5", "claude haiku 4.5"}})["no_longer_listed"] == [])
prev_seen = dict(prev, seen_listed=[["anthropic", "claude old 1"]])
check("diff: a model listed before, dropped, and listed again is not 'new'",
      vp.diff_listed_models(prev_seen, dict(listed_now, anthropic=listed_now["anthropic"] | {"claude old 1"}))["new_on_models_page"] == ["anthropic: claude sonnet 6"])
check("diff: identical lists report nothing", vp.diff_listed_models(prev, {"anthropic": {"claude haiku 4.5", "claude opus 5.5"}, "openai": {"gpt-6-astra"}})
      == {"new_on_models_page": [], "no_longer_listed": [], "baseline": False})

pages = {vp.MODELS_PAGES["anthropic"]: ANTHROPIC_MODELS_PAGE, vp.MODELS_PAGES["openai"]: OPENAI_MODELS_PAGE}
def fake_fetch(url):
    if url in pages:
        return pages[url]
    raise OSError("no such page")
vp._fetch = fake_fetch
try:
    result = vp.add_completeness_checks({"models": priced + [{"vendor": "openai", "model": "gpt-6-astra"}], "problems": []}, None)
finally:
    vp._fetch = real_fetch
check("combined: models page lists unpriced models, flags them", result["listed_but_unpriced"].get("anthropic") == ["claude fable 5.1", "claude opus 4.1"], result["listed_but_unpriced"])
check("combined: a models page that can't be read is a problem, not a crash", any(p.startswith("google") for p in result["problems"]), result["problems"])
check("combined: the listed-models history is saved for next time", ["anthropic", "claude opus 5.5"] in result["seen_listed"])

# ---- end to end through the cache: the second refresh sees what the vendor added since the first
real_cache, real_fetch_prices = vp.CACHE_PATH, vp.fetch_prices
vp.CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_cache_test.json")
try:
    vp.fetch_prices = lambda mc=None: {"models": [dict(m, source_url="u", as_of="t") for m in priced], "problems": [], "as_of": "t"}
    vp._fetch = fake_fetch
    first = vp.get_prices(force_refresh=True)
    check("cache: first refresh is a baseline", first["changes"]["baseline"] is True)
    pages[vp.MODELS_PAGES["anthropic"]] = ANTHROPIC_MODELS_PAGE + " New: Claude Brandnew 1."
    second = vp.get_prices(force_refresh=True)
    check("cache: the second refresh reports the model the vendor added since the first", second["changes"]["new_on_models_page"] == ["anthropic: claude brandnew 1"], second["changes"])
    third = vp.get_prices(force_refresh=True)
    check("cache: a model already reported isn't reported again", third["changes"]["new_on_models_page"] == [], third["changes"])
finally:
    vp.fetch_prices, vp._fetch = real_fetch_prices, real_fetch
    if os.path.exists(vp.CACHE_PATH):
        os.remove(vp.CACHE_PATH)
    vp.CACHE_PATH = real_cache


print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
