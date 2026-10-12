"""
Offline tests for model_benchmarks.py, using the response shape documented at
https://artificialanalysis.ai/api-reference (the real API needs a key). No network.

Run: python test_model_benchmarks.py
"""
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import model_benchmarks as mb

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


def entry(name, creator="OpenAI", index=60.0, tps=100.0, ttft=0.4, ttfat=None, p_in=2.0, p_out=10.0):
    item = {"id": name, "name": name, "model_creator": {"name": creator},
            "evaluations": {"artificial_analysis_intelligence_index": index},
            "pricing": {"price_1m_input_tokens": p_in, "price_1m_output_tokens": p_out},
            "median_output_tokens_per_second": tps, "median_time_to_first_token_seconds": ttft}
    if ttfat is not None:
        item["median_time_to_first_answer_token"] = ttfat
    return item


PAYLOAD = {"status": 200, "data": [
    entry("GPT-6.1 Sol", index=62.0, tps=120.0, ttft=0.5, p_in=2.0, p_out=10.0),
    entry("Claude Sonnet 5.5 (Reasoning)", "Anthropic", index=70.0, tps=80.0, ttft=14.0, ttfat=14.0, p_in=2.0, p_out=10.0),
    entry("Claude Sonnet 5.5 (Non-reasoning)", "Anthropic", index=55.0, tps=90.0, ttft=0.6, ttfat=0.6, p_in=2.0, p_out=10.0),
    entry("Claude Sonnet 5.5 (Extended)", "Anthropic", index=72.0, tps=70.0, ttft=20.0, ttfat=20.0, p_in=9.0, p_out=40.0),
    {"id": "x", "name": "Half Data Model"},
]}
models = mb.parse_models(PAYLOAD)
by = {m["name"]: m for m in models}

# ---- parsing
check("parse: every entry with a name is read", len(models) == 5, len(models))
check("parse: fields are read from the documented shape", by["GPT-6.1 Sol"]["intelligence_index"] == 62.0 and by["GPT-6.1 Sol"]["output_tokens_per_second"] == 120.0)
check("parse: missing fields become None, not a crash", by["Half Data Model"]["intelligence_index"] is None and by["Half Data Model"]["first_token_seconds"] is None)
check("parse: a negative or text value is treated as missing", mb.parse_models({"data": [{"name": "a", "median_output_tokens_per_second": -5, "median_time_to_first_token_seconds": "fast"}]})[0]["first_token_seconds"] is None)
check("parse: garbage input gives an empty list, no crash", mb.parse_models(None) == [] and mb.parse_models({"data": "x"}) == [] and mb.parse_models({"data": [None, 3]}) == [])

# ---- matching names
check("match key: qualifiers, case and punctuation are ignored", mb.match_key("GPT-6.1 Sol (high)") == mb.match_key("gpt-6.1-sol"))
check("match key: word order doesn't matter (the vendor and Artificial Analysis order names differently)", mb.match_key("Claude Haiku 4.5") == mb.match_key("Claude 4.5 Haiku (Reasoning)"))
check("match key: different versions don't collide", mb.match_key("gpt-5") != mb.match_key("gpt-5.5") and mb.match_key("Claude Opus 4") != mb.match_key("Claude Opus 4.5"))
check("match key: different models don't collide", mb.match_key("Claude Opus 5") != mb.match_key("Claude Sonnet 5"))

# ---- latency estimate
est = mb.estimate_response_ms(by["GPT-6.1 Sol"], output_tokens=240)
check("estimate: first token 500ms; 240 tokens at 120/s adds 2000ms", est == (500.0, 2500.0), est)
check("estimate: uses the first ANSWER token time when given (reasoning models think first)", mb.estimate_response_ms(by["Claude Sonnet 5.5 (Reasoning)"], 80)[0] == 14000.0)
check("estimate: no speed data -> None, not a guess", mb.estimate_response_ms(by["Half Data Model"], 100) is None)
check("estimate: no output size -> None", mb.estimate_response_ms(by["GPT-6.1 Sol"], None) is None and mb.estimate_response_ms(by["GPT-6.1 Sol"], 0) is None)

# ---- one model's benchmark: each configuration on its own
sol = mb.benchmark_for({"model": "gpt-6.1-sol", "input_per_1m": 2.0, "output_per_1m": 10.0}, models, 240)
check("benchmark: a single match is one configuration with its own figures",
      len(sol["configurations"]) == 1 and sol["configurations"][0]["first_answer_token_ms"] == 500 and sol["configurations"][0]["full_response_ms"] == 2500
      and sol["configurations"][0]["intelligence_index"] == 62.0, sol)
sonnet = mb.benchmark_for({"model": "Claude Sonnet 5.5", "input_per_1m": 2.0, "output_per_1m": 10.0}, models, 100)
names = [c["name"] for c in sonnet["configurations"]]
check("benchmark: configurations with the vendor's price are kept separately, not averaged", names == ["Claude Sonnet 5.5 (Non-reasoning)", "Claude Sonnet 5.5 (Reasoning)"], names)
check("benchmark: they are listed fastest first, with their own latency and quality",
      sonnet["configurations"][0]["first_answer_token_ms"] == 600 and sonnet["configurations"][1]["first_answer_token_ms"] == 14000
      and sonnet["configurations"][0]["quality_percentile"] < sonnet["configurations"][1]["quality_percentile"], sonnet["configurations"])
check("benchmark: a configuration with a very different price is left out and the note says so", "Claude Sonnet 5.5 (Extended)" not in names and any("left out" in n for n in sonnet["notes"]), sonnet)
check("benchmark: a model with no match gives None", mb.benchmark_for({"model": "no-such-model", "input_per_1m": 1, "output_per_1m": 1}, models, 100) is None)
wrong = mb.benchmark_for({"model": "gpt-6.1-sol", "input_per_1m": 50.0, "output_per_1m": 200.0}, models, 240)
check("benchmark: prices that disagree for every configuration are flagged, not hidden", any("differ" in n for n in wrong["notes"]), wrong)
nodata = mb.benchmark_for({"model": "Half Data Model", "input_per_1m": 1, "output_per_1m": 1}, models, 100)
check("benchmark: a match with no speed data has no response estimate, not an invented one", nodata["configurations"][0]["full_response_ms"] is None, nodata)
check("benchmark: ...and the note says only first-answer time is known", any("no output-speed data" in n for n in nodata["notes"]), nodata)

# ---- rank
ranks = mb.rank_percentiles(models)
check("rank: the highest index is 100 and the lowest is 0, by exact configuration name", ranks["Claude Sonnet 5.5 (Extended)"] == 100.0 and ranks["Claude Sonnet 5.5 (Non-reasoning)"] == 0.0, ranks)


# ---- key and cache
os.environ.pop("ARTIFICIAL_ANALYSIS_API_KEY", None)
try:
    mb.fetch_models()
    raised = False
except mb.BenchmarkSourceError:
    raised = True
check("fetch: no API key -> a clear error, no request made", raised)

real_cache, real_fetch = mb.CACHE_PATH, mb.fetch_models
mb.CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks_cache_test.json")
try:
    calls = []
    mb.fetch_models = lambda key=None: calls.append(1) or models
    first, problems = mb.get_models(force_refresh=True)
    second, _ = mb.get_models()
    check("cache: a second call within a day makes no request", len(calls) == 1 and second == first, len(calls))

    with open(mb.CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"models": models, "fetched_epoch": time.time() - 3 * 24 * 3600}, f)
    def down(key=None):
        raise OSError("network down")
    mb.fetch_models = down
    stale, problems = mb.get_models()
    check("cache: an old copy is used when the API is down, and the problem is reported", stale == models and any("older saved copy" in p for p in problems), problems)
    os.remove(mb.CACHE_PATH)
    try:
        mb.get_models()
        raised = False
    except OSError:
        raised = True
    check("cache: no cache and the API down -> the error is raised, not swallowed", raised)
finally:
    mb.fetch_models = real_fetch
    if os.path.exists(mb.CACHE_PATH):
        os.remove(mb.CACHE_PATH)
    mb.CACHE_PATH = real_cache

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
