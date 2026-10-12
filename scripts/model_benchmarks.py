"""
Latency and quality-ranking data for each model, from Artificial Analysis's free API
(https://artificialanalysis.ai/api-reference). Attribution to artificialanalysis.ai is
required wherever this data is shown. The key stays server-side (never in client code)
and the response is cached, as their terms ask; the free tier allows 1,000 requests a day.

What this is and isn't:
- Latency: Artificial Analysis publishes the median time to the first answer token and the
  median output speed (tokens per second) for a 1,000-token prompt. Code turns those into an
  estimated response time for the output size the Model Stack agent assumed. It is an
  estimate under their benchmark conditions, not a measurement of the PM's workload.
  Reasoning models spend time thinking before the first answer token, so the first-answer-token
  figure is used, not the first-token figure.
- Quality: an overall intelligence index. It ranks models in general. It does NOT say whether
  a model will reach a given accuracy on the PM's task, and must never be compared to an
  accuracy target as if it did.
- Prices: also published there. They are only used to cross-check the vendor-page prices.

Where one model name matches several configurations (mostly reasoning-effort settings, Low
to Max), each is returned on its own: they differ by a factor of 70 in latency, so averaging
or ranging them would hide the real choice.
"""
import json
import os
import re
import time
import urllib.request

from dotenv import load_dotenv

import runtime_paths

load_dotenv()

API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
CACHE_PATH = runtime_paths.writable_path("benchmarks_cache.json")
CACHE_TTL_SECONDS = int(os.environ.get("SHIPWISE_CACHE_TTL_SECONDS", 24 * 3600))
PRICE_MISMATCH_TOLERANCE = 0.10  # more than 10% apart means the match is probably a different variant
ATTRIBUTION = "Latency and quality data: Artificial Analysis (https://artificialanalysis.ai/)"


class BenchmarkSourceError(Exception):
    pass


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None


def _measurement(value):
    """A measured speed or latency. The API returns 0 for models it has no timing data for
    (seen on the real data: most entries have a first-answer time of exactly 0), and a
    latency of 0 is not a measurement. Treating it as one made unmeasured models look
    instant and put them at the top of a shortlist."""
    number = _number(value)
    return number if number else None


def parse_models(payload):
    """The API's `data` list as a plain list of dicts, with every missing field as None."""
    models = []
    for item in (payload or {}).get("data", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        evaluations = item.get("evaluations") if isinstance(item.get("evaluations"), dict) else {}
        pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
        creator = item.get("model_creator") if isinstance(item.get("model_creator"), dict) else {}
        models.append({
            "id": item.get("id"), "name": item["name"], "creator": creator.get("name"),
            "intelligence_index": _number(evaluations.get("artificial_analysis_intelligence_index")),
            "output_tokens_per_second": _measurement(item.get("median_output_tokens_per_second")),
            "first_token_seconds": _measurement(item.get("median_time_to_first_token_seconds")),
            "first_answer_token_seconds": _measurement(item.get("median_time_to_first_answer_token")),
            "input_price_per_1m": _number(pricing.get("price_1m_input_tokens")),
            "output_price_per_1m": _number(pricing.get("price_1m_output_tokens")),
            "release_date": item.get("release_date") if isinstance(item.get("release_date"), str) else None,
        })
    return models


def match_key(name):
    """A name reduced to its words and version numbers, lowercase, without a trailing
    "(reasoning)"-style qualifier and in sorted order, so "GPT-6.1 Sol (High)" meets
    "gpt-6.1-sol" and "Claude Haiku 4.5" meets Artificial Analysis's "Claude 4.5 Haiku"."""
    name = re.sub(r"\s*\([^)]*\)", "", str(name)).lower()
    return "|".join(sorted(re.findall(r"[a-z]+|\d+(?:\.\d+)?", name)))


def _close(a, b):
    return a is not None and b is not None and abs(a - b) <= PRICE_MISMATCH_TOLERANCE * max(abs(b), 1e-9)


def estimate_response_ms(variant, output_tokens):
    """(first_answer_token_ms, full_response_ms) or None if the data needed isn't there."""
    first = variant.get("first_answer_token_seconds")
    if first is None:
        first = variant.get("first_token_seconds")
    speed = variant.get("output_tokens_per_second")
    if first is None or not speed or speed <= 0 or not output_tokens or output_tokens <= 0:
        return None
    return first * 1000.0, (first + output_tokens / speed) * 1000.0


def benchmark_for(price_model, aa_models, output_tokens):
    """The configurations Artificial Analysis benchmarks for one model on the price list, or
    None if none match. What Artificial Analysis calls variants are mostly reasoning-effort
    settings (Low to Max) of the same model, and they differ enormously: one model's first
    answer arrives in 2 seconds at Low effort and 138 seconds at Max, with a lower quality
    score at Low. That choice is the real latency-versus-quality trade-off, so each
    configuration is returned on its own, never averaged or collapsed into a range.
    A configuration whose own prices are far from the vendor page's is dropped as probably
    a different product; if every one differs, they are all kept and the note says so."""
    key = match_key(price_model["model"])
    variants = [m for m in aa_models if match_key(m["name"]) == key]
    if not variants:
        return None
    notes = []
    priced = [v for v in variants if _close(v["input_price_per_1m"], price_model.get("input_per_1m"))
              and _close(v["output_price_per_1m"], price_model.get("output_per_1m"))]
    if priced and len(priced) < len(variants):
        notes.append(f"{len(variants) - len(priced)} configuration(s) with different prices left out")
        variants = priced
    elif not priced:
        notes.append("Artificial Analysis's prices differ from the vendor page's for every configuration; check the match")
    ranks = rank_percentiles(aa_models)
    configurations = []
    for v in variants:
        est = estimate_response_ms(v, output_tokens)
        first_ms = v["first_answer_token_seconds"] * 1000.0 if v["first_answer_token_seconds"] is not None else (
            v["first_token_seconds"] * 1000.0 if v["first_token_seconds"] is not None else None)
        configurations.append({
            "name": v["name"],
            "first_answer_token_ms": round(est[0]) if est else (round(first_ms) if first_ms is not None else None),
            "full_response_ms": round(est[1]) if est else None,
            "intelligence_index": v["intelligence_index"],
            "quality_percentile": ranks.get(v["name"]),
            "release_date": v.get("release_date"),
        })
    if any(c["full_response_ms"] is None for c in configurations):
        notes.append("some configurations have no output-speed data, so only their first-answer time is known")
    configurations.sort(key=lambda c: (c["full_response_ms"] is None, c["full_response_ms"] or c["first_answer_token_ms"] or 0))
    return {"configurations": configurations, "notes": notes}


def rank_percentiles(aa_models):
    """Each variant's position among all models that have an index, 0-100 (100 = highest),
    by exact name. Variants of one model can rank very differently, so this is not keyed
    by the matching name."""
    scored = sorted((m["intelligence_index"], m["name"]) for m in aa_models if m["intelligence_index"] is not None)
    n = len(scored)
    return {name: round(100.0 * i / (n - 1), 1) if n > 1 else 100.0 for i, (_, name) in enumerate(scored)}


def fetch_models(api_key=None):
    api_key = api_key or os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
    if not api_key:
        raise BenchmarkSourceError("no ARTIFICIAL_ANALYSIS_API_KEY set")
    req = urllib.request.Request(API_URL, headers={"x-api-key": api_key, "User-Agent": "Mozilla/5.0"})
    return parse_models(json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8")))


def get_models(force_refresh=False, api_key=None):
    """Cached for 24 hours (one request a day against a 1,000-a-day limit). If the fresh call
    fails, an older cache is used and marked stale; with neither, the error is raised."""
    cached = None
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            cached = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    if cached and not force_refresh and time.time() - cached.get("fetched_epoch", 0) < CACHE_TTL_SECONDS:
        return cached["models"], []
    try:
        models = fetch_models(api_key)
        if not models:
            raise BenchmarkSourceError("the response had no models")
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"models": models, "fetched_epoch": time.time()}, f)
        return models, []
    except Exception as e:
        if cached:
            return cached["models"], [f"Artificial Analysis could not be reached ({type(e).__name__}: {e}); using an older saved copy"]
        raise
