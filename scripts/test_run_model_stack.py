"""
Offline tests for run_model_stack_analysis.py as a code pipeline: the code-written overview,
building the options, and the whole call with fake price, benchmark and interpretation sources. There is
no agent call any more, so there is nothing to fake for one. No network, no sign-in.

Run: python test_run_model_stack.py
"""
import warnings
warnings.filterwarnings("ignore")

import feature_requirements as fr
import model_benchmarks
import model_stack_builder as sb
import run_model_stack_analysis as rm
import vendor_prices
from target_checks import add_target_checks

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + ("" if condition else f"   -> {detail}"))
    if not condition:
        failures.append(name)


def aa(name, index, tps, ttfat, p_in, p_out, release="2026-06-01"):
    return {"id": name, "name": name, "creator": "x", "intelligence_index": index, "output_tokens_per_second": tps,
            "first_token_seconds": ttfat, "first_answer_token_seconds": ttfat, "input_price_per_1m": p_in,
            "output_price_per_1m": p_out, "release_date": release}


AA = [aa("Quick Mini", 40.0, 200.0, 0.3, 0.1, 0.4), aa("Smart Pro (Low)", 70.0, 100.0, 0.7, 3.0, 15.0),
      aa("Smart Pro (High)", 80.0, 80.0, 20.0, 3.0, 15.0)]
PRICES = {"models": [{"vendor": "acme", "model": "Quick Mini", "input_per_1m": 0.1, "output_per_1m": 0.4, "source_url": "u"},
                     {"vendor": "acme", "model": "Smart Pro", "input_per_1m": 3.0, "output_per_1m": 15.0, "source_url": "u"}], "problems": []}
usage = dict(fr.DEFAULT_USAGE)

# ---- the overview is written by code
counts = {"considered": 94, "meet_all_given_targets": 2, "latency_unknown": 48}
ov = rm.code_overview(counts, {"latency": "under 800ms", "budget": "$40K annual"}, usage, "time until the reply starts")
check("overview: how many meet the targets that were given, naming only latency and budget", "2 of 94 configurations meet the latency and budget targets you gave" in ov and "accuracy" not in ov, ov)
check("overview: says what latency means", "Latency here means the time until the reply starts" in ov, ov)
check("overview: reports the configurations with no timing data", "48 could not be compared on latency" in ov, ov)
check("overview: states an assumed volume as assumed", "20,000 requests a month, because no volume was given" in ov, ov)
stated = dict(usage, volume_basis="stated", requests_per_month=15000, token_basis="estimated")
ov2 = rm.code_overview(counts, {"budget": "$1K a year"}, stated)
check("overview: a stated volume is attributed to the PM", "the 15,000 requests a month you gave" in ov2 and "because no volume" not in ov2, ov2)
check("overview: estimated token sizes are labelled estimated", "token sizes estimated from the kind of feature" in ov2, ov2)
check("overview: with no targets it says so", "no latency or budget target was given" in rm.code_overview({"considered": 5, "meet_all_given_targets": 5, "latency_unknown": 0}, {}, usage))

# ---- building the options
options, c, targets = rm.build_analysis(PRICES["models"], AA, usage, "under 800ms", "$40K annual")
check("build: options have a code-written description and comparison", all(o["trade_offs"] and o["comparison"].startswith("Compared with") for o in options), [(o["trade_offs"], o["comparison"]) for o in options])
check("build: shortlist and near-miss roles are both present", {o["role"] for o in options} == {"shortlist", "near_miss"}, {o["role"] for o in options})
check("build: a streamed chat compares the time until the reply starts", targets["latency_basis"] == "first_answer" and all("reply starts" in o["latency_means"] for o in options))
b_options, _, b_targets = rm.build_analysis(PRICES["models"], AA, usage, "under 800ms", None, "single_response")
check("build: a single-response feature compares the time until the reply is complete", b_targets["latency_basis"] == "full_response" and all("whole reply is complete" in o["latency_means"] for o in b_options), [o["latency_means"] for o in b_options])
check("build: ...and so an option that starts fast but finishes slowly is not on the shortlist for it", all(o["full_response_ms"] <= 800 for o in b_options if o["role"] == "shortlist"), [(o["model_stack"], o["full_response_ms"]) for o in b_options])
check("build: no description compares options", not any(w in o["trade_offs"] for o in options for w in (" than ", "faster", "cheaper", "higher", "better")))

# ---- the whole call, with fake sources
real_prices, real_aa, real_interp = vendor_prices.get_prices, model_benchmarks.get_models, fr.interpret_feature
INTERP = {"feature_category": "chat_assistant", "usage": dict(fr.DEFAULT_USAGE, requests_per_month=15000, volume_basis="stated", token_basis="estimated", avg_input_tokens=2000, basis="stated"),
          "reply_style": "streamed_chat", "caveats": ["This feature needs the model to read images."], "source": "model", "problems": []}
seen = {}


def fake_interp(model_client, description, deployment="gpt-4.1-nano"):
    seen["description"], seen["client"] = description, model_client
    return INTERP


try:
    vendor_prices.get_prices = lambda force_refresh=False, model_client=None: PRICES
    model_benchmarks.get_models = lambda force_refresh=False, api_key=None: (AA, [])
    fr.interpret_feature = fake_interp
    client = object()
    _, out = rm.call_model_stack_analysis("a support bot", None, "under 800ms", "$40K annual", model_client=client)
    check("call: the feature description and the shared model client go to the interpretation step", seen["description"] == "a support bot" and seen["client"] is client)
    check("call: the interpreted usage is what the costs are built on", out["volume_assumption"]["requests_per_month"] == 15000 and out["volume_assumption"]["avg_input_tokens"] == 2000)
    check("call: the interpretation's warnings are passed on", out["caveats"] == ["This feature needs the model to read images."], out["caveats"])
    check("call: it says who interpreted the feature, the reply style and the kind of feature", out["interpretation"] == {"source": "model", "reply_style": "streamed_chat", "feature_category": "chat_assistant"}, out["interpretation"])
    check("call: options, overview, accuracy note and attribution are all present",
          out["scenarios"] and "Compared with" in out["scenarios"][0]["comparison"] and "configurations meet" in out["overview"]
          and "No public benchmark measures accuracy" in out["accuracy_note"] and "Artificial Analysis" in out["attribution"], list(out))
    check("call: it says the notes were written by code, since there is no agent", out["notes_written_by"] == "code")
    check("call: grounded when prices and benchmarks both loaded", out["grounding_status"] == "grounded")
    verdicts = add_target_checks(out, {"accuracy_target": "90%", "latency_target": "under 800ms", "budget": "$40K annual"})["scenarios"][0]["target_check"]
    check("call: the downstream target check refuses to compare accuracy, and still gives latency and cost verdicts",
          verdicts["accuracy"].startswith("cannot be compared: no benchmark measures accuracy") and verdicts["latency"].startswith("meets") and verdicts["cost"].startswith("meets"), verdicts)

    import learning as L
    lbl = sb.option_label({"vendor": "acme", "model": "Quick Mini", "configuration": "Quick Mini"})
    cal_records = []
    for n in range(3):
        cal_records.append({"id": f"c{n}", "type": "accepted_recommendation", "company_id": "acme", "user_id": "u", "feature": {"category": "chat_assistant", "description": "d"},
                            "constraints_given": {}, "usage_assumption": {"requests_per_month": 20000, "volume_basis": "assumed"},
                            "chosen": {"label": lbl, "forecast": {"latency_ms": 300, "latency_means": "time until the first answer token arrives (the reply starts appearing)", "yearly_cost_usd": 48.0}},
                            "follow_ups": [], "outcomes": [{"id": "o", "reported_at": "2026-11-20T00:00:00+00:00", "model_differs": False, "confirmed_by_pm": True,
                                                            "fields": {"latency": {"value_ms": 450, "kind": "first_answer", "statistic": "median", "quote": "x"}}}]})
    fr.interpret_feature = fake_interp
    _, out = rm.call_model_stack_analysis("a support bot", None, "under 800ms", "$40K annual", model_client=client, calibration=L.Calibration(cal_records, "acme"))
    quick = next(o for o in out["scenarios"] if "Quick Mini" in o["model_stack"])
    check("call: the company's reports correct the forecast it is shown (300ms becomes 450ms)", quick["first_answer_ms"] == 450 and quick["raw_forecast"]["latency_ms"] == 300, (quick["first_answer_ms"], quick["raw_forecast"]))
    check("call: the output says how many of the company's features with reports were behind it", out["learning"]["company_features_with_reports"] == 3, out["learning"])
    check("call: with no calibration the output says none", rm.call_model_stack_analysis("a support bot", None, "under 800ms", "$40K annual", model_client=client)[1]["learning"]["company_features_with_reports"] == 0)

    INTERP_BATCH = dict(INTERP, reply_style="single_response", caveats=[])
    fr.interpret_feature = lambda mc, d, deployment="x": INTERP_BATCH
    _, out = rm.call_model_stack_analysis("a ticket tagger", None, "under 3 seconds", None, model_client=client)
    check("call: a single-response feature is compared on the time to complete the reply", "whole reply is complete" in out["latency_means"], out["latency_means"])

    fr.interpret_feature = fake_interp
    seen.clear()
    _, out = rm.call_model_stack_analysis("a support bot", None, "under 800ms", "$40K annual", usage=dict(fr.DEFAULT_USAGE), reply_style="single_response")
    check("call: usage supplied by the caller skips the interpretation step", out["interpretation"]["source"] == "caller" and "description" not in seen, (out["interpretation"], seen))
    seen.clear()
    _, out = rm.call_model_stack_analysis("a support bot", None, "under 800ms", "$40K annual")
    check("call: with no model client, defaults are used and the reason is reported", "description" not in seen and out["volume_assumption"]["volume_basis"] == "assumed" and any("no model client" in p for p in out["problems"]), out["problems"])

    def aa_down(**kw):
        raise RuntimeError("no key")
    model_benchmarks.get_models = aa_down
    _, out = rm.call_model_stack_analysis("a support bot", None, "under 800ms", "$40K annual", model_client=client)
    check("call: no benchmark data and a latency target -> no options, and it says why", out["scenarios"] == [] and "latency target could not be checked" in out["missing_inputs"][0], out["missing_inputs"])
    check("call: ...and the benchmark failure is listed in problems", any("could not be loaded" in p for p in out["problems"]), out["problems"])
    _, out = rm.call_model_stack_analysis("a support bot", None, None, "$40K annual", model_client=client)
    check("call: no benchmark data but only a budget -> cost-only options are still returned",
          len(out["scenarios"]) > 0 and all(o["estimated_latency"] == "not available" for o in out["scenarios"]), out["scenarios"][:1])

    model_benchmarks.get_models = lambda force_refresh=False, api_key=None: (AA, [])
    vendor_prices.get_prices = lambda force_refresh=False, model_client=None: {"models": [], "problems": ["anthropic: could not read prices"]}
    _, out = rm.call_model_stack_analysis("a support bot", None, "under 800ms", "$40K annual", model_client=client)
    check("call: no prices -> an honest empty result, never invented options",
          out["scenarios"] == [] and out["grounding_status"] == "ungrounded_fallback" and "no prices could be read" in out["missing_inputs"][0], out)
    check("call: ...and the price problem is listed", any("could not read prices" in p for p in out["problems"]))
finally:
    vendor_prices.get_prices, model_benchmarks.get_models, fr.interpret_feature = real_prices, real_aa, real_interp

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
