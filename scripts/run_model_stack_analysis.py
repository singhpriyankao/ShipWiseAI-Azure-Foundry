"""
Model Stack Analysis as a pipeline of code, with a model only where reading language is needed
. It is no longer an agent: there is no registered Foundry agent behind it.

Why it stopped being one. The agent used to search the web for prices and benchmarks and write the
options itself. Across versions its prices changed tenfold between runs, its accuracy figures looked
invented, its arithmetic was wrong, and its text made false claims. Each of those jobs moved into code,
and by the end nothing was left for an agent to decide: every step was fixed. A fixed series of steps
is a workflow, and a workflow is more reliable and cheaper than an agent when nothing needs deciding.

The steps:
  1. interpret   feature_requirements.py: a small model reads the feature description and estimates the
                 token sizes, whether the reply streams, and any requirements we can't check; code verifies
                 anything it says the PM stated
  2. prices      vendor_prices.py: the vendors' own pricing pages (a small model reads them, code verifies)
  3. benchmarks  model_benchmarks.py: latency and a general quality ranking from Artificial Analysis
  4. options     model_stack_builder.py: one option per model and reasoning-effort setting, yearly cost
                 calculated, latency and budget checked against the PM's targets, cut to a short list,
                 with code-written descriptions and comparisons
Accuracy is never compared to the PM's target: no public benchmark measures accuracy on their task.
Options carry a general quality percentile, labelled a proxy.

Called by the Orchestrator with the shared model client; run directly it signs in itself.
"""
import json
import os
import uuid
from datetime import datetime, timezone

from azure.cosmos import CosmosClient
from dotenv import load_dotenv

import feature_requirements
import model_benchmarks
import model_stack_builder as builder
import vendor_prices

load_dotenv()

COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT")
COSMOS_KEY = os.environ.get("COSMOS_KEY")


def code_overview(counts, targets_given, usage, latency_means=None):
    """The summary line, written by code. An agent-written overview once said two options "met all the
    targets for latency, cost, and accuracy" when accuracy is never checked."""
    given = [name for name in ("latency", "budget") if targets_given.get(name)]
    if given:
        sentence = f"{counts['meet_all_given_targets']} of {counts['considered']} configurations meet the {' and '.join(given)} target{'s' if len(given) > 1 else ''} you gave."
    else:
        sentence = f"{counts['considered']} configurations were considered; no latency or budget target was given to check them against."
    if latency_means and targets_given.get("latency"):
        sentence += f" Latency here means the {latency_means}."
    if counts.get("latency_unknown"):
        sentence += f" {counts['latency_unknown']} could not be compared on latency because no timing data exists for them."
    if usage.get("volume_basis") == "assumed":
        sentence += (f" Costs assume {usage['requests_per_month']:,} requests a month, because no volume was given,"
                     f" with {usage['avg_input_tokens']:,} input and {usage['avg_output_tokens']:,} output tokens each")
    else:
        sentence += (f" Costs use the {usage['requests_per_month']:,} requests a month you gave,"
                     f" with {usage['avg_input_tokens']:,} input and {usage['avg_output_tokens']:,} output tokens each")
    sentence += " (token sizes estimated from the kind of feature)." if usage.get("token_basis") == "estimated" else "."
    return sentence


def build_analysis(price_models, aa_models, usage, latency_target, budget, reply_style="streamed_chat", calibration=None, feature_category="other"):
    """Everything that is code. Returns (options, counts, targets)."""
    candidates = builder.build_candidates(price_models, aa_models, usage)
    targets = builder.parse_targets(latency_target, budget, reply_style)
    # What this company's own reports have taught is applied before the targets are checked, so a corrected figure can move
    # an option on or off the shortlist, and every correction is shown beside the original forecast.
    builder.apply_calibration(candidates, calibration, feature_category, targets["latency_basis"])
    result = builder.shortlist(candidates, targets)
    changes = builder.calibration_changes(candidates, targets)
    options = ([builder.to_scenario(c, "shortlist", usage, targets) for c in result["shortlist"]]
               + [builder.to_scenario(c, "near_miss", usage, targets) for c in result["near_misses"]])
    comparisons = builder.compare_options(options, targets)
    for option in options:
        option["trade_offs"] = builder.describe_option(option)
        option["comparison"] = comparisons[option["id"]]
    result["counts"]["calibration_changes"] = changes
    return options, result["counts"], targets


def call_model_stack_analysis(feature_description, accuracy_target=None, latency_target=None, budget=None,
                              model_client=None, usage=None, reply_style=None, source_text=None, calibration=None):
    # model_client: the shared plain model client from the Orchestrator (one sign-in for all parallel
    # calls), used by the interpretation step and the price reader. Without one, the defaults are used for
    # usage and the hand-written price parsers for prices, so this still runs on its own.
    user_input = {"feature_description": feature_description, "accuracy_target": accuracy_target,
                  "latency_target": latency_target, "budget": budget}
    problems, caveats = [], []
    feature_category = "other"

    if usage is None:
        if model_client is not None:
            # source_text is the PM's own message when there is one: the feature description the
            # Orchestrator passes has been summarised and may have lost a stated volume.
            interpretation = feature_requirements.interpret_feature(model_client, source_text or feature_description)
        else:
            interpretation = feature_requirements.validate_interpretation(None, feature_description)
            interpretation["problems"] = ["no model client was given, so the default usage was used"]
        usage, reply_style, caveats = interpretation["usage"], interpretation["reply_style"], interpretation["caveats"]
        feature_category = interpretation.get("feature_category", "other")
        problems += interpretation["problems"]
        interpreted_by = interpretation["source"]
    else:
        interpreted_by = "caller"
    reply_style = reply_style or feature_requirements.DEFAULT_REPLY_STYLE

    prices = vendor_prices.get_prices(model_client=model_client)
    problems += prices.get("problems", [])
    try:
        aa_models, aa_problems = model_benchmarks.get_models()
        problems += aa_problems
    except Exception as e:
        aa_models = []
        problems.append(f"Latency and quality data could not be loaded ({type(e).__name__}: {e}); options are cost-only")

    options, counts, targets = build_analysis(prices["models"], aa_models, usage, latency_target, budget, reply_style, calibration, feature_category)
    base = {"volume_assumption": usage, "counts": counts, "caveats": caveats, "problems": problems,
            "interpretation": {"source": interpreted_by, "reply_style": reply_style, "feature_category": feature_category},
            "notes_written_by": "code",
            "learning": {"company_features_with_reports": len(calibration.company) if calibration else 0, "changed_verdicts": counts.get("calibration_changes", [])}}
    if not options:
        reason = ("no prices could be read" if not prices["models"] else
                  "latency and quality data could not be loaded, so the latency target could not be checked" if not aa_models and latency_target else
                  "nothing matched your targets")
        return user_input, dict(base, scenarios=[], grounding_status="ungrounded_fallback", overview="",
                                missing_inputs=["No option could be built: " + reason])

    latency_means = builder.LATENCY_MEANS[targets["latency_basis"]]
    output = dict(base,
        scenarios=options,
        grounding_status="grounded" if aa_models and prices["models"] else "ungrounded_fallback",
        missing_inputs=[],
        overview=code_overview(counts, {"latency": latency_target, "budget": budget}, usage, latency_means),
        latency_means=latency_means,
        accuracy_note="No public benchmark measures accuracy on your task. Quality percentiles are a general ranking; test the shortlist on your own examples.",
        attribution=builder.ATTRIBUTION,
    )
    return user_input, output


def log_recommendation(user_input, output):
    cosmos = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    container = cosmos.get_database_client("shipwiseai").get_container_client("recommendations")
    record = {"id": str(uuid.uuid4()), "agent": "model_stack_analysis", "timestamp": datetime.now(timezone.utc).isoformat(),
              "input": user_input, "output": output, "outcome": None}
    container.upsert_item(record)
    return record


if __name__ == "__main__":
    from foundry_clients import Clients
    user_input, output = call_model_stack_analysis(
        feature_description="A chatbot for support agents to answer customer questions from our existing knowledge base.",
        accuracy_target=">85%", latency_target="<400ms", budget="$40K annually",
        model_client=Clients().model_client,
    )
    print(json.dumps(output, indent=2))
