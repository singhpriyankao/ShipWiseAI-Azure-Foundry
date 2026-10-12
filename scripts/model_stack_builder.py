"""
Builds the Model Stack options in code, from data code fetched, so no number in an option
comes from a model's memory or a web search.

Inputs:  the vendor price list (vendor_prices.py) and Artificial Analysis's latency and
         quality benchmarks (model_benchmarks.py).
Output:  candidate configurations (one per model and reasoning-effort setting) with a
         yearly cost, an estimated response time and a general quality percentile, checked
         against the PM's latency and budget targets, then cut down to a short list.

What is and isn't compared to the PM's targets:
- Budget: the yearly cost is calculated here from per-token prices and a usage assumption.
- Latency: the time until the first answer token arrives is compared to the target, which is
  what a user perceives as "the reply starts" in a streaming chat. The estimated time for the
  whole reply (first answer token plus the output at the model's measured speed) is reported
  alongside but not compared. I first compared the whole reply, and on real data that made
  almost every option miss an 800ms target (a 250-token reply at 100 tokens a second takes
  2.5 seconds on its own), which says more about the default output size than about the models.
- Accuracy: never compared. No public benchmark measures accuracy on the PM's task. The
  quality percentile is a general ranking and is labelled a proxy.
"""
import re
from datetime import date, datetime, timedelta

from feature_requirements import DEFAULT_USAGE as DEFAULT_USAGE_FROM_REQUIREMENTS
from learning import accuracy_line, cost_line, latency_line
from model_benchmarks import benchmark_for
from target_checks import compute_yearly_cost, format_yearly_cost, parse_cost, parse_latency

# Used when nothing about volume is known. Deliberately modest, and always shown to the PM
# as an assumption. 1,000 input tokens matches the prompt length Artificial Analysis
# benchmarks latency at; 250 output tokens is a short chat reply.
DEFAULT_USAGE = DEFAULT_USAGE_FROM_REQUIREMENTS  # see feature_requirements.py
MAX_MODEL_AGE_DAYS = 365  # older models are not offered as new options
MAX_SHORTLIST = 5
MAX_PER_MODEL = 2
MAX_NEAR_MISSES = 2

ATTRIBUTION = "Latency and quality data: Artificial Analysis (https://artificialanalysis.ai/)"


def _released_recently(release_date, today):
    try:
        released = datetime.strptime(str(release_date)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return False  # an unknown date can't be shown to be recent
    return today - released <= timedelta(days=MAX_MODEL_AGE_DAYS)


def build_candidates(price_models, aa_models, usage, today=None):
    """One candidate per benchmarked configuration of each priced model. If there is no
    benchmark data at all, one cost-only candidate per priced model (and the caller says
    latency and quality were unavailable)."""
    today = today or date.today()
    candidates = []
    for pm in price_models:
        if pm.get("input_per_1m") is None or pm.get("output_per_1m") is None:
            continue
        priced = {"input_price_per_1m_tokens": pm["input_per_1m"], "output_price_per_1m_tokens": pm["output_per_1m"]}
        cost = compute_yearly_cost(priced, usage)
        base = {"vendor": pm["vendor"], "model": pm["model"], "price_source_url": pm.get("source_url"),
                "input_price_per_1m": pm["input_per_1m"], "output_price_per_1m": pm["output_per_1m"],
                "yearly_cost_usd": cost, "price_flag": pm.get("flag")}
        if not aa_models:
            candidates.append(dict(base, configuration=None, benchmarked=False, first_answer_ms=None, full_response_ms=None,
                                   intelligence_index=None, quality_percentile=None, notes=[]))
            continue
        bench = benchmark_for(pm, aa_models, usage["avg_output_tokens"])
        for cfg in (bench["configurations"] if bench else []):
            if not _released_recently(cfg.get("release_date"), today):
                continue
            candidates.append(dict(base, configuration=cfg["name"], benchmarked=True,
                                   first_answer_ms=cfg["first_answer_token_ms"], full_response_ms=cfg["full_response_ms"],
                                   intelligence_index=cfg["intelligence_index"], quality_percentile=cfg["quality_percentile"],
                                   notes=list(bench["notes"])))
    for i, c in enumerate(candidates):
        c["id"] = f"opt{i + 1}"
    return candidates


LATENCY_FIELDS = {"first_answer": "first_answer_ms", "full_response": "full_response_ms"}
LATENCY_MEANS = {
    "first_answer": "time until the first answer token arrives (the reply starts appearing)",
    "full_response": "time until the whole reply is complete",
}


def latency_of(candidate, targets):
    """The latency figure the PM's target is compared to: the time until the reply starts for a streamed
    chat, or until it is complete for a feature whose output is used only once it is finished."""
    return candidate.get(LATENCY_FIELDS[targets.get("latency_basis", "first_answer")])


def apply_calibration(candidates, calibration, category, latency_basis):
    """Corrects the benchmark forecast with what this company's own confirmed reports have taught about each model setting
    (learning.py), and attaches plain sentences saying so. A figure only moves when learning.ratio_stats says the evidence
    is enough (three separate features that agree); below that the sentences say what has been reported and that nothing
    changed. The original forecast is always kept in `raw_forecast`, so a later report is compared with the benchmark
    and not with an already-corrected number (corrections must not compound). Only the latency that is compared
    against the PM's target is corrected, and only from reports of the same kind."""
    field = LATENCY_FIELDS[latency_basis]
    for c in candidates:
        c["raw_forecast"] = {"latency_ms": c.get(field), "yearly_cost_usd": c.get("yearly_cost_usd")}
        c["learned"] = []
        if calibration is None:
            continue
        info = calibration.for_option(option_label(c), category)
        lines = []
        stats = info["latency"].get(latency_basis)
        if stats and c.get(field) is not None:
            raw = c[field]
            if stats["applies"]:
                c[field] = round(raw * stats["median"])
                c["corrected"] = True
            lines.append(latency_line(stats, raw, c[field]))
        stats = info["cost"]
        if stats["n"] and c.get("yearly_cost_usd") is not None:
            raw = c["yearly_cost_usd"]
            if stats["applies"]:
                c["yearly_cost_usd"] = raw * stats["median"]
                c["corrected"] = True
            lines.append(cost_line(stats, raw, c["yearly_cost_usd"], _money))
        lines += accuracy_line(info["accuracy"], info["pooled_accuracy"], category)
        c["learned"] = [line for line in lines if line]


def calibration_changes(candidates, targets, limit=5):
    """Plain sentences for the options whose forecast was corrected AND whose verdict against the PM's own targets changed
    because of it. An option that quietly drops off the shortlist, or onto it, would be baffling without this. Call after
    shortlist() has set each candidate's verdicts."""
    changes = []
    for c in candidates:
        if not c.get("corrected") or not c.get("raw_forecast"):
            continue
        before = dict(c, yearly_cost_usd=c["raw_forecast"]["yearly_cost_usd"])
        before[LATENCY_FIELDS[targets.get("latency_basis", "first_answer")]] = c["raw_forecast"]["latency_ms"]
        was, now = evaluate(before, targets), c["verdicts"]
        moved = [(k, was[k], now[k]) for k in ("latency", "cost") if was[k] != now[k] and {was[k], now[k]} <= {"meets", "misses"}]
        if moved:
            what = " and ".join(f"{k} (it {'met' if w == 'meets' else 'missed'} your target on the benchmark forecast and now {'meets' if n == 'meets' else 'misses'} it)" for k, w, n in moved)
            changes.append(f"{option_label(c)}: your company's reports changed its {what}.")
    return changes[:limit]


def parse_targets(latency_target, budget, reply_style="streamed_chat"):
    """Latency in ms and yearly dollars, or None where the target is absent or can't be read
    (a budget with no stated period can't be compared, so it counts as unreadable)."""
    lat = parse_latency(latency_target) if latency_target else None
    bud = parse_cost(budget) if budget else None
    return {"latency_basis": "first_answer" if reply_style == "streamed_chat" else "full_response",
            "latency_ms": lat[1] if lat else None, "budget_usd": bud[1] if bud else None,
            "latency_given": bool(latency_target), "budget_given": bool(budget)}


def evaluate(candidate, targets):
    """'meets', 'misses', 'unknown' (target given but the figure or target can't be read) or
    'no target' for each of latency and cost."""
    if not targets["latency_given"]:
        latency = "no target"
    elif targets["latency_ms"] is None or latency_of(candidate, targets) is None:
        latency = "unknown"
    else:
        latency = "meets" if latency_of(candidate, targets) <= targets["latency_ms"] else "misses"
    if not targets["budget_given"]:
        cost = "no target"
    elif targets["budget_usd"] is None or candidate["yearly_cost_usd"] is None:
        cost = "unknown"
    else:
        cost = "meets" if candidate["yearly_cost_usd"] <= targets["budget_usd"] else "misses"
    return {"latency": latency, "cost": cost}


def _quality_key(c):
    return -(c["quality_percentile"] if c["quality_percentile"] is not None else -1)


def _overshoot(c, targets):
    """How far past its target a candidate's missed figure is, as a ratio (1.2 = 20% over)."""
    ratios = []
    if c["verdicts"]["latency"] == "misses":
        ratios.append(latency_of(c, targets) / targets["latency_ms"])
    if c["verdicts"]["cost"] == "misses":
        ratios.append(c["yearly_cost_usd"] / targets["budget_usd"])
    return max(ratios) if ratios else 0.0


def shortlist(candidates, targets):
    """Cuts the candidates to the few worth showing: the highest-quality options that meet
    every target the PM gave (at most two per model, five in all, and the cheapest one is
    always included), plus up to two near misses (the ones that miss exactly one target by
    the least), so the PM sees what a small trade-off would buy."""
    for c in candidates:
        c["verdicts"] = evaluate(c, targets)
    ok = ("meets", "no target")
    fits = [c for c in candidates if c["verdicts"]["latency"] in ok and c["verdicts"]["cost"] in ok]
    misses_one = [c for c in candidates
                  if [v for v in c["verdicts"].values() if v == "misses"] and
                  len([v for v in c["verdicts"].values() if v == "misses"]) == 1 and "unknown" not in c["verdicts"].values()]

    chosen, per_model = [], {}
    for c in sorted(fits, key=lambda c: (_quality_key(c), c["yearly_cost_usd"] if c["yearly_cost_usd"] is not None else 1e18)):
        key = (c["vendor"], c["model"])
        if per_model.get(key, 0) < MAX_PER_MODEL and len(chosen) < MAX_SHORTLIST:
            chosen.append(c)
            per_model[key] = per_model.get(key, 0) + 1
    priced_fits = [c for c in fits if c["yearly_cost_usd"] is not None]
    if priced_fits:
        cheapest = min(priced_fits, key=lambda c: c["yearly_cost_usd"])
        if cheapest not in chosen:
            key = (cheapest["vendor"], cheapest["model"])
            if per_model.get(key, 0) >= MAX_PER_MODEL:
                # keep the cap: swap out that model's lowest-quality entry (chosen is sorted best first)
                last = max(i for i, c in enumerate(chosen) if (c["vendor"], c["model"]) == key)
                chosen.pop(last)
            elif len(chosen) >= MAX_SHORTLIST:
                chosen.pop()
            chosen.append(cheapest)
    near = sorted(misses_one, key=lambda c: (_overshoot(c, targets), _quality_key(c)))[:MAX_NEAR_MISSES]

    def count(pred):
        return sum(1 for c in candidates if pred(c))
    counts = {
        "considered": len(candidates),
        "benchmarked": count(lambda c: c["benchmarked"]),
        "meet_latency": count(lambda c: c["verdicts"]["latency"] == "meets"),
        "meet_budget": count(lambda c: c["verdicts"]["cost"] == "meets"),
        "meet_all_given_targets": len(fits),
        "latency_unknown": count(lambda c: c["verdicts"]["latency"] == "unknown"),
        "cost_unknown": count(lambda c: c["verdicts"]["cost"] == "unknown"),
    }
    return {"shortlist": chosen, "near_misses": near, "counts": counts}


VENDOR_NAMES = {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Google"}


def option_label(candidate):
    """"Anthropic Claude Haiku 4.5 (Non-reasoning)": the vendor, the model and the setting only. The
    benchmark's own name repeats the model ("Claude 4.5 Haiku (Non-reasoning)"), so only the bracketed
    setting is kept."""
    label = f"{VENDOR_NAMES.get(candidate['vendor'], candidate['vendor'].title())} {candidate['model']}"
    config = candidate.get("configuration") or ""
    if config.endswith(")") and "(" in config:
        label += f" ({config[config.rfind('(') + 1:-1]})"
    return label


def ordinal(n):
    """83 -> "83rd", 11 -> "11th", 21 -> "21st". The percentile used to print as "83th"."""
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _money(dollars):
    if dollars >= 1_000_000:
        return f"${dollars / 1_000_000:.1f}M".replace(".0M", "M")
    if dollars >= 1000:
        return f"${dollars / 1000:.1f}K".replace(".0K", "K")
    return f"${dollars:.0f}"


def compare_options(options, targets):
    """One plain sentence per option comparing it with the others shown, and stating by how much it
    misses a target. Written by code from the options' numbers: an agent-written note once called a
    mid-priced option "higher cost than other options", and only numbers were being checked, not
    comparisons. Returns {option id: sentence}. Each clause is only written when the figures exist for
    this option and at least one other."""
    out = {}
    for o in options:
        others = [x for x in options if x["id"] != o["id"]]
        clauses = []

        def pick(field):
            have = [(x[field], x) for x in options if x.get(field) is not None]
            return o.get(field) is not None and len(have) > 1, have

        ok, have = pick("quality_percentile")
        if ok:
            best = max(have, key=lambda t: t[0])
            others_best = max((v for v, x in have if x["id"] != o["id"]), default=None)
            if o["quality_percentile"] > others_best:
                clauses.append(f"ranks highest in general quality ({o['quality_percentile']:g}, against {others_best:g} for the next)")
            elif o["quality_percentile"] == best[0]:
                clauses.append("ties for the highest general quality")
            else:
                clauses.append(f"ranks {best[0] - o['quality_percentile']:.1f} percentile points below the highest ({best[1]['model_stack']}, {best[0]:g}) in general quality".replace(".0 percentile", " percentile"))
        elif o.get("quality_percentile") is None:
            clauses.append("has no general quality ranking to compare")

        ok, have = pick("latency_ms")
        if ok:
            fastest = min(have, key=lambda t: t[0])
            verb = "starts replying" if targets.get("latency_basis", "first_answer") == "first_answer" else "finishes its reply"
            if o["latency_ms"] == fastest[0]:
                clauses.append(f"{verb} soonest ({o['latency_ms']}ms)")
            else:
                clauses.append(f"{verb} {o['latency_ms'] - fastest[0]}ms later than the fastest ({fastest[1]['model_stack']}, {fastest[0]}ms)")

        ok, have = pick("yearly_cost_usd")
        if ok:
            cheapest = min(have, key=lambda t: t[0])
            if o["yearly_cost_usd"] == cheapest[0]:
                clauses.append(f"costs the least ({_money(o['yearly_cost_usd'])} a year)")
            else:
                clauses.append(f"costs {_money(o['yearly_cost_usd'] - cheapest[0])} more a year than the cheapest ({cheapest[1]['model_stack']}, {_money(cheapest[0])})")

        if o["verdicts"]["latency"] == "misses" and targets.get("latency_ms"):
            clauses.append(f"misses the latency target by {o['latency_ms'] - targets['latency_ms']:.0f}ms ({o['latency_ms']}ms against {targets['latency_ms']:.0f}ms)")
        if o["verdicts"]["cost"] == "misses" and targets.get("budget_usd"):
            clauses.append(f"is over budget by {_money(o['yearly_cost_usd'] - targets['budget_usd'])} a year")

        out[o["id"]] = ("Compared with the other options shown, it " + "; ".join(clauses) + ".") if clauses and others else ""
    return out


_EFFORT = re.compile(r"\b(xhigh|max|high|medium|low)\b")


def describe_option(option):
    """One plain description of an option on its own, written by code: what its setting means, and what its
    target verdicts are. It replaces the agent's note; comparisons between options are
    separate (`compare_options`) and are never made here."""
    qualifier = option["model_stack"][option["model_stack"].rfind("(") + 1:-1].lower() if option["model_stack"].endswith(")") else ""
    if "non-reasoning" in qualifier:
        setting = "Answers directly, without a separate thinking step."
    elif "reasoning" in qualifier or _EFFORT.search(qualifier):
        effort = _EFFORT.search(qualifier)
        setting = "Thinks before it answers" + (f"; the {effort.group(1)} effort setting controls how long it may think." if effort else ".")
    else:
        setting = ""
    v, parts = option["verdicts"], []
    if v["latency"] == "meets":
        parts.append("meets the latency target")
    elif v["latency"] == "misses":
        parts.append("misses the latency target")
    elif v["latency"] == "unknown":
        parts.append("could not be compared on latency")
    if v["cost"] == "meets":
        parts.append("fits the budget")
    elif v["cost"] == "misses":
        parts.append("is over budget")
    verdict = ("It " + " and ".join(parts) + ".") if parts else ""
    return " ".join(x for x in (setting, verdict) if x)


def to_scenario(candidate, role, usage, targets=None):
    """An option in the shape the rest of the system already reads (strings the target check
    can parse), with the raw figures alongside. `role` is 'shortlist' or 'near_miss'."""
    cost = candidate["yearly_cost_usd"]
    pct = candidate["quality_percentile"]
    name = option_label(candidate)
    if candidate["benchmarked"]:
        accuracy = ("No benchmark measures accuracy on your task. "
                    + (f"Ranks at the {ordinal(pct)} percentile of benchmarked models in general (a proxy, not an accuracy figure)."
                       if pct is not None else "No general quality ranking is available for this configuration."))
    else:
        accuracy = "No benchmark data was available, so neither accuracy nor latency is known."
    return {
        "id": candidate["id"], "role": role,
        "model_stack": name,
        "estimated_cost": format_yearly_cost(cost) if cost is not None else "not calculated",
        "yearly_cost_usd": cost,
        "first_answer_ms": candidate["first_answer_ms"],
        "full_response_ms": candidate["full_response_ms"],
        "latency_ms": latency_of(candidate, targets or {}),
        "estimated_latency": f"{latency_of(candidate, targets or {})}ms" if latency_of(candidate, targets or {}) is not None else "not available",
        "latency_means": LATENCY_MEANS[(targets or {}).get("latency_basis", "first_answer")],
        "accuracy_notes": accuracy,
        "accuracy_is_proxy": True,
        "quality_percentile": pct,
        "intelligence_index": candidate["intelligence_index"],
        "input_price_per_1m_tokens": candidate["input_price_per_1m"],
        "output_price_per_1m_tokens": candidate["output_price_per_1m"],
        "price_basis": (f"${candidate['input_price_per_1m']:g} per 1M input tokens and ${candidate['output_price_per_1m']:g} per 1M output tokens; "
                        f"{usage['requests_per_month']:,} requests a month x 12 x ({usage['avg_input_tokens']:,} input and {usage['avg_output_tokens']:,} output tokens"
                        + (", sizes estimated from the kind of feature)" if usage.get("token_basis") == "estimated" else ")")),
        "source": "Prices: the vendor's pricing page. Latency and quality: Artificial Analysis (artificialanalysis.ai).",
        "verdicts": candidate["verdicts"],
        "notes": candidate["notes"] + ([candidate["price_flag"]] if candidate.get("price_flag") else []),
        "trade_offs": "",
        "comparison": "",
        "learned": candidate.get("learned", []),
        "raw_forecast": candidate.get("raw_forecast"),
    }
