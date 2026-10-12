"""
The learning stage: what the confirmed outcome reports teach about each model setting.

Plain arithmetic (a running average of forecast error per
company), not an agent and not a model. A model reasoning about *why* a forecast missed would need far more
reports than ten users will produce; with the data there is, it would only write confident guesses.

What is learned, and what is deliberately not:

- Latency and cost have a forecast, so each report gives a ratio: reported / forecast. Per model setting, the
  median ratio becomes a correction, but only when three conditions hold: at least three separate features
  reported (three reports about one project count once, the latest), the ratios agree with each other (every
  one within half to double the median), and the reports are like-for-like (below).
- Accuracy has no forecast, because no public benchmark measures it on a PM's task, so nothing is adjusted. What
  PMs report is shown as a range with the number of reports and the reminder that each PM measured it on their
  own test set. It is evidence, not a calibration.
- Nothing is changed silently. A corrected figure always sits beside the original forecast and the number of
  reports behind it. Below the threshold the reports are shown and nothing moves.

Like-for-like rules, because a ratio of two different things teaches nothing:
- Latency: the reported kind (time to the first words, or to the whole reply) must match what was forecast, and the
  statistic must be a median or average, since the forecast is a median. A 95th-percentile figure is kept on the
  record and not used.
- Cost: only model cost (API or token charges) is ever reported. The forecast depended on a volume
  assumption, so the forecast is first scaled to the volume the PM actually reported; without a reported volume (and
  with an assumed one) the report can't separate price error from volume error and isn't used for cost.
- Reports that name a different model than the one recorded are never used.

Scope: corrections use the same company's reports (calibrated to *your* data). Accuracy evidence from other
companies is shown only as an anonymous range when at least three different companies have reported, and is never
applied to anything.
"""
from statistics import median

from target_checks import _target_accuracy, parse_cost, parse_latency

MIN_FEATURES = 3
CONSISTENCY = (0.5, 2.0)          # every ratio must sit within this band around the median
MIN_POOLED_COMPANIES = 3
COMPARABLE_STATISTICS = ("median", "average")


# --------------------------------------------------------------------------
# One observation per feature
# --------------------------------------------------------------------------

def latest_observations(records, company_id=None):
    """[(record, outcome)]: the latest confirmed report on each recommendation that wasn't about a different model.
    One per recommendation, so a project that reports at 30, 60 and 90 days counts once."""
    pairs = []
    for record in records:
        if company_id is not None and record.get("company_id") != company_id:
            continue
        usable = [o for o in record.get("outcomes", []) if o.get("confirmed_by_pm") and not o.get("model_differs")]
        if usable:
            pairs.append((record, max(usable, key=lambda o: o.get("reported_at", ""))))
    return pairs


def forecast_basis(forecast):
    means = str(forecast.get("latency_means") or "").lower()
    return "first_answer" if ("reply starts" in means or "first answer" in means) else "full_response"


def _raw(record, key):
    """The forecast as the benchmark gave it, before any learned correction. Records written before corrections
    existed have only the one forecast."""
    chosen = record["chosen"]
    return (chosen.get("forecast_raw") or chosen["forecast"]).get(key)


def latency_observation(record, outcome):
    """{"basis", "ratio", ...} or (None, why)."""
    reported = outcome["fields"].get("latency")
    forecast = _raw(record, "latency_ms")
    if not reported or not forecast:
        return None, "no latency figure to compare"
    basis = forecast_basis(record["chosen"]["forecast"])
    if reported["kind"] != basis:
        return None, "the reported latency is not the same kind as the forecast one"
    if reported["statistic"] not in COMPARABLE_STATISTICS:
        return None, "the forecast is a median, and the reported figure is not a median or average"
    return {"basis": basis, "ratio": reported["value_ms"] / forecast, "forecast_ms": forecast, "reported_ms": reported["value_ms"]}, None


def cost_observation(record, outcome):
    reported = outcome["fields"].get("cost")
    forecast = _raw(record, "yearly_cost_usd")
    if not reported or not forecast:
        return None, "no cost figure to compare"
    usage = record.get("usage_assumption") or {}
    assumed = usage.get("requests_per_month")
    volume = outcome["fields"].get("volume")
    if volume and assumed:
        scaled = forecast * volume["requests_per_month"] / assumed       # the forecast at the volume that really happened
    elif usage.get("volume_basis") == "stated":
        scaled = forecast                                                  # the PM's own volume was already used
    else:
        return None, "the forecast assumed a volume and none was reported, so price can't be told from volume"
    return {"ratio": reported["yearly_usd"] / scaled, "forecast_usd": scaled, "reported_usd": reported["yearly_usd"]}, None


def accuracy_observation(record, outcome):
    reported = outcome["fields"].get("accuracy")
    if not reported:
        return None
    return {"value": reported["value_percent"], "how_measured": reported.get("how_measured"), "company_id": record.get("company_id")}


# --------------------------------------------------------------------------
# Turning observations into something usable
# --------------------------------------------------------------------------

def ratio_stats(ratios):
    """{"n", "median", "min", "max", "applies", "why"}. `applies` is the whole rule: enough separate features, and
    agreeing."""
    if not ratios:
        return {"n": 0, "median": None, "min": None, "max": None, "applies": False, "why": "no reports"}
    m = median(ratios)
    low, high = CONSISTENCY
    consistent = all(low * m <= r <= high * m for r in ratios)
    if len(ratios) < MIN_FEATURES:
        why = f"{len(ratios)} of the {MIN_FEATURES} separate features needed"
    elif not consistent:
        why = "the reports disagree too much to apply"
    else:
        why = "applied"
    return {"n": len(ratios), "median": m, "min": min(ratios), "max": max(ratios), "applies": len(ratios) >= MIN_FEATURES and consistent, "why": why}


def accuracy_stats(values):
    if not values:
        return {"n": 0}
    nums = sorted(v["value"] for v in values)
    return {"n": len(nums), "min": nums[0], "max": nums[-1], "median": median(nums), "measured_by_how": sum(1 for v in values if v.get("how_measured"))}


class Calibration:
    """What the reports say about each model setting. Built once from the stored records."""

    def __init__(self, records, company_id):
        self.company_id = company_id
        self.company = latest_observations(records, company_id)
        self.everyone = latest_observations(records)

    def for_option(self, label, category=None):
        """What is known about one model setting for one kind of feature: latency by basis, cost, accuracy evidence, and
        the pooled accuracy range if enough different companies have reported."""
        mine = [(r, o) for r, o in self.company if r["chosen"]["label"] == label]
        latency, cost = {}, []
        skipped = {"latency": 0, "cost": 0}
        for r, o in mine:
            obs, _ = latency_observation(r, o)
            if obs:
                latency.setdefault(obs["basis"], []).append(obs["ratio"])
            elif o["fields"].get("latency"):
                skipped["latency"] += 1
            obs, _ = cost_observation(r, o)
            if obs:
                cost.append(obs["ratio"])
            elif o["fields"].get("cost"):
                skipped["cost"] += 1
        same_kind = [accuracy_observation(r, o) for r, o in mine if r["feature"].get("category") == category]
        accuracy = [a for a in same_kind if a]
        pooled = [a for a in (accuracy_observation(r, o) for r, o in self.everyone
                              if r["chosen"]["label"] == label and r["feature"].get("category") == category) if a]
        pooled_ok = len({a["company_id"] for a in pooled}) >= MIN_POOLED_COMPANIES
        return {
            "label": label, "reports": len(mine),
            "latency": {basis: ratio_stats(v) for basis, v in latency.items()},
            "cost": ratio_stats(cost),
            "unused": skipped,
            "accuracy": accuracy_stats(accuracy),
            "pooled_accuracy": accuracy_stats(pooled) if pooled_ok else {"n": 0},
        }


# --------------------------------------------------------------------------
# Plain sentences, written by code
# --------------------------------------------------------------------------

def _x(r):
    return f"{r:.2f}x"


def latency_line(stats, raw_ms, adjusted_ms):
    if stats["n"] == 0:
        return None
    spread = f"{_x(stats['min'])} to {_x(stats['max'])}" if stats["n"] > 1 else _x(stats["median"])
    if stats["applies"]:
        return (f"Latency: {stats['n']} reports from your company put this setting at {_x(stats['median'])} the forecast ({spread}), "
                f"so the forecast was adjusted from {raw_ms}ms to {adjusted_ms}ms.")
    return (f"Latency: {stats['n']} report{'s' if stats['n'] != 1 else ''} from your company ({spread} the forecast) so far; "
            f"{stats['why']}, so the forecast was not changed.")


def cost_line(stats, raw_usd, adjusted_usd, money):
    if stats["n"] == 0:
        return None
    spread = f"{_x(stats['min'])} to {_x(stats['max'])}" if stats["n"] > 1 else _x(stats["median"])
    if stats["applies"]:
        return (f"Cost: {stats['n']} reports from your company put model cost at {_x(stats['median'])} the forecast ({spread}), "
                f"so the forecast was adjusted from {money(raw_usd)} to {money(adjusted_usd)} a year.")
    return (f"Cost: {stats['n']} report{'s' if stats['n'] != 1 else ''} from your company ({spread} the forecast) so far; "
            f"{stats['why']}, so the forecast was not changed.")


def accuracy_line(stats, pooled, category):
    kind = (category or "this kind of").replace("_", " ")
    lines = []
    if stats["n"]:
        rng = f"{stats['min']:g}%" if stats["min"] == stats["max"] else f"{stats['min']:g}% to {stats['max']:g}%"
        lines.append(f"Accuracy your company reported for this setting on {kind} features: {rng} ({stats['n']} report{'s' if stats['n'] != 1 else ''}, "
                     "each measured on the reporter's own test set, not by ShipWise).")
    if pooled["n"]:
        rng = f"{pooled['min']:g}%" if pooled["min"] == pooled["max"] else f"{pooled['min']:g}% to {pooled['max']:g}%"
        lines.append(f"Across companies, anonymously: {rng} ({pooled['n']} reports). Shown as evidence only; nothing is adjusted from it.")
    return lines


# --------------------------------------------------------------------------
# Did the shipped feature meet the PM's own targets?
# --------------------------------------------------------------------------

def _targets_met(record, outcome):
    """{"accuracy": True/False/None, "latency": ..., "cost": ...}: None where the PM set no target or reported no figure
    that can be compared."""
    given = record.get("constraints_given") or {}
    fields = outcome["fields"]
    result = {"accuracy": None, "latency": None, "cost": None}

    target = _target_accuracy(given.get("accuracy_target")) if given.get("accuracy_target") else None
    if target and fields.get("accuracy"):
        result["accuracy"] = fields["accuracy"]["value_percent"] >= target[0]

    lat_target = parse_latency(given.get("latency_target")) if given.get("latency_target") else None
    reported = fields.get("latency")
    if lat_target and reported:
        basis = forecast_basis(record["chosen"]["forecast"])
        if reported["kind"] in (basis, "unspecified"):
            result["latency"] = reported["value_ms"] <= lat_target[1]

    budget = parse_cost(given.get("budget")) if given.get("budget") else None
    if budget and fields.get("cost"):
        result["cost"] = fields["cost"]["yearly_usd"] <= budget[1]
    return result


def decision_quality(records, company_id=None):
    """The share of shipped features whose reported results met the targets the PM set themselves (accuracy, latency
    and budget, each only where a target was set and a comparable figure reported). No forecast of accuracy is needed,
    which is the point: it is measurable. Returns counts too, because a rate over two features means little."""
    pairs = latest_observations(records, company_id)
    comparable = met_all = 0
    per_target = {"accuracy": [0, 0], "latency": [0, 0], "cost": [0, 0]}
    for record, outcome in pairs:
        met = _targets_met(record, outcome)
        checked = {k: v for k, v in met.items() if v is not None}
        if not checked:
            continue
        comparable += 1
        met_all += all(checked.values())
        for k, v in checked.items():
            per_target[k][1] += 1
            per_target[k][0] += bool(v)
    return {"features_with_reports": len(pairs), "comparable": comparable, "met_all": met_all,
            "rate": (met_all / comparable) if comparable else None,
            "by_target": {k: {"met": m, "of": n} for k, (m, n) in per_target.items()}}
