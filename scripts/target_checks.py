"""
Compares each Model Stack option's stated accuracy, latency and cost to the
PM's targets, in code, so the writing model copies a verdict instead of making
one up. Found necessary on 2026-10-01: the writing step said Claude 2 met a 90%
accuracy target when the sub-agent's own figure was 85-88%.

The sub-agent's figures are free text ("~$35K annually", "300-700ms", "85-88%"),
so parsing is deliberately strict. A figure is read only when it contains exactly
one number or one range with a clear unit. Anything else, a unit mismatch, or a
period that can't be matched comes back as "cannot be compared", never as a
guess. A wrong "meets" is the error that matters, so every doubt resolves to
"cannot be compared".
"""
import copy
import re

NUM = r"(\d[\d,]*(?:\.\d+)?)"
MONEY = re.compile(r"\$\s*" + NUM + r"\s*([kKmM])?\b(?:\s*(?:-|–|to)\s*\$?\s*" + NUM + r"\s*([kKmM])?\b)?")
PERCENT = re.compile(NUM + r"\s*(?:(?:-|–|to)\s*" + NUM + r"\s*)?(?:%|percent)")
TIME = re.compile(r"(?<![\w.])" + NUM + r"\s*(?:(?:-|–|to)\s*" + NUM + r"\s*)?(ms|milliseconds?|seconds?|secs?|s)\b", re.I)
MONTH = re.compile(r"month|/\s*mo\b", re.I)
YEAR = re.compile(r"year|annual|/\s*yr\b", re.I)


def _num(text):
    return float(text.replace(",", ""))


def _scale(suffix):
    return {"k": 1e3, "m": 1e6}.get((suffix or "").lower(), 1.0)


def parse_accuracy(text):
    """Returns (low, high) in percent, or None."""
    matches = PERCENT.findall(text or "")
    if len(matches) != 1:
        return None
    first, second = matches[0]
    low = _num(first)
    return (low, _num(second)) if second else (low, low)


def _target_accuracy(text):
    """The PM's target. Usually "90%", but a bare "90" is accepted too."""
    parsed = parse_accuracy(text)
    if parsed:
        return parsed
    bare = re.fullmatch(r"\s*(?:at least\s*)?" + NUM + r"\s*", text or "")
    return (_num(bare.group(1)),) * 2 if bare and _num(bare.group(1)) <= 100 else None


def parse_latency(text):
    """Returns (low, high) in milliseconds, or None."""
    matches = TIME.findall(text or "")
    if len(matches) != 1:
        return None
    first, second, unit = matches[0]
    factor = 1.0 if unit.lower().startswith("m") else 1000.0
    low = _num(first) * factor
    return (low, _num(second) * factor if second else low)


def parse_cost(text):
    """Returns ((low, high), period) in dollars per year, or None. The period
    must be stated; "$15K" with no period can't be compared to anything."""
    text = text or ""
    if "token" in text.lower():
        return None  # a per-token price, not a total
    matches = MONEY.findall(text)
    if len(matches) != 1:
        return None
    first, s1, second, s2 = matches[0]
    month, year = bool(MONTH.search(text)), bool(YEAR.search(text))
    if month == year:  # neither stated, or both
        return None
    factor = 12.0 if month else 1.0
    low = _num(first) * _scale(s1 or s2) * factor
    high = _num(second) * _scale(s2 or s1) * factor if second else low
    return (low, high)


def _target_cost(text):
    # Same rule as the option side: no period on the target, no comparison.
    return parse_cost(text)


def _verdict(kind, option, target):
    low, high = option
    if kind == "accuracy":  # higher is better; the whole range must clear it
        if low >= target:
            return "meets"
        return "misses" if high < target else "cannot be compared"
    if high <= target:  # latency and cost: lower is better
        return "meets"
    return "misses" if low > target else "cannot be compared"


def check_scenario(scenario, constraints):
    """One verdict per target the PM gave. Keys: accuracy, latency, cost."""
    checks = {}

    def record(kind, label, raw_option, option, target_text, target):
        if option is None or target is None:
            checks[label] = (f"cannot be compared: option states {raw_option.strip()!r}, target is {target_text!r}"
                             if raw_option else f"cannot be compared: the option states nothing for this (target {target_text!r})")
        else:
            checks[label] = f"{_verdict(kind, option, target)}: option states {raw_option.strip()!r}, target is {target_text!r}"

    t_acc, t_lat, t_bud = (constraints.get(k) for k in ("accuracy_target", "latency_target", "budget"))
    if t_acc and scenario.get("accuracy_is_proxy"):
        # The option only carries a general quality ranking. It can't be compared to an
        # accuracy percentage, whatever the number in it looks like.
        checks["accuracy"] = (f"cannot be compared: no benchmark measures accuracy on the PM's task (target is {t_acc!r}); "
                              f"{scenario.get('accuracy_notes', '')}".strip())
    elif t_acc:
        parsed = _target_accuracy(t_acc)
        record("accuracy", "accuracy", scenario.get("accuracy_notes") or "", parse_accuracy(scenario.get("accuracy_notes")),
               t_acc, parsed[0] if parsed else None)
    if t_lat:
        parsed = parse_latency(t_lat)
        record("latency", "latency", scenario.get("estimated_latency") or "", parse_latency(scenario.get("estimated_latency")),
               t_lat, parsed[1] if parsed else None)
    if t_bud:
        parsed = _target_cost(t_bud)
        record("cost", "cost", scenario.get("estimated_cost") or "", parse_cost(scenario.get("estimated_cost")),
               t_bud, parsed[1] if parsed else None)
    return checks


def add_target_checks(model_output, constraints):
    """Returns a copy of the Model Stack output with `target_check` on each
    scenario. The original (which gets saved) is left untouched."""
    out = copy.deepcopy(model_output)
    if not isinstance(out, dict) or not isinstance(out.get("scenarios"), list):
        return out
    volume = out.get("volume_assumption") if isinstance(out.get("volume_assumption"), dict) else {}
    for scenario in out["scenarios"]:
        if isinstance(scenario, dict):
            checks = check_scenario(scenario, constraints or {})
            # A cost verdict only holds at the request volume the agent used. If the
            # agent assumed that volume (the PM didn't give one), say so on the verdict.
            if "cost" in checks and volume.get("basis") == "assumed" and not checks["cost"].startswith("cannot"):
                checks["cost"] += f" (assumes {volume.get('requests_per_month', 'an unstated number of')} requests per month; the PM did not give a volume)"
            scenario["target_check"] = checks
    return out


# --------------------------------------------------------------------------
# Yearly cost, calculated in code
# --------------------------------------------------------------------------

def _number(value, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if (value > 0 or (allow_zero and value == 0)) else None


def compute_yearly_cost(scenario, volume):
    """Yearly dollars from the per-1M-token prices on the scenario and the
    request volume, or None if any of the inputs is missing or not a number.
    Added because the agent's own multiplication was wrong in all three options
    of the v6 test run (by up to a factor of 2, in both directions)."""
    in_price = _number(scenario.get("input_price_per_1m_tokens"), allow_zero=True)
    out_price = _number(scenario.get("output_price_per_1m_tokens"), allow_zero=True)
    requests = _number((volume or {}).get("requests_per_month"))
    in_tokens = _number((volume or {}).get("avg_input_tokens"))
    out_tokens = _number((volume or {}).get("avg_output_tokens"))
    if None in (in_price, out_price, requests, in_tokens, out_tokens):
        return None
    return requests * 12 * (in_tokens * in_price + out_tokens * out_price) / 1_000_000


def format_yearly_cost(dollars):
    if dollars >= 1_000_000:
        return f"${dollars / 1_000_000:.1f}M per year".replace(".0M", "M")
    if dollars >= 1000:
        return f"${dollars / 1000:.1f}K per year".replace(".0K", "K")
    return f"${dollars:.0f} per year"


def apply_computed_costs(model_output):
    """Returns a copy of the Model Stack output in which each option's
    `estimated_cost` is the code-calculated yearly cost. The agent's own figure is
    kept as `agent_stated_cost`. If the prices or volume are missing, the cost
    becomes "not calculated", which the target check reports as
    "cannot be compared" instead of passing on a number nobody verified."""
    out = copy.deepcopy(model_output)
    if not isinstance(out, dict) or not isinstance(out.get("scenarios"), list):
        return out
    volume = out.get("volume_assumption") if isinstance(out.get("volume_assumption"), dict) else {}
    for scenario in out["scenarios"]:
        if not isinstance(scenario, dict):
            continue
        scenario["agent_stated_cost"] = scenario.get("estimated_cost")
        dollars = compute_yearly_cost(scenario, volume)
        if dollars is None:
            scenario["estimated_cost"] = "not calculated: per-1M-token prices or request volume missing"
        else:
            scenario["estimated_cost"] = format_yearly_cost(dollars)
            scenario["cost_calculated_in_code"] = True
    return out
