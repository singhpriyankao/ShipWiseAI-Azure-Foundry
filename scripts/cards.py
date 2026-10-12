"""
The three scenario cards the PM Dashboard shows, picked from the Model Stack options.

The Model Stack step returns a shortlist (options that meet every target the PM gave) and up to two near
misses. The page shows three cards, chosen by a fixed rule that code applies, not a model:

  A  "Recommended"  the highest general-quality option among those that meet all targets
  B  "Lowest cost"  the cheapest of the remaining options that meet all targets
  C  "Caution"      the near miss that misses by the least, with what it misses written underneath

If there is no near miss, C is the fastest remaining option on the shortlist ("Fastest"). If nothing meets all
the targets, the cards are the closest near misses, all marked Caution, and none is called Recommended.

General quality is a percentile from a public benchmark: a proxy for how good a model is in general, not its
accuracy on the PM's task. The card says so. Cost is the yearly estimate divided by 12.
"""
from target_checks import parse_cost, parse_latency

BADGES = {"recommended": "Recommended", "lowest_cost": "Lowest cost", "caution": "Caution", "fastest": "Fastest"}


def _target_ms(constraints):
    parsed = parse_latency((constraints or {}).get("latency_target"))
    return parsed[1] if parsed else None


def _target_yearly(constraints):
    parsed = parse_cost((constraints or {}).get("budget"))
    return parsed[1] if parsed else None


def misses(option, constraints):
    """What a near miss falls short on, as plain sentences with the amounts, and how far in proportion to the
    target (used only to rank near misses). Accuracy is never listed: it cannot be compared."""
    found = []
    verdicts = option.get("verdicts") or {}
    if verdicts.get("latency") == "misses":
        target, value = _target_ms(constraints), option.get("latency_ms")
        if target and value:
            found.append({"target": "latency", "by": round(value - target), "relative": (value - target) / target,
                          "text": f"Misses latency by {value - target:,.0f} ms ({value:,.0f} ms against {target:,.0f} ms)"})
    if verdicts.get("cost") == "misses":
        target, value = _target_yearly(constraints), option.get("yearly_cost_usd")
        if target and value:
            found.append({"target": "cost", "by": round(value - target), "relative": (value - target) / target,
                          "text": f"Over budget by ${value - target:,.0f} a year (${value:,.0f} against ${target:,.0f})"})
    return found


def _closeness(option, constraints):
    """Smaller is closer: the largest relative miss among the targets it misses."""
    found = misses(option, constraints)
    return max((m["relative"] for m in found), default=float("inf"))


def _quality(option):
    value = option.get("quality_percentile")
    return value if isinstance(value, (int, float)) else -1.0


def _cost(option):
    value = option.get("yearly_cost_usd")
    return value if isinstance(value, (int, float)) else float("inf")


def _latency(option):
    value = option.get("latency_ms")
    return value if isinstance(value, (int, float)) else float("inf")


def _card(option, slot, badge, tone, constraints):
    yearly = option.get("yearly_cost_usd")
    means = option.get("latency_means") or ""
    return {
        "slot": slot,
        "badge": BADGES[badge],
        "tone": tone,
        "option_id": option.get("id"),
        "model": option.get("model_stack"),
        "yearly_cost_usd": yearly,
        "monthly_cost_usd": round(yearly / 12, 2) if isinstance(yearly, (int, float)) else None,
        "latency_ms": option.get("latency_ms"),
        "latency_kind": "to the first words" if "first answer" in means else "for the full reply",
        "quality_percentile": option.get("quality_percentile"),
        "quality_is_proxy": True,
        "verdicts": option.get("verdicts") or {},
        "misses": [m["text"] for m in misses(option, constraints)],
        "trade_offs": option.get("trade_offs"),
        "comparison": option.get("comparison"),
        "learned": option.get("learned") or [],
    }


def build_cards(model_output, constraints):
    """Up to three cards from the Model Stack output (the one with target checks added is fine). Returns []
    when there are no options."""
    scenarios = (model_output or {}).get("scenarios") or []
    shortlist = [s for s in scenarios if s.get("role") == "shortlist"]
    near = sorted((s for s in scenarios if s.get("role") == "near_miss"), key=lambda s: (_closeness(s, constraints), _cost(s)))
    cards = []

    if not shortlist:
        return [_card(s, "ABC"[i], "caution", "caution", constraints) for i, s in enumerate(near[:3])]

    best = max(shortlist, key=lambda s: (_quality(s), -_cost(s)))
    cards.append(_card(best, "A", "recommended", "good", constraints))
    rest = [s for s in shortlist if s is not best]
    cheapest = min(rest, key=lambda s: (_cost(s), -_quality(s))) if rest else None
    if cheapest:
        cards.append(_card(cheapest, "B", "lowest_cost", "neutral", constraints))
    if near:
        cards.append(_card(near[0], "C" if cheapest else "B", "caution", "caution", constraints))
    else:
        left = [s for s in rest if s is not cheapest]
        if left:
            fastest = min(left, key=lambda s: (_latency(s), _cost(s)))
            cards.append(_card(fastest, "C", "fastest", "neutral", constraints))
    return cards
