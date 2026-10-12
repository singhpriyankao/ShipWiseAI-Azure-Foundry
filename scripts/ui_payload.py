"""
What the page needs from one finished request, shaped by code: the three verdicts for the
top of the page, the legal block, up to three scenario cards, how to read the options, the competitor matrix with
its coverage, the guardrails that fired, and the written summary. Nothing here calls a model or adds a claim; every
field is copied or computed from what the agents and the checks already produced.
"""
import re
from urllib.parse import urlparse

from act_text import citation_label
from cards import build_cards
from recommendation_store import can_accept

# What the legal block looks like for each legal result.
BANNER_TONE = {
    "prohibited": "blocked",
    "high_risk": "warning",
    "transparency_obligated": "notice",
    "no_specific_obligation_identified": "clear",
}
BANNER_TITLE = {
    "prohibited": "Prohibited under the EU AI Act",
    "high_risk": "High-risk under the EU AI Act",
    "transparency_obligated": "Transparency duty under the EU AI Act",
    "no_specific_obligation_identified": "No specific obligation found under the EU AI Act",
    "missing_inputs": "More detail needed for the legal check",
    "insufficient_information": "More detail needed for the legal check",
    "out_of_scope": "Outside what the legal check covers",
    "analysis_failed": "The legal check could not finish",
}
# The short name shown on the verdict tile.
SHORT_LABEL = {
    "prohibited": "Prohibited",
    "high_risk": "High-risk",
    "transparency_obligated": "Transparency duty",
    "no_specific_obligation_identified": "No specific obligation",
    "missing_inputs": "Needs more detail",
    "insufficient_information": "Needs more detail",
    "out_of_scope": "Out of scope",
    "analysis_failed": "Check did not finish",
}

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_FULL_DATE = re.compile(rf"\b\d{{1,2}} (?:{_MONTHS}) \d{{4}}\b")


def _ok_output(results, name):
    result = (results or {}).get(name) or {}
    return result.get("output") if result.get("status") == "ok" else None


def _money(dollars):
    if dollars is None:
        return "n/a"
    return f"${dollars / 1000:.1f}K".replace(".0K", "K") if dollars >= 1000 else f"${dollars:,.0f}"


def regulatory_block(results):
    """The legal result. `status` says whether the check ran at all, so a failed check is never shown as 'no
    obligation'. `quotes` is filled in later by the server (act_text.py) and is empty here."""
    result = (results or {}).get("regulations")
    if not result:
        return {"status": "not_run"}
    output = _ok_output(results, "regulations")
    if output is None:
        return {"status": "failed", "title": BANNER_TITLE["analysis_failed"], "short": SHORT_LABEL["analysis_failed"], "tone": "unclear", "answer": result.get("error"), "dates": [], "citations": [], "quotes": []}
    classification = output.get("classification")
    status_text = output.get("applicability_status")
    return {
        "status": "checked",
        "classification": classification,
        "title": BANNER_TITLE.get(classification, "Legal result"),
        "short": SHORT_LABEL.get(classification, "Legal result"),
        "tone": BANNER_TONE.get(classification, "unclear"),
        "answer": output.get("answer"),
        "applicability_status": status_text,
        "dates": list(dict.fromkeys(_FULL_DATE.findall(status_text or ""))),
        "citations": [citation_label(c) for c in output.get("citations") or []],
        "quotes": [],
        "missing_inputs": output.get("missing_inputs") or [],
        "disclaimer": output.get("disclaimer"),
    }


def chips(constraints, model_output):
    """The targets the page shows as chips. Accuracy is left out on purpose: no benchmark measures it on the PM's
    task, so a chip for it would suggest a check that doesn't happen. Volume is shown because it drives cost most,
    with whether the PM stated it or it was assumed. The reply style is read from the description, so it is shown
    too (the PM may not have said whether answers stream)."""
    usage = (model_output or {}).get("volume_assumption") or {}
    style = ((model_output or {}).get("interpretation") or {}).get("reply_style")
    return {
        "latency": (constraints or {}).get("latency_target"),
        "budget": (constraints or {}).get("budget"),
        "volume": ({"requests_per_month": usage.get("requests_per_month"), "basis": usage.get("volume_basis") or usage.get("basis")}
                   if usage.get("requests_per_month") else None),
        "reply_style": {"streamed_chat": "streamed chat", "single_response": "one finished reply"}.get(style),
        "accuracy_target_given": bool((constraints or {}).get("accuracy_target")),
    }


def reading_block(model_output):
    """'How to read these options', as separate facts the page can lay out: what was compared, what can't be told,
    and the assumptions behind the cost and latency figures."""
    if not model_output:
        return None
    counts = model_output.get("counts") or {}
    usage = model_output.get("volume_assumption") or {}
    interpretation = model_output.get("interpretation") or {}
    return {
        "compared": counts.get("considered"),
        "meet_all": counts.get("meet_all_given_targets"),
        "latency_unknown": counts.get("latency_unknown"),
        "accuracy_note": model_output.get("accuracy_note"),
        "latency_means": model_output.get("latency_means"),
        "assumptions": {
            "requests_per_month": usage.get("requests_per_month"), "volume_basis": usage.get("volume_basis") or usage.get("basis"),
            "avg_input_tokens": usage.get("avg_input_tokens"), "avg_output_tokens": usage.get("avg_output_tokens"), "token_basis": usage.get("token_basis"),
            "reply_style": interpretation.get("reply_style"),
        },
        "caveats": model_output.get("caveats") or [],
        "attribution": model_output.get("attribution"),
    }


def _host(url):
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return None


def competitors_block(results):
    """One row per competitor with each claim's check result, and the hosts of the pages it cited (so a reader can
    see when a claim rested on a page that is not the vendor's own)."""
    output = _ok_output(results, "competitive_analysis")
    if output is None:
        return None
    rows = []
    for c in output.get("competitors") or []:
        urls = [c.get(f"{f}_source_url") for f in ("pricing", "accuracy", "latency")]
        rows.append({"product": c.get("product"), "vendor": c.get("vendor"),
                     "cited_hosts": list(dict.fromkeys(h for h in (_host(u) for u in urls if u) if h)),
                     "claims": [{"field": field, "note": c.get(f"{field}_notes"), "source_url": c.get(f"{field}_source_url"),
                                 "check": (c.get("source_checks") or {}).get(field)} for field in ("pricing", "accuracy", "latency")]})
    return rows


def competitor_coverage(rows):
    """How many of the claims could be backed by their source. A claim counts as verified only if the cited page was
    opened and supported it."""
    if rows is None:
        return None
    checks = [c.get("check") or "" for r in rows for c in r["claims"]]
    return {"total": len(checks), "verified": sum(1 for c in checks if c == "verified"),
            "not_supported": sum(1 for c in checks if c == "page_does_not_support_claim"),
            "could_not_check": sum(1 for c in checks if c.startswith("could_not_be_checked")),
            "no_source": sum(1 for c in checks if c in ("no_source_stated", "no_source_note_replaced")),
            "unchecked": sum(1 for c in checks if c == "")}


_GUARDRAIL_TEXT = {
    "chat_assistant_reply_style_corrected": "Read the feature as a streamed chat, so latency is judged on the first words",
    "applicability_built_in_code": "The date the rules apply was built from Article 113 by code, not taken from the model",
    "citation_check": "A citation was not in the retrieved law, so the legal answer was rejected",
    "retry_after_failed_guardrail": "The legal check was run again after a guardrail rejected its first answer",
    "regulatory_finding_kept_in_reply": "The written summary left out part of the legal finding; it was put back",
    "invented_question_removed": "A question the system had not asked was removed from the summary",
    "prohibited_feature_model_withheld": "No model recommendation was given because the feature is prohibited",
    "prohibited_feature_no_model_run": "No model comparison was run because the feature is prohibited",
    "high_risk_pick_flagged_for_validation": "The feature is high-risk, so any model pick needs validation first",
    "regulations_unavailable_pick_not_cleared": "The legal check was unavailable, so the model pick is not presented as cleared",
    "writing_step_refused_by_content_filter": "The summary was assembled by code because a safety filter refused the written one",
    "unverified_date_replaced": "A date the model gave was not in the retrieved law, so it was replaced with a pointer to Article 113",
    "ungrounded_applicability_replaced": "An application date with no support in the retrieved law was removed",
    "non_answer_classification_cleared": "The legal check could not assess this, so no classification was given",
    "article_6_3_citation_added": "A required citation (Article 6(3)) was added by code",
    "recital_only_check": "The legal answer rested on recitals only, which are not binding, so it was rejected",
    "grounding_coverage_check": "The legal answer was not backed by the retrieved passages, so it was rejected",
    "derogation_addressed_check": "The legal answer skipped the exception in Article 6(3), so it was rejected",
    "overall_guardrail_result": "The legal check did not pass its guardrails",
    "nothing_retrieved_marked_ungrounded": "Nothing relevant was retrieved, so the answer is marked ungrounded",
}


def _describe_event(event):
    rule, detail = event.get("rule") or "", event.get("detail") or ""
    found = re.match(r"\d+", detail)
    count = found.group(0) if found else None
    if rule.startswith("source_check:"):
        kind = rule.split(":", 1)[1]
        who = f"{count} competitor claim{'s' if count != '1' else ''}" if count else "Some competitor claims"
        if kind == "page_does_not_support_claim":
            return f"{who} removed because the cited page did not support them"
        if kind == "no_source_note_replaced":
            return f"{who} removed because no source was given"
        return None
    if rule == "token_estimate_held_to_typical_range":
        return "The token estimate was too low for this kind of feature and was raised to the typical range" + (f" ({detail})" if detail else "")
    if rule == "citation_check":
        return _GUARDRAIL_TEXT[rule] + (f" ({detail})" if detail else "")
    return _GUARDRAIL_TEXT.get(rule) or rule.replace("_", " ")


def guardrail_summary(events):
    """How many checks ran and which of them changed or refused something, in plain words, one line each."""
    events = events or []
    lines = []
    for e in events:
        if e.get("fired"):
            text = _describe_event(e)
            if text and text not in lines:
                lines.append(text)
    return {"checked": len(events), "fired": sum(1 for e in events if e.get("fired")), "lines": lines}


def verdict_tiles(regulatory, cards, withheld, model_ran, coverage, competitor_count=0):
    """The three one-line verdicts at the top of the page. Each states only what the section below it shows."""
    if regulatory.get("status") == "checked":
        cites = regulatory.get("citations") or []
        dates = regulatory.get("dates") or []
        legal = {"title": regulatory["short"], "detail": "EU AI Act" + (", " + ", ".join(cites[:3]) if cites else ""),
                 "extra": ("Applies from " + dates[0]) if len(dates) == 1 else ("Dates: " + "; ".join(dates[:2]) if dates else None), "tone": regulatory["tone"]}
    elif regulatory.get("status") == "failed":
        legal = {"title": regulatory["short"], "detail": "Try again before relying on this feature", "extra": None, "tone": "unclear"}
    else:
        legal = {"title": "Not checked", "detail": "No legal check was run for this request", "extra": None, "tone": "unclear"}

    if withheld:
        reco = {"title": "No recommendation", "detail": "The feature is prohibited, so no model is suggested", "extra": None, "tone": "blocked"}
    elif cards and cards[0]["badge"] == "Recommended":
        c = cards[0]
        latency = f"{c['latency_ms']} ms {c['latency_kind']}" if c.get("latency_ms") is not None else "latency n/a"
        reco = {"title": c["model"], "detail": f"{_money(c['monthly_cost_usd'])}/month · {latency}",
                "extra": (f"{len(cards) - 1} other option{'s' if len(cards) != 2 else ''} below" if len(cards) > 1 else None), "tone": "good"}
    elif cards:
        reco = {"title": "No option meets every target", "detail": "Closest: " + cards[0]["model"], "extra": (cards[0]["misses"] or [None])[0], "tone": "caution"}
    elif model_ran:
        reco = {"title": "No options to show", "detail": "The comparison produced nothing for these targets", "extra": None, "tone": "caution"}
    else:
        reco = {"title": "Not compared", "detail": "Give a target (latency or budget) to compare models", "extra": None, "tone": "unclear"}

    if coverage is None:
        market = {"title": "Not researched", "detail": "No competitor research was run for this request", "extra": None, "tone": "unclear"}
    else:
        market = {"title": f"{competitor_count} competitor{'s' if competitor_count != 1 else ''} researched" if competitor_count else "No competitors found",
                  "detail": f"{coverage['verified']} of {coverage['total']} claims verified",
                  "extra": "No claim could be backed by its source" if coverage["total"] and not coverage["verified"] else None,
                  "tone": "good" if coverage["total"] and coverage["verified"] == coverage["total"] else "caution"}
    return [dict(key="legal", heading="LEGAL", **legal), dict(key="recommended", heading="RECOMMENDED", **reco), dict(key="market", heading="MARKET", **market)]


def build_ui_result(outcome, session_id, guardrail_events=None):
    """The one document the page renders for a finished request."""
    outcome = outcome or {}
    decision = outcome.get("decision") or {}
    state = outcome.get("session_state") or {}
    results = state.get("results") or outcome.get("results") or {}
    facts = outcome.get("facts") or {}
    regulatory = regulatory_block(results)
    withheld = regulatory.get("classification") == "prohibited"

    model_output = facts.get("model_stack_analysis") if isinstance(facts.get("model_stack_analysis"), dict) and facts["model_stack_analysis"].get("scenarios") else _ok_output(results, "model_stack_analysis")
    constraints = state.get("constraints") or decision.get("constraints") or {}
    accept_ok, accept_reason = can_accept(state)
    cards = [] if withheld else build_cards(model_output, constraints)
    competitors = competitors_block(results)
    coverage = competitor_coverage(competitors)

    return {
        "session_id": session_id,
        "request_type": decision.get("request_type"),
        "agents": decision.get("agents") or [],
        "summary": outcome.get("reply"),
        "feature": state.get("feature_description") or decision.get("feature_description"),
        "chips": chips(constraints, model_output),
        "verdicts": verdict_tiles(regulatory, cards, withheld, "model_stack_analysis" in results, coverage, len(competitors or [])),
        "regulatory": regulatory,
        "withheld": withheld,
        "cards": cards,
        "overview": (model_output or {}).get("overview"),
        "reading": reading_block(model_output),
        "volume_assumption": (model_output or {}).get("volume_assumption"),
        "caveats": (model_output or {}).get("caveats") or [],
        "accuracy_note": (model_output or {}).get("accuracy_note"),
        "attribution": (model_output or {}).get("attribution"),
        "competitors": competitors,
        "competitor_coverage": coverage,
        "guardrails": guardrail_summary(guardrail_events),
        "can_accept": bool(accept_ok and not withheld),
        "cannot_accept_reason": None if accept_ok else accept_reason,
        "accepted": state.get("accepted") or [],
    }
