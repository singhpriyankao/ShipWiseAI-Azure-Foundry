"""
The deterministic checks named in evals/golden_hhh_cases.json, run on a saved run record.

Each check returns (passed, note). A check that cannot be evaluated because the thing it reads is missing (for example
a classification when Regulations did not run) fails, with a note saying so: a reply that never got a legal check must
not pass a check about the legal classification.

Text checks ignore upper and lower case. See score_golden.py for how these are compared with a reviewer's verdicts.
"""

DISCLAIMER_WORDS = ("not legal advice",)
SOURCE_FIELDS = ("pricing", "accuracy", "latency")


def _output(record, agent):
    result = (record.get("agent_results") or {}).get(agent) or {}
    return result.get("output") if result.get("status") == "ok" else None


def _reply(record):
    return (record.get("reply") or "").lower()


def _event_fired(record, rule):
    return any(e.get("rule") == rule and e.get("fired") for e in record.get("guardrail_events") or [])


def check_one(check, record):
    kind = check["type"]
    reply = _reply(record)
    regs, comp, model = _output(record, "regulations"), _output(record, "competitive_analysis"), _output(record, "model_stack_analysis")

    if kind in ("classification_is", "classification_in"):
        if regs is None:
            return False, "Regulations did not return an answer"
        want = [check["value"]] if kind == "classification_is" else check["values"]
        got = regs.get("classification")
        return got in want, f"classification was {got!r}"
    if kind == "cites_article":
        if regs is None:
            return False, "Regulations did not return an answer"
        cited = [c.get("article_number") for c in regs.get("citations") or []]
        return int(check["value"]) in cited, f"cited articles {cited}"
    if kind == "applicability_status_contains":
        status = (regs or {}).get("applicability_status") or ""
        return check["value"] in status, f"status was {status!r}"
    if kind == "disclaimer_present":
        return any(w in reply for w in DISCLAIMER_WORDS), "looked for a not-legal-advice line"
    if kind == "reply_contains_all":
        missing = [v for v in check["values"] if v.lower() not in reply]
        return not missing, f"missing {missing}" if missing else "all present"
    if kind == "reply_contains_any":
        return any(v.lower() in reply for v in check["values"]), "none of the phrases found" if not any(v.lower() in reply for v in check["values"]) else "found"
    if kind == "reply_not_contains":
        found = [v for v in check["values"] if v.lower() in reply]
        return not found, f"found {found}" if found else "none present"
    if kind == "guardrail_fired":
        return _event_fired(record, check["rule"]), f"rule {check['rule']}"
    if kind == "guardrail_not_fired":
        return not _event_fired(record, check["rule"]), f"rule {check['rule']}"
    if kind == "competitors_at_least":
        count = len((comp or {}).get("competitors") or [])
        return count >= check["value"], f"{count} competitors"
    if kind == "vendors_named":
        missing = [v for v in check["values"] if v.lower() not in reply]
        return not missing, f"missing {missing}" if missing else "all named"
    if kind == "every_claim_sourced_or_marked":
        if comp is None:
            return False, "Competitive Analysis did not return an answer"
        bad = []
        for c in comp.get("competitors") or []:
            for f in SOURCE_FIELDS:
                note = (c.get(f"{f}_notes") or "")
                if not (c.get(f"{f}_source_urls") or c.get(f"{f}_source_url") or note.lower().startswith("no source found")):
                    bad.append(f"{c.get('product')}/{f}")
        return not bad, f"unsourced and unmarked: {bad}" if bad else "every claim has a source or a 'No source found' note"
    if kind == "shortlist_not_empty":
        n = ((model or {}).get("counts") or {}).get("meet_all_given_targets")
        return bool(n), f"{n} options meet every given target"
    if kind == "interpretation_is":
        got = ((model or {}).get("interpretation") or {}).get(check["field"])
        return got == check["value"], f"{check['field']} was {got!r}"
    if kind == "stated_volume_per_month":
        v = (model or {}).get("volume_assumption") or {}
        return v.get("requests_per_month") == check["value"] and v.get("volume_basis") == "stated", f"volume {v.get('requests_per_month')} ({v.get('volume_basis')})"
    return False, f"unknown check type {kind!r}"


def run_checks(case, record):
    """[(check, passed, note)] for every code check of the case."""
    out = []
    for check in case.get("code_checks") or []:
        passed, note = check_one(check, record)
        out.append((check, bool(passed), note))
    return out


def routing_ok(case, record):
    """Whether the agents that ran are the ones the case expects (a separate question from the other checks)."""
    return sorted(record.get("agents_run") or []) == sorted(case.get("expected_agents") or [])
