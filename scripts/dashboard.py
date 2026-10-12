"""
What the Dashboard (the overview page after sign-in) shows, built from the company's saved recommendations and the
user's own run records. Every figure is counted or measured from saved data; nothing is estimated
and no baseline is assumed. Sample data (records flagged `sample`) is counted and shown like any other but carries
its flag so the page can label it.

  Evaluations saved         how many recommendations the company has approved
  Awaiting feedback         approved recommendations with a follow-up date that has passed and no report yet
  Decision Quality          the share of shipped features whose reported results met the PM's OWN targets
                            (learning.decision_quality); empty until a feature has been reported
  Median time to a result   the median time from sending a request to getting the options, over this user's runs
"""
from collections import Counter
from datetime import datetime, timezone
from statistics import median

from learning import _targets_met, decision_quality, latest_observations
from recommendation_store import RECORD_TYPE


def _parse(text):
    return datetime.fromisoformat(text) if text else None


def _short(text, limit=90):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _latency_kind(forecast):
    means = str((forecast or {}).get("latency_means") or "").lower()
    return "to the first words" if ("first answer" in means or "reply starts" in means) else "for the full reply"


def _met_text(met):
    """'Met your targets' / 'Missed a target' / None, from {'latency': True/False/None, ...}."""
    checked = [v for v in met.values() if v is not None]
    if not checked:
        return None
    return "met" if all(checked) else "missed"


def evaluation_row(record, now):
    chosen = record.get("chosen") or {}
    forecast = chosen.get("forecast") or {}
    usable = [o for o in record.get("outcomes", []) if o.get("confirmed_by_pm") and not o.get("model_differs")]
    latest = max(usable, key=lambda o: o.get("reported_at", "")) if usable else None
    pending = [f for f in record.get("follow_ups", []) if f.get("status") == "pending"]
    due_now = [f for f in pending if _parse(f["due"]) <= now]
    status = "reported" if latest else ("feedback_due" if due_now else "awaiting")
    next_up = (due_now or pending or [None])[0]
    row = {
        "id": record.get("id"),
        "feature": _short((record.get("feature") or {}).get("description") or (record.get("feature") or {}).get("source_text")),
        "model": chosen.get("label"),
        "forecast": {"yearly_cost_usd": forecast.get("yearly_cost_usd"), "latency_ms": forecast.get("latency_ms"), "latency_kind": _latency_kind(forecast)},
        "accepted_at": record.get("accepted_at"),
        "accepted_misses": chosen.get("accepted_misses") or [],
        "legal": (record.get("legal") or {}).get("classification"),
        "status": status,
        "next_follow_up": {"day": next_up["day"], "due": next_up["due"]} if next_up else None,
        "sample": bool(record.get("sample")),
        "outcome": None,
    }
    if latest:
        fields = latest.get("fields") or {}
        met = _targets_met(record, latest)
        row["outcome"] = {
            "reported_at": latest.get("reported_at"),
            "latency_ms": (fields.get("latency") or {}).get("value_ms"),
            "yearly_cost_usd": (fields.get("cost") or {}).get("yearly_usd"),
            "verdict": _met_text(met),
            "targets": {k: ("met" if v else "missed") for k, v in met.items() if v is not None},
        }
    return row


def build_dashboard(records, run_records, user_id, company_id, now=None):
    """The document the Dashboard page renders. `records` are all accepted recommendations (any company; the
    company's are picked here); `run_records` are all run records (this user's are picked here)."""
    now = now or datetime.now(timezone.utc)
    mine = sorted((r for r in records if r.get("type") == RECORD_TYPE and r.get("company_id") == company_id),
                  key=lambda r: r.get("accepted_at", ""), reverse=True)
    rows = [evaluation_row(r, now) for r in mine]

    quality = decision_quality(mine, company_id)
    durations = [r["duration_seconds"] for r in run_records
                 if r.get("user_id") == user_id and r.get("request_type") == "new_evaluation" and not r.get("error")
                 and isinstance(r.get("duration_seconds"), (int, float)) and r["duration_seconds"] > 0
                 # a full record must show at least one agent that worked; the light listing (no agent outputs) relies on 'no error'
                 and ("agent_results" not in r or any((v or {}).get("status") == "ok" for v in (r.get("agent_results") or {}).values()))]
    top = Counter(r["model"] for r in rows if r["model"]).most_common(3)

    return {
        "stats": {
            "evaluations_saved": len(rows),
            "awaiting_feedback": sum(1 for r in rows if r["status"] == "feedback_due"),
            "decision_quality": {"rate": quality["rate"], "met_all": quality["met_all"], "comparable": quality["comparable"]},
            "median_seconds_to_result": round(median(durations), 1) if durations else None,
            "runs_measured": len(durations),
        },
        "evaluations": rows,
        "action_needed": next((r for r in rows if r["status"] == "feedback_due"), None),
        "top_models": [{"model": m, "count": n} for m, n in top],
        "has_sample_data": any(r["sample"] for r in rows),
    }
