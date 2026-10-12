"""
The record of one request, kept for post-launch review.

One document per PM message: the message, what the Orchestrator decided, each sub-agent's full output, every
guardrail event, the final reply, and empty fields for a reviewer to fill in. Foundry traces cut off long answers
and don't see our guardrails (they run in our code after the agent call), so this is the complete copy.

Guardrails report what they did by calling `event(...)`. That call does nothing unless a request is being
recorded, so the guardrail code and its tests are unchanged when no one is recording. The sub-agents run in
worker threads; `bind` hands each one the recorder of the thread that submitted it.

Stores: LocalRunStore (a JSON file, git-ignored) and CosmosRunStore (the existing `recommendations` container,
as documents of type "run_record"; the account's 1,000 RU/s free-tier limit is already used by that container,
so a new one would cost money or be refused).
"""
import contextvars
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone

RECORD_TYPE = "run_record"
MAX_RECORD_BYTES = 1_500_000  # Cosmos refuses an item over 2 MB

_current = contextvars.ContextVar("run_log_collector", default=None)


class Collector:
    def __init__(self):
        self.events = []
        self._lock = threading.Lock()
        self.started = time.monotonic()

    def add(self, agent, rule, fired, detail=None):
        with self._lock:
            self.events.append({"agent": agent, "rule": rule, "fired": bool(fired), **({"detail": detail} if detail else {})})


def begin():
    """Start recording this request. Returns the collector; call finish_run() with it when the request ends."""
    collector = Collector()
    _current.set(collector)
    return collector


def end():
    _current.set(None)


def event(agent, rule, fired, detail=None):
    """Record that a guardrail ran: `fired` is True when it changed something or refused something."""
    collector = _current.get()
    if collector is not None:
        collector.add(agent, rule, fired, detail)


def bind(fn, *args, **kwargs):
    """Returns a callable that runs fn(*args) inside a copy of the CALLING thread's context, so a worker thread
    records into the same collector. The copy is taken here, in the submitting thread; copying inside the
    worker would copy the worker's own empty context. Use as pool.submit(run_log.bind(fn, ...))."""
    context = contextvars.copy_context()
    return lambda: context.run(fn, *args, **kwargs)


# --------------------------------------------------------------------------
# Events read from a sub-agent's finished output (no change to that agent's code)
# --------------------------------------------------------------------------

_SOURCE_FIXED = ("no_source_note_replaced", "page_does_not_support_claim")


def events_from_competitive(output):
    """One event per source-check outcome kind, from the `source_checks` the competitive checks leave on each
    competitor (competitive_checks.py)."""
    counts = {}
    for comp in (output or {}).get("competitors") or []:
        for field, status in (comp.get("source_checks") or {}).items():
            kind = status.split(" (")[0]
            counts[kind] = counts.get(kind, 0) + 1
    events = [{"agent": "competitive_analysis", "rule": "source_check:" + kind, "fired": kind in _SOURCE_FIXED, "detail": f"{n} claim(s)"}
              for kind, n in sorted(counts.items())]
    if not events and (output or {}).get("competitors"):
        events.append({"agent": "competitive_analysis", "rule": "source_check", "fired": False, "detail": "no checks recorded"})
    return events


def events_from_regulations(guardrail_status):
    """The Regulations agent's overall guardrail result: passed, insufficient_information or failed_guardrails."""
    return [{"agent": "regulations", "rule": "overall_guardrail_result", "fired": guardrail_status != "passed", "detail": guardrail_status}]


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------

def all_events(collector, results):
    """Every guardrail event for one request: those the guardrail code reported as it ran, plus those read from
    the agents' finished outputs (the competitor source checks and the Regulations overall result)."""
    events = list(collector.events) if collector else []
    for name, result in (results or {}).items():
        if name == "competitive_analysis" and result.get("status") == "ok":
            events += events_from_competitive(result.get("output"))
        if name == "regulations" and result.get("guardrail_status"):
            events += events_from_regulations(result["guardrail_status"])
    return events


REVIEW_TEMPLATE = {
    "reviewer": None,
    "reviewed_at": None,
    "verdict": None,        # "correct" | "incorrect" | "partly_correct" | "cannot_tell"
    "wrong_field": None,    # e.g. "classification", "article_cited", "competitor_claim", "reply_style"
    "notes": None,
}


def build_record(user_message, outcome, collector, user_id, company_id, session_id=None, trace_ids=None, error=None):
    """The document for one request. `outcome` is what handle_message returned (or None if it raised)."""
    outcome = outcome or {}
    decision = outcome.get("decision") or {}
    results = outcome.get("results") or {}
    events = all_events(collector, results)
    record = {
        "id": str(uuid.uuid4()),
        "type": RECORD_TYPE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "company_id": company_id,
        "session_id": session_id,
        "message": user_message,
        "request_type": decision.get("request_type"),
        "agents_run": decision.get("agents"),
        "constraints": decision.get("constraints"),
        "agent_results": {name: {"status": r.get("status"), "guardrail_status": r.get("guardrail_status"),
                                 "output": r.get("output"), "error": r.get("error")} for name, r in results.items()},
        "guardrail_events": events,
        "reply": outcome.get("reply"),
        "duration_seconds": round(time.monotonic() - collector.started, 1) if collector else None,
        "trace_ids": trace_ids or [],
        "error": error,
        "review": dict(REVIEW_TEMPLATE),
    }
    if len(json.dumps(record, default=str).encode("utf-8")) > MAX_RECORD_BYTES:
        record["agent_results"] = {n: {**v, "output": "(left out: record too large)"} for n, v in record["agent_results"].items()}
        record["truncated"] = True
    return json.loads(json.dumps(record, default=str))


# --------------------------------------------------------------------------
# Stores
# --------------------------------------------------------------------------

class LocalRunStore:
    def __init__(self, path):
        self.path = path

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def add(self, record):
        data = self._load()
        data[record["id"]] = record
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return record["id"]

    def get(self, record_id):
        return self._load().get(record_id)

    def list_runs(self):
        return sorted((r for r in self._load().values() if r.get("type") == RECORD_TYPE), key=lambda r: r["created_at"])


class CosmosRunStore:
    def __init__(self, endpoint=None, key=None):
        from azure.cosmos import CosmosClient
        client = CosmosClient(endpoint or os.environ["COSMOS_ENDPOINT"], key or os.environ["COSMOS_KEY"])
        self._container = client.get_database_client("shipwiseai").get_container_client("recommendations")

    def add(self, record):
        self._container.upsert_item(record)
        return record["id"]

    def get(self, record_id):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        try:
            return self._container.read_item(item=record_id, partition_key=record_id)
        except CosmosResourceNotFoundError:
            return None

    def list_runs(self):
        return sorted(self._container.query_items(
            query="SELECT * FROM c WHERE c.type = @t", parameters=[{"name": "@t", "value": RECORD_TYPE}],
            enable_cross_partition_query=True), key=lambda r: r["created_at"])

    def list_run_timings(self, user_id):
        """Only the few fields the Dashboard's time-to-result needs, for one user, so the page doesn't read every
        agent output ever saved."""
        return list(self._container.query_items(
            query="SELECT c.id, c.user_id, c.request_type, c.duration_seconds, c.error FROM c WHERE c.type = @t AND c.user_id = @u",
            parameters=[{"name": "@t", "value": RECORD_TYPE}, {"name": "@u", "value": user_id}],
            enable_cross_partition_query=True))
