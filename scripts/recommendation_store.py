"""
The accept step: when a PM chooses one of the Model Stack options, record exactly what
they were shown and chose, so that what they later report happened can be compared with it. This is the
start of the feedback loop; reporting outcomes and the learning arithmetic come after it.

Plain code, no model. What gets saved is a snapshot, not a pointer: the forecast figures, the usage the
cost assumed, the legal result and the warnings as they were at the moment of choosing, because prices
and benchmarks change and a later comparison has to be against what the PM actually saw.

Where it lives: the Cosmos `recommendations` container (partition key `/id`) that was created for this.
Older documents in it, written by the first Model Stack script, have no `type` field and are ignored.
LocalRecommendationStore is the same interface on a JSON file, for tests and for running without Cosmos.
"""
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone

from model_stack_builder import ordinal

FOLLOW_UP_DAYS = (30, 60, 90)
RECORD_TYPE = "accepted_recommendation"
STATUS_AWAITING = "awaiting_outcome"


# --------------------------------------------------------------------------
# Working out which option the PM meant
# --------------------------------------------------------------------------

_WORD_NUMBERS = {"one": 1, "first": 1, "1st": 1, "two": 2, "second": 2, "2nd": 2, "three": 3, "third": 3, "3rd": 3,
                 "four": 4, "fourth": 4, "4th": 4, "five": 5, "fifth": 5, "5th": 5}
_FILLER = {"option", "options", "number", "no", "the", "one", "accept", "go", "with", "choose", "take", "pick", "ill",
           "i", "we", "will", "lets", "let", "s", "want", "use", "that", "please", "for", "this", "a", "an", "and",
           "model", "setting", "settings", "going", "to", "from", "shortlist", "it", "my", "our", "on", "im", "ll"}


def _tokens(text):
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", str(text).lower())


def resolve_option(chosen, options):
    """Which of the options shown the PM meant. `options` are the scenarios in the order shown (the
    shortlist first). Returns (option, None) or (None, a plain message saying what to clarify). A number
    ("option 2", "the second one") means that position on the shortlist; anything else is matched against
    the option's name, and if it fits more than one (two effort settings of one model) the PM is asked."""
    if not chosen or not str(chosen).strip():
        return None, "Which option do you want to go with? You can give its number on the shortlist or its name."
    shortlist = [o for o in options if o.get("role") == "shortlist"]
    tokens = _tokens(chosen)

    for o in options:  # an internal id, if the PM or the system gives one
        if o.get("id") and str(o["id"]).lower() in str(chosen).lower().split():
            return o, None

    number = None
    for t in tokens:
        if t in _WORD_NUMBERS:
            number = _WORD_NUMBERS[t]
            break
    if number is None:
        m = re.search(r"(?:option|number|no\.?|#)\s*(\d+)\b|^\s*(\d+)\s*$", str(chosen).lower())
        if m:
            number = int(m.group(1) or m.group(2))
    if number is not None and not any(re.fullmatch(r"\d+\.\d+", t) for t in tokens):
        if 1 <= number <= len(shortlist):
            return shortlist[number - 1], None
        return None, f"I only showed {len(shortlist)} option{'s' if len(shortlist) != 1 else ''} on the shortlist, so there is no option {number}."

    wanted = [t for t in tokens if t not in _FILLER]
    if not wanted:
        return None, "Which option do you want to go with? You can give its number on the shortlist or its name."
    matches = [o for o in options if all(w in set(_tokens(o["model_stack"])) for w in wanted)]
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, f"I couldn't match \"{chosen}\" to one of the options shown. Give its number on the shortlist or its name."
    names = "; ".join(o["model_stack"] for o in matches[:4])
    return None, f"\"{chosen}\" fits more than one option: {names}. Which one do you mean?"


# --------------------------------------------------------------------------
# Whether there is something to accept, and the record itself
# --------------------------------------------------------------------------

def _model_stack_output(session_state):
    result = ((session_state or {}).get("results") or {}).get("model_stack_analysis") or {}
    return result.get("output") if result.get("status") == "ok" else None


def _regulations_output(session_state):
    result = ((session_state or {}).get("results") or {}).get("regulations") or {}
    return result.get("output") if result.get("status") == "ok" else None


def can_accept(session_state):
    """(True, None) or (False, a plain reason)."""
    output = _model_stack_output(session_state)
    if not output or not output.get("scenarios"):
        return False, "There is nothing to accept yet: no model options have been produced in this conversation."
    regs = _regulations_output(session_state)
    if regs and regs.get("classification") == "prohibited":
        return False, ("This feature was found prohibited under the EU AI Act, so I won't record a model choice for it. "
                       "Review the legal finding with Legal before going further.")
    return True, None


def _legal_snapshot(session_state):
    result = ((session_state or {}).get("results") or {}).get("regulations")
    regs = _regulations_output(session_state)
    if regs:
        return {"status": "checked", "classification": regs.get("classification"),
                "applicability_status": regs.get("applicability_status"),
                "derogation_assessment": regs.get("derogation_assessment")}
    if result:
        return {"status": "check_failed", "classification": None}
    return {"status": "not_run", "classification": None}


def build_acceptance_record(session_state, option, user_id, company_id, now=None):
    now = now or datetime.now(timezone.utc)
    output = _model_stack_output(session_state)
    interpretation = output.get("interpretation") or {}
    misses = sorted(k for k, v in (option.get("verdicts") or {}).items() if v == "misses")
    return {
        "id": str(uuid.uuid4()),
        "type": RECORD_TYPE,
        "user_id": user_id,
        "company_id": company_id,
        "accepted_at": now.isoformat(),
        "status": STATUS_AWAITING,
        "feature": {
            "source_text": (session_state or {}).get("source_text"),
            "description": (session_state or {}).get("feature_description"),
            "category": interpretation.get("feature_category", "other"),
            "reply_style": interpretation.get("reply_style"),
        },
        "constraints_given": (session_state or {}).get("constraints"),
        "chosen": {
            "option_id": option["id"],
            "label": option["model_stack"],
            "role": option.get("role"),
            "accepted_misses": misses,
            "forecast": {
                "yearly_cost_usd": option.get("yearly_cost_usd"),
                "latency_ms": option.get("latency_ms"),
                "latency_means": option.get("latency_means"),
                "first_answer_ms": option.get("first_answer_ms"),
                "full_response_ms": option.get("full_response_ms"),
                "quality_percentile": option.get("quality_percentile"),
                "input_price_per_1m_tokens": option.get("input_price_per_1m_tokens"),
                "output_price_per_1m_tokens": option.get("output_price_per_1m_tokens"),
            },
            "verdicts": option.get("verdicts"),
            # The benchmark forecast before any learned correction, so a later report is compared with it and
            # corrections do not compound; and the learned sentences the PM was shown.
            "forecast_raw": option.get("raw_forecast"),
            "learned_shown": option.get("learned") or [],
        },
        "usage_assumption": output.get("volume_assumption"),
        "caveats": output.get("caveats") or [],
        "legal": _legal_snapshot(session_state),
        "alternatives_shown": [o["model_stack"] for o in output["scenarios"] if o["id"] != option["id"]],
        "sources": {"prices": "the vendors' own pricing pages", "latency_and_quality": "Artificial Analysis (artificialanalysis.ai)"},
        "follow_ups": [{"day": d, "due": (now + timedelta(days=d)).isoformat(), "status": "pending"} for d in FOLLOW_UP_DAYS],
        "outcomes": [],
    }


def _money(dollars):
    if dollars is None:
        return "an unknown amount"
    if dollars >= 1000:
        return f"${dollars / 1000:.1f}K".replace(".0K", "K")
    return f"${dollars:.0f}"


def confirmation_message(record):
    chosen, f = record["chosen"], record["chosen"]["forecast"]
    lines = [f"Recorded: you chose {chosen['label']}."]
    forecast = []
    if f.get("yearly_cost_usd") is not None:
        forecast.append(f"about {_money(f['yearly_cost_usd'])} a year")
    if f.get("latency_ms") is not None:
        forecast.append(f"{f['latency_ms']}ms ({f.get('latency_means') or 'latency'})")
    if f.get("quality_percentile") is not None:
        forecast.append(f"a general quality ranking at the {ordinal(f['quality_percentile'])} percentile")
    if forecast:
        lines.append("What was forecast: " + ", ".join(forecast) + ".")
    if chosen["accepted_misses"]:
        lines.append("You accepted this knowing it misses your " + " and ".join(chosen["accepted_misses"]) + " target.")
    if record["legal"]["status"] != "checked":
        lines.append("Note: the legal check " + ("did not complete" if record["legal"]["status"] == "check_failed" else "was not run") + " for this feature, so no legal result is recorded with it.")
    elif record["legal"]["classification"]:
        lines.append(f"The legal result at the time was {record['legal']['classification']}.")
    due = ", ".join(f"{fu['day']} days ({fu['due'][:10]})" for fu in record["follow_ups"])
    lines.append(f"Follow-ups are due at {due}.")
    lines.append("Once it has launched, report what actually happened and the saved forecast will be compared with it.")
    return " ".join(lines)


def accept_option(session_state, chosen_text, user_id, company_id, store, now=None):
    """The whole step. Returns {"ok", "message", "record", "session_update"}. Never raises on bad input."""
    ok, reason = can_accept(session_state)
    if not ok:
        return {"ok": False, "message": reason, "record": None, "session_update": {}}
    output = _model_stack_output(session_state)
    option, problem = resolve_option(chosen_text, output["scenarios"])
    if problem:
        return {"ok": False, "message": problem, "record": None, "session_update": {}}

    for done in (session_state or {}).get("accepted", []):
        if done.get("option_id") == option["id"]:
            record = store.get(done["record_id"])
            return {"ok": True, "message": f"You already recorded {option['model_stack']} for this feature, so nothing changed.",
                    "record": record, "session_update": {}}

    record = build_acceptance_record(session_state, option, user_id, company_id, now)
    store.add(record)
    accepted = list((session_state or {}).get("accepted", [])) + [{"option_id": option["id"], "record_id": record["id"]}]
    return {"ok": True, "message": confirmation_message(record), "record": record, "session_update": {"accepted": accepted}}


# --------------------------------------------------------------------------
# Stores
# --------------------------------------------------------------------------

def _is_due(follow_up, now):
    return follow_up["status"] == "pending" and datetime.fromisoformat(follow_up["due"]) <= now


class _StoreQueries:
    """Questions both stores answer the same way, on top of get/list_all."""

    def for_user(self, user_id):
        return [r for r in self.list_all() if r.get("user_id") == user_id]

    def due_follow_ups(self, now=None):
        """(record, follow_up) for every follow-up that has fallen due on a recommendation still waiting."""
        now = now or datetime.now(timezone.utc)
        due = []
        for record in self.list_all():
            if record.get("status") != STATUS_AWAITING:
                continue
            due += [(record, fu) for fu in record.get("follow_ups", []) if _is_due(fu, now)]
        return due


class LocalRecommendationStore(_StoreQueries):
    def __init__(self, path):
        self.path = path

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, data):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def add(self, record):
        data = self._load()
        data[record["id"]] = record
        self._save(data)
        return record["id"]

    def update(self, record):
        self.add(record)

    def get(self, record_id):
        return self._load().get(record_id)

    def list_all(self):
        return [r for r in self._load().values() if r.get("type") == RECORD_TYPE]


class CosmosRecommendationStore(_StoreQueries):
    def __init__(self, endpoint=None, key=None):
        from azure.cosmos import CosmosClient
        client = CosmosClient(endpoint or os.environ["COSMOS_ENDPOINT"], key or os.environ["COSMOS_KEY"])
        self._container = client.get_database_client("shipwiseai").get_container_client("recommendations")

    def add(self, record):
        self._container.upsert_item(record)
        return record["id"]

    def update(self, record):
        self._container.upsert_item(record)

    def get(self, record_id):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        try:
            return self._container.read_item(item=record_id, partition_key=record_id)
        except CosmosResourceNotFoundError:
            return None

    def list_all(self):
        return list(self._container.query_items(
            query="SELECT * FROM c WHERE c.type = @t", parameters=[{"name": "@t", "value": RECORD_TYPE}],
            enable_cross_partition_query=True))
