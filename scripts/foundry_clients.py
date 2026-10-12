"""
One shared sign-in and the two model clients every Orchestrator call needs.

Each sub-agent script used to create its own browser sign-in per call. Fine
when run by hand, but the Orchestrator makes several calls at once (three
sub-agents in parallel, plus its own parse and writing steps), which would
open several sign-in windows at the same moment. So: sign in once here, warm
the tokens up before any threads start, and hand the same clients to everyone.

Two clients because Azure exposes two surfaces on the same resource:
- the project's Responses API, used to call registered agents by name+version
- the classic Azure OpenAI endpoint, used for plain model calls (the
  Orchestrator's parse and writing steps, query rewriting, embeddings)
"""
import json
import os
import threading
import time

import openai
from azure.ai.projects import AIProjectClient
from azure.identity import ClientSecretCredential, InteractiveBrowserCredential
from dotenv import load_dotenv

import runtime_paths
from ingest_to_azure_search import get_openai_client as _classic_openai_client

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
# Pinning the tenant stops the browser sign-in picking up a cached account from
# another tenant and failing with a tenant-mismatch error. Set AZURE_TENANT_ID in scripts/.env.
TENANT_ID = os.environ.get("AZURE_TENANT_ID")
_SCOPES = ("https://ai.azure.com/.default", "https://cognitiveservices.azure.com/.default")


def build_credential():
    """Who the code signs in to Azure as. On a laptop: the PM's own browser sign-in. On a server there is no
    browser, so an app identity (a service principal) is used when AZURE_TENANT_ID, AZURE_CLIENT_ID and
    AZURE_CLIENT_SECRET are all set. On Vercel without them it stops with a clear message, because the browser
    sign-in would otherwise wait forever for a window nobody can see."""
    tenant, client_id, secret = (os.environ.get(k) for k in ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"))
    if tenant and client_id and secret:
        return ClientSecretCredential(tenant_id=tenant, client_id=client_id, client_secret=secret)
    if runtime_paths.is_hosted():
        raise RuntimeError("AZURE_TENANT_ID, AZURE_CLIENT_ID and AZURE_CLIENT_SECRET must be set when the app runs hosted; there is no browser to sign in with.")
    return InteractiveBrowserCredential(tenant_id=TENANT_ID)


class Clients:
    def __init__(self):
        self.credential = build_credential()
        # Fetch a token for each scope up front, one after the other. If the
        # first calls happened inside parallel threads, each could try to open
        # its own sign-in window.
        for scope in _SCOPES:
            self.credential.get_token(scope)
        self.agent_client = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=self.credential).get_openai_client()
        self.model_client = _classic_openai_client(self.credential)


RATE_LIMIT_ATTEMPTS = 4


def with_rate_limit_retry(fn, *args):
    """
    The gpt-4o deployment has a per-minute token quota and each call here
    sends ~40 retrieved chunks, so back-to-back calls can hit a 429 (one did
    in testing and crashed the whole case). The SDK's own retries are too
    short for a per-minute window, so wait here - using the server's
    Retry-After if it sends one, otherwise 30s, then 60s, then 60s.
    Running sub-agents in parallel makes this more likely, not less.
    """
    for attempt in range(1, RATE_LIMIT_ATTEMPTS + 1):
        try:
            return fn(*args)
        except openai.RateLimitError as e:
            if attempt == RATE_LIMIT_ATTEMPTS:
                raise
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after"))
            except (AttributeError, TypeError, ValueError):
                pass
            wait = min(retry_after if retry_after else 30 * min(attempt, 2), 65)
            print(f"  (rate limited, waiting {wait:.0f}s before retry {attempt + 1}/{RATE_LIMIT_ATTEMPTS})")
            time.sleep(wait)


AGENT_JSON_ATTEMPTS = 2


def _strip_code_fence(text):
    # Models wrap JSON in ```json ... ``` fences even when told not to.
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def call_agent_for_json(label, make_call, max_attempts=AGENT_JSON_ATTEMPTS):
    """
    Calls an agent and returns its reply parsed as JSON. `make_call` returns the
    response object (it should already include the rate-limit retry).

    The two search-based agents occasionally return malformed JSON on an input
    that worked a minute earlier (an unescaped character mid-string). That is
    random, not a prompt bug, so the same call is repeated once before giving up.
    The cost is one extra agent call, and only on a failure. The failed reply is
    saved per agent (so parallel calls don't overwrite each other) for diagnosis.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        raw_text = _strip_code_fence(make_call().output_text)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            last_error = e
            debug_path = runtime_paths.writable_path(f"last_failure_raw_{label}.txt")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(raw_text)
            print(f"  ({label}: reply was not valid JSON on attempt {attempt}: {e}; "
                  f"{'retrying' if attempt < max_attempts else 'giving up'})")
    raise RuntimeError(f"{label} reply was not valid JSON after {max_attempts} attempts ({last_error}). "
                       f"Last reply written to {debug_path}") from last_error


# --------------------------------------------------------------------------
# A token budget kept in our own code
# --------------------------------------------------------------------------

class TokenBudget:
    """
    Keeps this process under a deployment's tokens-per-minute limit by making a call wait
    until it fits, instead of sending it and getting a 429.

    Why: the gpt-4o deployment allows 50,000 tokens a minute (read from Azure's own
    rate-limit headers), and one Orchestrator request uses about 39,000, so a second request
    within the same minute can never fit. Waiting for the retry after a 429 got the Regulations
    call through only some of the time. With a budget, a busy minute means a later answer,
    not a missing legal check.

    Each call is counted from the moment it starts, using an estimate, and corrected to the
    real token count when it finishes. A call counts for 60 seconds. If one call alone is
    bigger than the budget it is let through when nothing else is in flight. The clock and
    sleep are injectable so the tests don't have to wait. It only knows about this process:
    two separate processes sharing one deployment still rely on the 429 retry.
    """

    def __init__(self, tokens_per_minute, margin=0.9, window_seconds=60.0, clock=time.time, sleep=time.sleep):
        self.capacity = tokens_per_minute * margin
        self.window = window_seconds
        self.clock, self.sleep = clock, sleep
        self._entries = []  # [start_time, tokens]
        self._lock = threading.Lock()
        self.total_waited = 0.0

    def _used(self, now):
        self._entries = [e for e in self._entries if now - e[0] < self.window]
        return sum(e[1] for e in self._entries)

    def acquire(self, estimated_tokens):
        """Blocks until the estimate fits, then reserves it. Returns the reservation to pass to
        `settle`."""
        waited = 0.0
        while True:
            with self._lock:
                now = self.clock()
                used = self._used(now)
                if used + estimated_tokens <= self.capacity or not self._entries:
                    entry = [now, estimated_tokens]
                    self._entries.append(entry)
                    self.total_waited += waited
                    return entry
                # Wait until enough of the oldest entries have aged out of the window.
                freed, wake = used, now + 1.0
                for start, tokens in sorted(self._entries):
                    freed -= tokens
                    wake = start + self.window
                    if freed + estimated_tokens <= self.capacity:
                        break
                delay = max(wake - now, 0.05)
            if waited == 0.0:
                print(f"  (waiting for token budget: this call needs ~{estimated_tokens:,} tokens and ~{used:,.0f} of {self.capacity:,.0f} are in use)")
            self.sleep(delay)
            waited += delay

    def settle(self, entry, actual_tokens):
        """Replaces the estimate with what the call really used, if known."""
        if actual_tokens:
            with self._lock:
                entry[1] = actual_tokens


GPT4O_TOKENS_PER_MINUTE = int(os.environ.get("GPT4O_TOKENS_PER_MINUTE", "50000"))
GPT4O_BUDGET = TokenBudget(GPT4O_TOKENS_PER_MINUTE)
EXPECTED_OUTPUT_TOKENS = 1500


def estimate_tokens(payload, expected_output=EXPECTED_OUTPUT_TOKENS, overhead=0):
    """Rough token count for a request: characters / 4 for the text we send, plus the reply we
    expect, plus `overhead` for instructions the service adds (a registered agent's system prompt)."""
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return len(text) // 4 + expected_output + overhead


def _actual_tokens(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    total = getattr(usage, "total_tokens", None)
    if total:
        return total
    parts = [getattr(usage, n, None) or 0 for n in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens")]
    return sum(parts) or None


# What each kind of call really used, against what we estimated. The first live test over-estimated the
# legal check by about 28%, which made it wait a full minute for its own sibling calls to age out of the
# window. Each kind of call now learns its own correction from its real token counts. Clamped so one odd
# call can't swing it far, and it starts at 1.0 so a first estimate is cautious, not optimistic.
_CORRECTIONS = {}
_CORRECTION_LOCK = threading.Lock()
CORRECTION_RANGE = (0.5, 1.3)
CORRECTION_WEIGHT = 0.5  # how much the latest call counts against the earlier ones


def correction_for(key):
    with _CORRECTION_LOCK:
        return _CORRECTIONS.get(key, 1.0)


def learn_correction(key, estimated, actual):
    if not key or not estimated or not actual:
        return
    low, high = CORRECTION_RANGE
    with _CORRECTION_LOCK:
        previous = _CORRECTIONS.get(key, 1.0)
        observed = (actual / estimated) * previous  # what the correction should have been
        _CORRECTIONS[key] = min(max((1 - CORRECTION_WEIGHT) * previous + CORRECTION_WEIGHT * observed, low), high)


def budgeted_call(make_call, payload, budget=None, expected_output=EXPECTED_OUTPUT_TOKENS, overhead=0, key=None):
    """Runs a gpt-4o call once it fits the token budget, with the 429 retry as a second line of
    defence. `make_call` takes no arguments and returns the response."""
    budget = budget or GPT4O_BUDGET
    raw_estimate = estimate_tokens(payload, expected_output, overhead)
    estimate = max(int(raw_estimate * correction_for(key)), 1)
    entry = budget.acquire(estimate)
    try:
        response = with_rate_limit_retry(make_call)
    except Exception:
        budget.settle(entry, 0)  # leave the estimate: a failed call may still have counted
        raise
    actual = _actual_tokens(response)
    budget.settle(entry, actual)
    learn_correction(key, estimate, actual)
    return response
