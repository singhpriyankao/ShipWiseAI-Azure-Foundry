"""
ShipWise AI web server. This first slice does one thing: it knows who is signed in.

Sign-in itself happens in the browser, against Supabase Auth (email and password). The
browser holds a signed token for the user. Every call to this server must carry that
token, and the server asks Supabase to confirm it before doing anything (`current_user`)
instead of checking the signature itself, so there is no key or algorithm for us to get
wrong. It then reads the user's profile (name, company) from Supabase using the user's
own token, so Supabase's row-level security decides what is visible, not our code.

Only the public (publishable) key is used, here and in the browser. The Supabase secret
key is never needed and must not be put in .env.

It also runs the Orchestrator (POST /api/evaluate) and records every request (run_log.py).

Run (from this folder):  python -m uvicorn server:app --port 8000
Then open http://localhost:8000
On Vercel this file is the entrypoint (app/server.py); see ../vercel.json.
"""
import os
import sys
import threading
import time

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, SCRIPTS)
load_dotenv(os.path.join(SCRIPTS, ".env"))

SUPABASE_URL = os.environ["NEXT_PUBLIC_SUPABASE_URL"].strip().rstrip("/")
SUPABASE_KEY = os.environ["NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY"].strip()
SUPABASE_TIMEOUT_SECONDS = 15

app = FastAPI(title="ShipWise AI")


def current_user(authorization: str = Header(default="")):
    """The signed-in user and their profile, or a 401. Used as a dependency by every
    endpoint that needs to know who is asking."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Sign in first.")
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {token}"}
    try:
        who = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers, timeout=SUPABASE_TIMEOUT_SECONDS)
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="The sign-in service could not be reached. Try again.")
    if who.status_code != 200:
        raise HTTPException(status_code=401, detail="Your session has expired. Sign in again.")
    user = who.json()

    try:
        rows = requests.get(f"{SUPABASE_URL}/rest/v1/profiles", headers=headers, timeout=SUPABASE_TIMEOUT_SECONDS,
                            params={"select": "full_name,company,company_id", "id": f"eq.{user['id']}"})
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="The sign-in service could not be reached. Try again.")
    profile = rows.json()[0] if rows.status_code == 200 and rows.json() else None
    if profile is None:
        raise HTTPException(status_code=403, detail="Your account has no profile. Contact support.")
    return {"user_id": user["id"], "email": user.get("email"), **profile}


@app.get("/config")
def public_config():
    """What the browser needs to talk to Supabase. Both values are public by design."""
    return {"supabase_url": SUPABASE_URL, "supabase_key": SUPABASE_KEY}


@app.get("/api/me")
def me(user=Depends(current_user)):
    return user


@app.get("/api/health")
def health(deep: bool = False):
    """Is the deployment alive; with ?deep=1, can it load the Orchestrator and its packages (the first thing
    to check on a new platform). Reveals no secrets, only whether each setting is present."""
    import platform
    out = {"ok": True, "python": platform.python_version(), "hosted": bool(os.environ.get("VERCEL")),
           "settings_present": {k: bool(os.environ.get(k)) for k in (
               "FOUNDRY_PROJECT_ENDPOINT", "COSMOS_ENDPOINT", "COSMOS_KEY", "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY",
               "REGULATIONS_AGENT_VERSION", "COMPETITIVE_ANALYSIS_AGENT_VERSION", "ARTIFICIAL_ANALYSIS_API_KEY",
               "AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET")}}
    # The starting copies of the price and benchmark caches should ship with the deployment (runtime_paths.py).
    out["bundled_caches_present"] = {n: os.path.exists(os.path.join(SCRIPTS, n)) for n in ("price_cache.json", "benchmarks_cache.json")}
    if deep:
        try:
            import orchestrator  # noqa: F401
            import run_log  # noqa: F401
            out["orchestrator_import"] = "ok"
        except Exception as e:
            out["ok"] = False
            out["orchestrator_import"] = f"{type(e).__name__}: {e}"
    return out


_clients = None
_clients_lock = threading.Lock()


def get_clients():
    """One set of Azure clients per server instance, made on first use, so a request doesn't pay for a new
    sign-in each time."""
    global _clients
    with _clients_lock:
        if _clients is None:
            from foundry_clients import Clients
            _clients = Clients()
        return _clients


class EvaluateRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.post("/api/evaluate")
def evaluate(body: EvaluateRequest, user=Depends(current_user)):
    """Runs the Orchestrator on one PM message and records the request. This is a single-turn call for now:
    follow-up messages that depend on an earlier turn (adding targets, accepting an option) need the session
    saved between calls, which is not built here yet."""
    import run_log
    from orchestrator import handle_message
    from recommendation_store import CosmosRecommendationStore

    collector = run_log.begin()
    outcome, failure = None, None
    try:
        outcome = handle_message(body.message, None, get_clients(), recommendation_store=CosmosRecommendationStore(),
                                 user_id=user["user_id"], company_id=user["company_id"])
    except Exception as e:
        failure = f"{type(e).__name__}: {e}"
    finally:
        # A failed request is recorded too. If the database can't be reached the reply is still returned.
        record_id, record_problem = None, None
        try:
            record = run_log.build_record(body.message, outcome, collector, user["user_id"], user["company_id"], error=failure)
            record_id = run_log.CosmosRunStore().add(record)
        except Exception as e:
            record_problem = f"{type(e).__name__}: {e}"
        run_log.end()
    if failure:
        raise HTTPException(status_code=502, detail={"message": "The request could not be completed.", "run_id": record_id, "record_problem": record_problem})
    decision = outcome["decision"]
    return {"reply": outcome["reply"], "request_type": decision["request_type"], "agents": decision["agents"],
            "run_id": record_id, "record_problem": record_problem,
            "duration_seconds": round(time.monotonic() - collector.started, 1)}


class StreamRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)


def _sse(event):
    import json
    return "data: " + json.dumps(event, default=str) + "\n\n"


@app.post("/api/evaluate/stream")
def evaluate_stream(body: StreamRequest, user=Depends(current_user)):
    """Runs the Orchestrator and sends progress as it happens (server-sent events), then the finished result.
    Events: reading, note, plan, agent_started, agent_finished, writing, then result (or error). A session id
    carries a conversation across messages; the session belongs to the user who made it. The request is recorded
    whether or not it works (run_log.py)."""
    import queue
    import uuid

    events = queue.Queue()
    session_id = body.session_id or str(uuid.uuid4())

    def worker():
        import run_log
        from orchestrator import handle_message
        from recommendation_store import CosmosRecommendationStore
        from session_store import SharedContainerSessionStore
        from ui_payload import build_ui_result

        collector = run_log.begin()
        outcome, ui, failure = None, None, None
        try:
            sessions = SharedContainerSessionStore()
            state = sessions.get(session_id, user["user_id"]) if body.session_id else None
            outcome = handle_message(body.message, state, get_clients(), on_event=lambda text: events.put({"type": "note", "message": text}),
                                     recommendation_store=CosmosRecommendationStore(), user_id=user["user_id"], company_id=user["company_id"],
                                     progress=events.put)
            if outcome.get("session_state") is not None:
                sessions.put(session_id, user["user_id"], outcome["session_state"])
            ui = build_ui_result(outcome, session_id, guardrail_events=run_log.all_events(collector, outcome.get("results")))
            if ui["regulatory"].get("status") == "checked":
                # The exact wording of what the legal answer cites, read from the index; left out if it can't be fetched.
                try:
                    from act_text import quote_citations
                    regs = ((outcome.get("session_state") or {}).get("results") or {}).get("regulations") or {}
                    ui["regulatory"]["quotes"] = quote_citations((regs.get("output") or {}).get("citations"))
                except Exception:
                    pass
        except Exception as e:
            failure = f"{type(e).__name__}: {e}"
        record_id = None
        try:
            record = run_log.build_record(body.message, outcome, collector, user["user_id"], user["company_id"], session_id=session_id, error=failure)
            record_id = run_log.CosmosRunStore().add(record)
        except Exception:
            pass  # the reply is still returned; the record is lost, not the answer
        run_log.end()
        if failure:
            events.put({"type": "error", "message": "The request could not be completed. Try again in a moment.", "run_id": record_id})
        else:
            events.put({"type": "result", "run_id": record_id, **ui})
        events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            try:
                item = events.get(timeout=10)
            except queue.Empty:
                yield ": keepalive\n\n"  # keeps a quiet connection from being closed on the way
                continue
            if item is None:
                return
            yield _sse(item)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/dashboard")
def dashboard_data(user=Depends(current_user)):
    """The overview page: the company's approved recommendations with their status, and four counted or measured
    figures (dashboard.py). Nothing here calls a model."""
    import run_log
    from dashboard import build_dashboard
    from recommendation_store import CosmosRecommendationStore

    records = CosmosRecommendationStore().list_all()
    timings = run_log.CosmosRunStore().list_run_timings(user["user_id"])
    return build_dashboard(records, timings, user["user_id"], user["company_id"])


class AcceptRequest(BaseModel):
    session_id: str = Field(max_length=64)
    option_id: str = Field(max_length=64)


@app.post("/api/accept")
def accept(body: AcceptRequest, user=Depends(current_user)):
    """The PM approves one option from a finished evaluation. Saves it with its forecast and the 30, 60 and 90
    day follow-up dates (recommendation_store.py). Refused for a prohibited feature."""
    from recommendation_store import CosmosRecommendationStore, accept_option
    from session_store import SharedContainerSessionStore

    sessions = SharedContainerSessionStore()
    state = sessions.get(body.session_id, user["user_id"])
    if state is None:
        raise HTTPException(status_code=404, detail="That evaluation could not be found. Run it again.")
    result = accept_option(state, body.option_id, user["user_id"], user["company_id"], CosmosRecommendationStore())
    if result["session_update"]:
        sessions.put(body.session_id, user["user_id"], {**state, **result["session_update"]})
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["message"])
    record = result["record"]
    return {"message": result["message"], "recommendation_id": record["id"], "chosen": record["chosen"]["label"],
            "accepted_misses": record["chosen"]["accepted_misses"], "follow_ups": record["follow_ups"], "legal": record["legal"]}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
