"""
Calls the Competitive Analysis agent with a feature description and prints
its parsed JSON response.

No Cosmos DB write here, unlike run_model_stack_analysis.py — this agent
makes factual claims about competitors, not forecasts, so there's nothing
for a Learning Agent to reconcile later and no reason to persist it.

Same calling convention as Model Stack Analysis: built in Foundry's agent
builder surface, called via the OpenAI-compatible Responses API using the
agent's NAME and VERSION.
"""
import json
import os

from azure.ai.projects import AIProjectClient
from azure.identity import InteractiveBrowserCredential
from dotenv import load_dotenv

import runtime_paths
from competitive_checks import check_competitor_sources
from competitive_parse import parse_v4
from foundry_clients import budgeted_call, call_agent_for_json

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
AGENT_NAME = "competitive-analysis"
AGENT_VERSION = os.environ["COMPETITIVE_ANALYSIS_AGENT_VERSION"]
# The agent's instructions plus the web search results it reads (measured at about 4,700 input tokens in total).
AGENT_AND_SEARCH_TOKENS = 4500
# v4 makes up to four searches; three measured replies used 4,500 to 5,200 input tokens (2026-10-06).
V4_AGENT_AND_SEARCH_TOKENS = 5500
V4_ATTEMPTS = 2


def uses_text_layout():
    """v4 and later answer in plain text with the search tool's citations (competitive_parse.py); v3 answers in JSON."""
    try:
        return int(AGENT_VERSION) >= 4
    except ValueError:
        return False


def call_agent_for_layout(make_call, label="competitive_analysis", max_attempts=V4_ATTEMPTS):
    """Calls the v4 agent and reads its reply with parse_v4. A reply that can't be read as the layout at all is asked
    for once more (the same one-retry rule as the JSON agents); a reply that is readable but has few claims is
    returned as it is, because fewer claims is the safe outcome."""
    last_text = None
    for attempt in range(1, max_attempts + 1):
        response = make_call()
        output = parse_v4(response)
        if output is not None:
            return output
        last_text = getattr(response, "output_text", None) or ""
        path = runtime_paths.writable_path(f"last_failure_raw_{label}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(last_text)
        print(f"  ({label}: reply did not follow the layout on attempt {attempt}; {'retrying' if attempt < max_attempts else 'giving up'})")
    raise RuntimeError(f"{label} reply did not follow the v4 layout after {max_attempts} attempts. Last reply written to {path}")


def call_competitive_analysis(feature_description, openai_client=None):
    # openai_client: the shared Responses-API client when called from the
    # Orchestrator. Otherwise sign in here, pinned to the personal tenant
    # explicitly — without this, the browser login can pick up a cached
    # MathWorks work-account session and fail with a tenant-mismatch error.
    if openai_client is None:
        credential = InteractiveBrowserCredential(tenant_id=os.environ.get("AZURE_TENANT_ID"))
        project_client = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential)
        openai_client = project_client.get_openai_client()

    user_input = {"feature_description": feature_description}

    text_layout = uses_text_layout()
    make_call = lambda: budgeted_call(lambda: openai_client.responses.create(
        input=[{"role": "user", "content": json.dumps(user_input)}],
        extra_body={
            "agent_reference": {
                "name": AGENT_NAME,
                "version": AGENT_VERSION,
                "type": "agent_reference",
            }
        },
    ), user_input, expected_output=800, overhead=V4_AGENT_AND_SEARCH_TOKENS if text_layout else AGENT_AND_SEARCH_TOKENS, key="competitive")
    output = call_agent_for_layout(make_call) if text_layout else call_agent_for_json("competitive_analysis", make_call)

    # Source rules run in code, not in the prompt.
    output = check_competitor_sources(output)

    return user_input, output


if __name__ == "__main__":
    user_input, output = call_competitive_analysis(
        feature_description=(
            "A chatbot for support agents to answer customer questions "
            "from our existing knowledge base."
        ),
    )

    print("AGENT OUTPUT:")
    print(json.dumps(output, indent=2))
