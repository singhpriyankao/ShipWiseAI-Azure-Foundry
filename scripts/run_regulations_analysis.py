"""
Calls the Regulations agent end to end: retrieval (bounded fan-out search +
the Annex III -> Article 6 expansion), the agent call itself, all four
guardrails, and the fixed legal disclaimer.

Same harness role as run_model_stack_analysis.py - run manually for now,
proves the retrieve -> call -> guard pattern works end to end before the
Orchestrator exists.

No Cosmos DB write, on purpose: the `recommendations` container exists so a
PM can later report whether a forecast held up (the `outcome` field), and a
legal classification isn't a forecast a PM can confirm after shipping. Run
logging for post-prod evals is a separate piece, to be designed with them.

Guardrail failure handling (a deliberate policy per guardrail, not one blanket rule):
- Grounding-coverage gate and applicability-date check can fail in a way
  where the correct value is already known the moment the check runs
  (retrieval was empty, or nothing retrieved discusses timing) - corrected
  directly in code, no extra agent call, since retrying would just send the
  same input and very likely get the same wrong answer again.
- Citation validator, recital-only block, and the *other* grounding-gate
  failure mode ("grounded" claimed but zero citations even though
  retrieved_chunks was non-empty) are "probabilistic" failures - the model
  had usable chunks and didn't use them well. Retried up to MAX_ATTEMPTS
  times, same pattern as bounded_fanout_search.py's decompose_query retry
  (an otherwise-identical call occasionally does better on a second try).
- If retries are exhausted without a clean pass, the script returns an
  honest failure result rather than showing an answer already known to
  have failed a guardrail.
"""
import json
import os

from azure.ai.projects import AIProjectClient
from azure.identity import InteractiveBrowserCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

from bounded_fanout_search import (
    decompose_query, hybrid_search, reciprocal_rank_fusion,
    expand_high_risk_context, expand_applicability_context, expand_definitions_context, get_openai_client, _strip_code_fence,
    SEARCH_ENDPOINT, SEARCH_KEY, INDEX_NAME, RESULTS_PER_SUBQUERY,
)
import run_log
import runtime_paths
from foundry_clients import budgeted_call, with_rate_limit_retry
from regulations_guardrails import (
    validate_citations, check_grounding_coverage, check_recital_only,
    check_applicability_grounding, check_applicability_dates, check_derogation_addressed,
    add_missing_derogation_citation, prohibition_applicability_status, mark_insufficient_information,
    derive_applicability_status,
)

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
AGENT_NAME = "regulations"
AGENT_VERSION = os.environ["REGULATIONS_AGENT_VERSION"]
MAX_ATTEMPTS = 2

APPLICABILITY_UNVERIFIED = (
    "The applicability date could not be verified against the retrieved text. "
    "See Article 113 (Entry into force and application) for the phase-in dates."
)

DISCLAIMER = (
    "This is AI-generated regulatory analysis, not legal advice. Review "
    "with Legal before acting on it, especially for 'prohibited' or "
    "'high_risk' classifications."
)


def retrieve_chunks(feature_description, model_client=None):
    # model_client: the shared classic-endpoint client when called from the
    # Orchestrator; created here (one browser sign-in) when run by hand.
    openai_client = model_client or get_openai_client()
    search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))

    queries = decompose_query(openai_client, feature_description)
    result_lists = [hybrid_search(search_client, openai_client, q, RESULTS_PER_SUBQUERY) for q in queries]
    fused = reciprocal_rank_fusion(result_lists)
    expand_high_risk_context(fused, search_client)
    expand_applicability_context(fused, search_client)
    expand_definitions_context(fused, search_client)

    return [record for doc_id, score, record in fused]


# The registered agent's own instructions, which the service adds to what we send (about 3,500 tokens).
AGENT_PROMPT_TOKENS = 3500

# Fields the agent never reads. The guardrails run on the full chunks in code, not on this copy,
# so dropping them from what the model sees can't weaken a check.
MODEL_IGNORES = ("source_url", "celex_id", "oj_reference")


def slim_for_model(retrieved_chunks):
    """The chunks as the model should see them: the same information with less repetition.
    Each chunk carries its whole parent article, so one article retrieved through five
    paragraphs was sent five times (about a third of a typical request, 39,000 tokens, which is
    what hit the per-minute token limit). The first chunk of an article keeps the full parent
    text and later ones point back to it, which the prompt's "read the parent text" rule can
    follow. Search scores, highlights, captions and empty fields are dropped as well. Returns a
    new list; the originals, which the guardrails check against, are untouched."""
    slim, first_seen = [], {}
    for chunk in retrieved_chunks:
        copy = {k: v for k, v in chunk.items()
                if not k.startswith("@search") and k not in MODEL_IGNORES and v is not None and v != ""}
        parent = copy.get("parent_article_text")
        if parent:
            if parent in first_seen:
                copy["parent_article_text"] = f"[Same article text as chunk {first_seen[parent]}, given above.]"
            else:
                first_seen[parent] = chunk.get("id")
        slim.append(copy)
    return slim


def call_regulations_agent(feature_description, retrieved_chunks, agent_client=None):
    # agent_client: the shared Responses-API client when called from the
    # Orchestrator. Otherwise sign in here, pinned to the personal tenant -
    # without that the browser login can pick up a cached work-account
    # session and fail with a tenant-mismatch error.
    if agent_client is None:
        credential = InteractiveBrowserCredential(tenant_id=os.environ.get("AZURE_TENANT_ID"))
        agent_client = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential).get_openai_client()
    openai_client = agent_client

    user_input = {
        "feature_description": feature_description,
        "retrieved_chunks": slim_for_model(retrieved_chunks),
    }

    response = budgeted_call(lambda: openai_client.responses.create(
        input=[{"role": "user", "content": json.dumps(user_input)}],
        extra_body={
            "agent_reference": {
                "name": AGENT_NAME,
                "version": AGENT_VERSION,
                "type": "agent_reference",
            }
        },
    ), user_input, expected_output=700, overhead=AGENT_PROMPT_TOKENS, key="regulations")

    raw_text = _strip_code_fence(response.output_text)
    try:
        output = json.loads(raw_text)
    except json.JSONDecodeError as e:
        debug_path = runtime_paths.writable_path("last_failure_raw.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(raw_text)
        raise RuntimeError(
            f"Agent reply was not valid JSON ({e}). Full response written to {debug_path}"
        ) from e

    return user_input, output


def run_regulations_analysis(feature_description, clients=None):
    # clients: a foundry_clients.Clients when called by the Orchestrator, so
    # retrieval and the agent call share one sign-in.
    model_client = clients.model_client if clients else None
    agent_client = clients.agent_client if clients else None
    retrieved_chunks = with_rate_limit_retry(retrieve_chunks, feature_description, model_client)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        user_input, output = call_regulations_agent(feature_description, retrieved_chunks, agent_client)

        # Deterministic corrections first - these two failure modes have a
        # known-correct value the instant the check runs, so fix them in
        # code rather than burning a retry (and an API call) on them.
        no_chunks = not retrieved_chunks and output.get("grounding_status") != "ungrounded_fallback"
        if no_chunks:
            output["grounding_status"] = "ungrounded_fallback"
        run_log.event("regulations", "nothing_retrieved_marked_ungrounded", no_chunks)

        ungrounded_date = check_applicability_grounding(output, retrieved_chunks)
        if ungrounded_date:
            output["applicability_status"] = "No applicability date found in retrieved provisions."
        run_log.event("regulations", "ungrounded_applicability_replaced", ungrounded_date)

        # A non-answer (too vague, out of scope, nothing retrieved) must not
        # carry a real classification. Explicit value, no retry: the agent
        # already told us it couldn't assess, retrying won't change that.
        insufficient = mark_insufficient_information(output)
        run_log.event("regulations", "non_answer_classification_cleared", insufficient)
        if insufficient:
            output["disclaimer"] = DISCLAIMER
            return user_input, output, "insufficient_information"

        # Assessment filled in but Article 6(3) not cited: the right citation is
        # known, so add it in code instead of burning a retry on it.
        run_log.event("regulations", "article_6_3_citation_added", add_missing_derogation_citation(output, retrieved_chunks))

        # A date the model stated that isn't in the retrieved text (or no date
        # at all) - replace with a pointer to the article that holds the real
        # dates. Article 113 is always retrieved, so this is always true.
        unverified_date = check_applicability_dates(output, retrieved_chunks)
        if unverified_date:
            output["applicability_status"] = APPLICABILITY_UNVERIFIED
        run_log.event("regulations", "unverified_date_replaced", unverified_date)

        # high_risk / transparency_obligated: the model can match a real date
        # to the wrong provision (Article 50 given the 2025 date, which belongs
        # to other chapters), which the check above can't see. Build the status
        # from the cited provisions and Article 113 in code. "" = nothing
        # datable cited (leave as is); None = couldn't map, use the pointer.
        if output.get("classification") in ("high_risk", "transparency_obligated"):
            derived = derive_applicability_status(output, retrieved_chunks)
            run_log.event("regulations", "applicability_built_in_code", bool(derived) or derived is None)
            if derived:
                output["applicability_status"] = derived
            elif derived is None:
                output["applicability_status"] = APPLICABILITY_UNVERIFIED

        # A prohibition's date is fixed by Article 113(a) (Chapter II), so take
        # it from the retrieved text in code rather than trusting the model's.
        if output.get("classification") == "prohibited":
            output["applicability_status"] = prohibition_applicability_status(retrieved_chunks) or APPLICABILITY_UNVERIFIED

        # Whatever's left needs the model to actually try again - there's
        # no value we can just substitute in code for these.
        bad_citations = validate_citations(output, retrieved_chunks)
        recital_problem = check_recital_only(output)
        grounding_problem = check_grounding_coverage(output, retrieved_chunks)  # only the "grounded but zero citations" case can still trigger here
        derogation_problem = check_derogation_addressed(output, retrieved_chunks)

        run_log.event("regulations", "citation_check", bool(bad_citations), f"{len(bad_citations)} invalid" if bad_citations else None)
        run_log.event("regulations", "recital_only_check", bool(recital_problem))
        run_log.event("regulations", "grounding_coverage_check", bool(grounding_problem))
        run_log.event("regulations", "derogation_addressed_check", bool(derogation_problem))
        if not bad_citations and not recital_problem and not grounding_problem and not derogation_problem:
            output["disclaimer"] = DISCLAIMER
            return user_input, output, "passed"
        run_log.event("regulations", "retry_after_failed_guardrail", attempt < MAX_ATTEMPTS, f"attempt {attempt}")

        print(
            f"  (attempt {attempt} failed a guardrail - bad_citations={bad_citations}, "
            f"recital_problem={recital_problem!r}, grounding_problem={grounding_problem!r}, "
            f"derogation_problem={derogation_problem!r}, "
            f"{'retrying' if attempt < MAX_ATTEMPTS else 'giving up'})"
        )

    # Retries exhausted - return an honest failure rather than an answer
    # already known to have failed a guardrail.
    failure_output = {
        "answer": "Unable to produce a reliably grounded answer after retrying - this needs manual review rather than an automated result.",
        "citations": [],
        "classification": "analysis_failed",
        "derogation_assessment": {"outcome": "not_assessed", "reason": ""},
        "applicability_status": "not_assessed",
        "grounding_status": "ungrounded_fallback",
        "missing_inputs": [],
        "disclaimer": DISCLAIMER,
    }
    failure_input = {"feature_description": feature_description, "retrieved_chunks": retrieved_chunks}
    return failure_input, failure_output, "failed_guardrails"


if __name__ == "__main__":
    user_input, output, guardrail_status = run_regulations_analysis(
        "our support bot reads a customer's message and guesses if they sound frustrated"
    )

    print("AGENT OUTPUT:")
    print(json.dumps(output, indent=2))
    print("\nGUARDRAIL STATUS:", guardrail_status)
