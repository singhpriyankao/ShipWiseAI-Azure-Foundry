"""
Bounded fan-out search - rewrites a plain-English feature description into
several sharper, regulation-flavored search queries using a small cheap
model, runs each one as a hybrid search against the eu-regulations index,
and merges the ranked result lists into one combined ranking.

This is the fix for the vocabulary mismatch between how a PM describes a
feature and how the Act is worded: a plain feature description ("guesses if a customer sounds frustrated")
shares almost no words with the Act's own language ("emotion recognition
system"), so a single search, however well it is built, can miss the exact
provision that matters. Rewriting the question into several queries closer
to the Act's own vocabulary, before searching, closes that gap without an
open-ended agentic loop - one cheap rewriting step, then parallel searches,
then a merge. Not built: a single conditional retry (re-querying once if the
merged top score is still weak), deferred until fan-out alone is shown to
be insufficient.

Usage:
    python bounded_fanout_search.py "some feature description"
"""
import sys
import os
import json

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv

from ingest_to_azure_search import get_openai_client, INDEX_NAME, EMBEDDING_DEPLOYMENT

load_dotenv()

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
DECOMPOSITION_DEPLOYMENT = "gpt-4.1-nano"
RRF_K = 60
RESULTS_PER_SUBQUERY = 10
FINAL_TOP_N = 5

FIELDS = ["id", "article_number", "article_title", "is_annex", "annex_id", "annex_item_number",
            "annex_sub_point", "is_recital", "recital_number", "text", "parent_article_text",
            "paragraph_number", "regulation", "celex_id", "oj_reference", "source_url"]

DECOMPOSITION_PROMPT = """You help find relevant provisions in the EU AI Act \
given a plain description of an AI feature.

Given the feature description below, write 3 to 5 short search queries that \
would help find the exact relevant provisions - phrase them using the kind \
of formal regulatory language the Act itself uses (for example: risk \
classification, prohibited practices, transparency obligations, human \
oversight, biometric data, emotion recognition, high-risk AI systems), \
rather than repeating the feature description in plain conversational \
language. Each query should target a distinct angle on the feature, not \
just reword the same idea.

Output only a JSON array of strings, nothing else - no markdown code fences, \
no explanation.

Feature description: {feature_description}"""


def _strip_code_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def decompose_query(openai_client, feature_description, max_attempts=2):
    # Both gpt-4o (see model-stack-analysis.md) and gpt-4.1-nano occasionally
    # emit malformed JSON on an otherwise-identical call - non-deterministic,
    # not a prompt bug. Retrying once was deliberately deferred back then;
    # building it now since it's actually blocking a real test run.
    last_error = None
    for attempt in range(1, max_attempts + 1):
        response = openai_client.chat.completions.create(
            model=DECOMPOSITION_DEPLOYMENT,
            messages=[{"role": "user", "content": DECOMPOSITION_PROMPT.format(feature_description=feature_description)}],
        )
        raw_text = _strip_code_fence(response.choices[0].message.content)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            last_error = e
            print(f"  (decomposition attempt {attempt} produced malformed JSON, {'retrying' if attempt < max_attempts else 'giving up'})")
    raise RuntimeError(f"Decomposition failed after {max_attempts} attempts: {last_error}")


def hybrid_search(search_client, openai_client, query, top_k):
    vector = openai_client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=[query]).data[0].embedding
    results = search_client.search(
        search_text=query,
        vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=top_k, fields="content_vector")],
        select=["id", "article_number", "article_title", "is_annex", "annex_id", "annex_item_number",
                "annex_sub_point", "is_recital", "recital_number", "text","parent_article_text", "paragraph_number","regulation","celex_id", "oj_reference", "source_url"],
        top=top_k,
    )
    return list(results)


def reciprocal_rank_fusion(result_lists, k=RRF_K):
    scores = {}
    records = {}
    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            doc_id = r["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            records[doc_id] = r
    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(doc_id, score, records[doc_id]) for doc_id, score in fused]


def label_for(r):
    if r["is_annex"]:
        loc = f"Annex {r['annex_id']} item {r['annex_item_number']}"
        if r["annex_sub_point"]:
            loc += f"({r['annex_sub_point']})"
        return loc
    if r["is_recital"]:
        return f"Recital {r['recital_number']}"
    title = f" - {r['article_title']}" if r["article_title"] else ""
    return f"Article {r['article_number']}{title}"

def expand_high_risk_context(fused, search_client):
      # fused is a list of (doc_id, score, record) tuples. Build the set of IDs
      # already present so we don't fetch something we already have.
      existing_ids = {doc_id for doc_id, score, r in fused}

      # Only expand if something from Annex III specifically showed up - the
      # Article 6(2)/6(3) rule is what gives Annex III membership its meaning,
      # and doesn't apply to the other 12 annexes.
      has_annex_iii_hit = any(r["annex_id"] == "III" for doc_id, score, r in fused)
      if not has_annex_iii_hit:
          return

      # Article 6(2) is the high-risk trigger, 6(3) is the narrow derogation -
      # always fetch both together, since 6(3) is meaningless without 6(2).
      for chunk_id in ("art_6_para_2", "art_6_para_3"):
          if chunk_id not in existing_ids:
              # get_document returns a LookupDocument, not a plain dict like
              # search() results - convert so it's JSON-serializable later.
              record = dict(search_client.get_document(key=chunk_id, selected_fields=FIELDS))
              fused.append((chunk_id, 0.0, record))

def expand_applicability_context(fused, search_client):
    # Article 113 is the Act's entry-into-force-and-application article. Every
    # answer needs an applicability_status, and search won't reliably surface
    # the article that holds the dates (in testing it often didn't, and the
    # model filled the gap from memory). One extra chunk per call, always.
    existing_ids = {doc_id for doc_id, score, r in fused}
    if "art_113" not in existing_ids:
        record = dict(search_client.get_document(key="art_113", selected_fields=FIELDS))
        fused.append(("art_113", 0.0, record))


def expand_definitions_context(fused, search_client):
    # Article 3 holds the Act's definitions (it is ingested as one chunk, so it
    # can't be pulled per term). Classification often turns on a defined term -
    # in a live run the agent called a text-only chat feature "emotion
    # recognition" without ever seeing that Article 3(39) defines it by
    # biometric data. Always retrieved, same as Article 113: about 4k tokens
    # (roughly a cent) per call.
    existing_ids = {doc_id for doc_id, score, r in fused}
    if "art_3" not in existing_ids:
        record = dict(search_client.get_document(key="art_3", selected_fields=FIELDS))
        fused.append(("art_3", 0.0, record))


def main():
    feature_description = sys.argv[1] if len(sys.argv) > 1 else (
        "our support bot reads a customer's message and guesses if they sound frustrated"
    )
    print(f"Feature description: {feature_description!r}\n")

    openai_client = get_openai_client()
    search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))

    queries = decompose_query(openai_client, feature_description)
    print("Decomposed into:")
    for q in queries:
        print(f"  - {q}")
    print()

    result_lists = [hybrid_search(search_client, openai_client, q, RESULTS_PER_SUBQUERY) for q in queries]
    fused = reciprocal_rank_fusion(result_lists)
    expand_high_risk_context(fused, search_client)
    expand_applicability_context(fused, search_client)
    expand_definitions_context(fused, search_client)

    print(f"Top {FINAL_TOP_N} after merging all sub-query results:\n")
    for doc_id, score, r in fused[:FINAL_TOP_N]:
        print(f"[{score:.4f}] {label_for(r)}  ({doc_id})")
        print(f"  {r['text'][:180]}")
        print()


if __name__ == "__main__":
    main()
