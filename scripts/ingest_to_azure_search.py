"""
Creates the `eu-regulations` Azure AI Search index and pushes the parsed EU
AI Act chunks (from parse_eu_ai_act.py) into it, embedding each chunk's
`text` field with text-embedding-3-large along the way.

Design notes:
- `text` (paragraph/recital/annex-item level) is the searchable + embedded
  field. `parent_article_text` is stored and retrievable for
  hydration at answer time, but deliberately NOT marked searchable: since
  every paragraph of an article carries the same parent_article_text, making
  it searchable would let one article's content match many times over,
  skewing hybrid search's ranking.
- Every document carries a `regulation` field ("EU_AI_Act" for now) even
  though this is the only regulation in the index today, so a
  future split into a second index is a routing change, not a re-ingestion.
- No semantic configuration is defined yet - the Free (F0) tier this index
  currently lives on doesn't support the semantic reranker. Hybrid search
  (keyword + vector, fused server-side) works fine on Free; semantic config
  gets added when/if the service is upgraded to Basic, once post-launch
  grading shows the cost is justified.

Usage:
    python ingest_to_azure_search.py <chunks.json>
"""

import sys
import os
import json

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from urllib.parse import urlparse
from openai import AzureOpenAI
from azure.identity import InteractiveBrowserCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv()

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
FOUNDRY_PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]

INDEX_NAME = "eu-regulations"
EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
VECTOR_PROFILE_NAME = "hnsw-profile"
VECTOR_ALGORITHM_NAME = "hnsw-config"
EMBEDDING_BATCH_SIZE = 16
UPLOAD_BATCH_SIZE = 100


def build_index():
    return SearchIndex(
        name=INDEX_NAME,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="regulation", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="celex_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="oj_reference", type=SearchFieldDataType.String),
            SimpleField(name="article_number", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SearchableField(name="article_title", type=SearchFieldDataType.String),
            SimpleField(name="paragraph_number", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="recital_number", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="is_recital", type=SearchFieldDataType.Boolean, filterable=True),
            SimpleField(name="is_annex", type=SearchFieldDataType.Boolean, filterable=True),
            SimpleField(name="annex_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="annex_item_number", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="annex_sub_point", type=SearchFieldDataType.String, filterable=True),
            SimpleField(
                name="xrefs",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True,
            ),
            SearchableField(name="text", type=SearchFieldDataType.String),
            SimpleField(name="parent_article_text", type=SearchFieldDataType.String),
            SimpleField(name="source_url", type=SearchFieldDataType.String),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=EMBEDDING_DIMENSIONS,
                vector_search_profile_name=VECTOR_PROFILE_NAME,
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name=VECTOR_ALGORITHM_NAME)],
            profiles=[
                VectorSearchProfile(
                    name=VECTOR_PROFILE_NAME,
                    algorithm_configuration_name=VECTOR_ALGORITHM_NAME,
                )
            ],
        ),
    )


def get_openai_client(credential=None):
    # Same interactive-browser pattern as the other scripts, pinned to the
    # personal tenant so it doesn't pick up a cached work-account session.
    #
    # Deliberately NOT using AIProjectClient.get_openai_client() here: that
    # client is scoped to the project's "/openai/v1/" surface used for
    # calling agents (Responses API), which 404s on /embeddings. Embedding
    # deployments are called through the resource's own classic Azure OpenAI
    # endpoint instead - same underlying resource, different API surface.
    # A caller that makes several calls (the Orchestrator) passes one shared
    # credential so the browser sign-in happens once, not once per call.
    if credential is None:
        credential = InteractiveBrowserCredential(tenant_id=os.environ.get("AZURE_TENANT_ID"))
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
    resource_root = urlparse(FOUNDRY_PROJECT_ENDPOINT)
    resource_endpoint = f"{resource_root.scheme}://{resource_root.netloc}/"
    return AzureOpenAI(
        azure_endpoint=resource_endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-06-01",
    )


def embed_all(openai_client, records):
    for i in range(0, len(records), EMBEDDING_BATCH_SIZE):
        batch = records[i : i + EMBEDDING_BATCH_SIZE]
        texts = [r["text"] for r in batch]
        response = openai_client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=texts)
        for record, item in zip(batch, response.data):
            record["content_vector"] = item.embedding
        print(f"  embedded {min(i + EMBEDDING_BATCH_SIZE, len(records))}/{len(records)}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python ingest_to_azure_search.py <chunks.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} chunk records")

    print("Creating/updating index schema...")
    index_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))
    index_client.create_or_update_index(build_index())

    print(f"Embedding {len(records)} chunks with {EMBEDDING_DEPLOYMENT}...")
    openai_client = get_openai_client()
    embed_all(openai_client, records)

    print("Uploading documents to the index...")
    search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))
    for i in range(0, len(records), UPLOAD_BATCH_SIZE):
        batch = records[i : i + UPLOAD_BATCH_SIZE]
        result = search_client.upload_documents(documents=batch)
        failed = [r for r in result if not r.succeeded]
        if failed:
            print(f"  {len(failed)} documents FAILED in batch starting at {i}:")
            for f in failed[:5]:
                print(f"    {f.key}: {f.error_message}")
        else:
            print(f"  uploaded {min(i + UPLOAD_BATCH_SIZE, len(records))}/{len(records)}")

    print("Done.")


if __name__ == "__main__":
    main()
