"""
Quick manual test of hybrid search against the eu-regulations index -
BM25 keyword search + vector search, fused server-side, no reranker (Free
tier doesn't support the semantic reranker).

Usage:
    python test_search.py "some feature description"
"""
import sys
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv

from ingest_to_azure_search import get_openai_client, INDEX_NAME, EMBEDDING_DEPLOYMENT

load_dotenv()

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else (
        "our support bot reads a customer's message and guesses if they sound frustrated"
    )
    print(f"Query: {query!r}\n")

    openai_client = get_openai_client()
    vector = openai_client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=[query]).data[0].embedding

    search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))
    results = search_client.search(
        search_text=query,
        vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=5, fields="content_vector")],
        select=["id", "article_number", "article_title", "is_annex", "annex_id", "annex_item_number", "text"],
        top=5,
    )

    for r in results:
        label = f"Annex {r['annex_id']} item {r['annex_item_number']}" if r["is_annex"] else \
                f"Article {r['article_number']}" + (f" - {r['article_title']}" if r["article_title"] else "")
        print(f"[{r['@search.score']:.3f}] {label}  ({r['id']})")
        print(f"  {r['text'][:180]}")
        print()


if __name__ == "__main__":
    main()
