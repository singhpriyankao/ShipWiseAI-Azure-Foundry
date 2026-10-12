"""
The exact wording of the provisions a legal answer cites, read from the same index the Regulations agent searched
. No model is involved: a citation such as Article 50(1) is looked up by its numbers and the stored
text is returned as it is, so the PM can read the Act itself next to the agent's explanation.

If a passage can't be fetched, it is left out; the answer never depends on this.
"""
import os
import re

MAX_CITATIONS = 4
MAX_CHARS = 1400


def citation_label(citation):
    """'Article 50(1)', 'Annex III, item 5(b)', 'Recital 53'."""
    if citation.get("article_number") is not None:
        label = f"Article {citation['article_number']}"
        if citation.get("paragraph_number") is not None:
            label += f"({citation['paragraph_number']})"
        return label
    if citation.get("annex_id"):
        label = f"Annex {citation['annex_id']}"
        if citation.get("annex_item_number") is not None:
            label += f", item {citation['annex_item_number']}"
            if citation.get("annex_sub_point"):
                label += f"({citation['annex_sub_point']})"
        return label
    if citation.get("recital_number") is not None:
        return f"Recital {citation['recital_number']}"
    return "EU AI Act"


def _int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _safe(value):
    """Letters and digits only: the values come from the agent's output, so nothing else may reach a filter."""
    text = str(value) if value is not None else ""
    return text if re.fullmatch(r"[A-Za-z0-9]{1,8}", text) else None


def passage_filter(citation):
    """The search filter for one citation, or None if it can't be expressed safely."""
    article, paragraph, recital = _int(citation.get("article_number")), _int(citation.get("paragraph_number")), _int(citation.get("recital_number"))
    if article is not None:
        return f"article_number eq {article}" + (f" and paragraph_number eq {paragraph}" if paragraph is not None else "")
    annex = _safe(citation.get("annex_id"))
    if annex:
        out = f"annex_id eq '{annex}'"
        item, sub = _int(citation.get("annex_item_number")), _safe(citation.get("annex_sub_point"))
        if item is not None:
            out += f" and annex_item_number eq {item}"
            if sub:
                out += f" and annex_sub_point eq '{sub}'"
        return out
    if recital is not None:
        return f"recital_number eq {recital}"
    return None


def _default_client():
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from ingest_to_azure_search import INDEX_NAME
    return SearchClient(os.environ["AZURE_SEARCH_ENDPOINT"], INDEX_NAME, AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]))


def quote_citations(citations, search_client=None, max_chars=MAX_CHARS):
    """[{"label", "title", "text", "source_url"}] for up to four distinct citations. A citation to a whole article
    (no paragraph) returns its first paragraphs in order. Text longer than `max_chars` is cut with an ellipsis."""
    quotes, seen = [], set()
    client = search_client
    for citation in citations or []:
        label = citation_label(citation)
        if label in seen or len(quotes) >= MAX_CITATIONS:
            continue
        seen.add(label)
        flt = passage_filter(citation)
        if not flt:
            continue
        try:
            client = client or _default_client()
            hits = list(client.search(search_text="*", filter=flt, top=6))
        except Exception:
            continue
        if not hits:
            continue
        if citation.get("paragraph_number") is None and len(hits) > 1:
            hits = sorted(hits, key=lambda h: (h.get("paragraph_number") is None, h.get("paragraph_number") or 0))
        text = "\n\n".join(f"{h['paragraph_number']}. {h['text']}" if h.get("paragraph_number") is not None and len(hits) > 1 else h["text"] for h in hits if h.get("text"))
        text = re.sub(r"[ \t]+", " ", text).strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0].rstrip() + " …"
        first = hits[0]
        quotes.append({"label": label, "title": first.get("article_title"), "text": text, "source_url": first.get("source_url")})
    return quotes
