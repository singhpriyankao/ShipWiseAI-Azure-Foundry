"""
Reads the Competitive Analysis agent's v4 reply: a fixed plain-text layout whose statements are cited with the
search tool's own citations .

Why this exists. The v3 agent answered in JSON, which stops the Bing tool from attaching the pages it found, so
every address in the JSON was typed by the model from memory and was often invented. In plain text the tool returns
real citations: each has the page address and its start and end position in the reply text. This module attaches to
each Pricing, Accuracy and Latency line the addresses cited inside that line, so the model never types an address.

What it returns is the structure the rest of the system already uses (`pricing_notes`, `pricing_source_url`, and so
on, plus `pricing_source_urls` when a line cites several pages), so the source check, the Orchestrator and the page
need no change.

Rules, all in code:
- A line with no citation inside it has no source, whatever it says. The source check then keeps it off the page.
- A line the agent marked "No source found." keeps no address.
- Citation markers are removed from the notes.
- If the reply doesn't follow the layout, fewer claims are returned, never guessed ones, and if nothing can be read
  at all the result is None so the caller can ask again.
"""
import re

MARKER = re.compile(r"[ \t]*【[^】]*】")
LABEL = re.compile(r"^(GROUNDING|MISSING INPUTS|OUT OF SCOPE|COMPETITOR\s+\d+|Product|Vendor|Pricing|Accuracy|Latency|Comparison)\s*:?\s*(.*)$", re.IGNORECASE)
SOURCED_FIELDS = ("pricing", "accuracy", "latency")
NO_SOURCE_PREFIX = "No source found:"   # the prefix competitive_checks.py recognises


def _clean(text):
    """Remove citation markers and tidy the spaces they leave behind (' .' becomes '.')."""
    text = MARKER.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([.,;:])", r"\1", text)


def _message_parts(response):
    """(text, annotations) from a Responses API reply, given as a dict (model_dump) or the SDK object."""
    data = response if isinstance(response, dict) else response.model_dump()
    for item in data.get("output") or []:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    return part.get("text") or "", part.get("annotations") or []
    return None, []


def _urls_between(annotations, start, end):
    """Addresses of the citations that sit inside [start, end), in order, each once. Only web addresses."""
    found = []
    for a in annotations:
        position, url = a.get("start_index"), a.get("url")
        if a.get("type") == "url_citation" and isinstance(position, int) and start <= position < end and isinstance(url, str) and url.startswith(("http://", "https://")) and url not in found:
            found.append(url)
    return found


def parse_v4(response):
    """The v3-shaped output, or None if the reply can't be read as the v4 layout at all."""
    text, annotations = _message_parts(response)
    if not text or not text.strip():
        return None

    output = {"competitors": [], "grounding_status": None, "missing_inputs": []}
    current, seen_layout = None, False
    offset = 0
    for line in text.split("\n"):
        start, end = offset, offset + len(line)
        offset = end + 1
        found = LABEL.match(line.strip())
        if not found:
            continue
        label, value = found.group(1).upper(), found.group(2)
        seen_layout = True

        if label == "GROUNDING":
            status = _clean(value).lower()
            output["grounding_status"] = "ungrounded_fallback" if "ungrounded" in status else "grounded"
        elif label == "MISSING INPUTS":
            cleaned = _clean(value)
            output["missing_inputs"] = [] if cleaned.lower() in ("", "none", "none.") else [p.strip() for p in cleaned.split(";") if p.strip()]
        elif label == "OUT OF SCOPE":
            output["missing_inputs"] = ["out of scope: " + (_clean(value) or "not a competitive question")]
            output["grounding_status"] = "ungrounded_fallback"
        elif label.startswith("COMPETITOR"):
            current = {"product": "", "vendor": "", "comparison_to_feature": ""}
            for field in SOURCED_FIELDS:
                current.update({f"{field}_notes": "", f"{field}_source_url": None, f"{field}_source_urls": []})
            output["competitors"].append(current)
        elif current is not None:
            name = label.lower()
            if name in ("product", "vendor"):
                current[name] = _clean(value)
            elif name == "comparison":
                current["comparison_to_feature"] = _clean(value)
            elif name in SOURCED_FIELDS:
                note = _clean(value)
                urls = _urls_between(annotations, start, end)
                if note.lower().startswith("no source found"):
                    note, urls = NO_SOURCE_PREFIX + " the research found no source for this.", []
                current[f"{name}_notes"] = note
                current[f"{name}_source_urls"] = urls
                current[f"{name}_source_url"] = urls[0] if urls else None

    output["competitors"] = [c for c in output["competitors"] if c["product"]]
    if not seen_layout:
        return None
    if output["grounding_status"] is None:
        output["grounding_status"] = "grounded" if output["competitors"] else "ungrounded_fallback"
    return output
