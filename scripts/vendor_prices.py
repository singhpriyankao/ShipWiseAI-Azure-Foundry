"""
Reads model prices straight from the vendors' own pricing pages, in code.

Why: the Model Stack agent was asked to find prices by web search, and over
three prompt versions its numbers never became trustworthy (the same model was
priced at $0.03 and then $0.003 per 1K tokens in consecutive runs, and it kept
proposing models several generations old). Here code fetches the page, reads
the table, and hands the agent a price list it can't alter. The list also tells
us which models are current, because the vendor's page only lists current ones.

Covers Anthropic (current and older models) and OpenAI (the flagship text models only;
its other tables mix in image, audio and tool pricing). Not covered yet: Google, and
open-source models, whose price depends on the host.

If a page can't be fetched or its layout no longer parses, that vendor is
reported in `problems` and left out. Nothing is ever filled in from memory.
Prices are per 1M tokens, Standard (not batch or priority) rates, short context.
"""
import html
import json
import os
import re
import time
import urllib.request
from datetime import date, datetime, timezone

import runtime_paths

ANTHROPIC_URL = "https://docs.anthropic.com/en/docs/about-claude/pricing"
OPENAI_URL = "https://platform.openai.com/docs/pricing"

CACHE_PATH = runtime_paths.writable_path("price_cache.json")
# A hosted demo can ship a saved copy and set this longer, so a fresh instance does not re-read every vendor page.
CACHE_TTL_SECONDS = int(os.environ.get("SHIPWISE_CACHE_TTL_SECONDS", 24 * 3600))
MIN_MODELS_PER_VENDOR = 3  # fewer than this means the page layout probably changed
MAX_PLAUSIBLE_PRICE = 1000.0  # dollars per 1M tokens; anything above is a misread


class PriceSourceError(Exception):
    pass


def _page_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
    raw = re.sub(r"(?is)<(script|style).*?</\1>", "", raw)
    raw = re.sub(r"(?i)</(tr|p|div|h\d|li)>", "\n", raw)
    raw = re.sub(r"(?i)</t[dh]>", " | ", raw)
    raw = re.sub(r"<[^>]+>", "", raw)
    return html.unescape(re.sub(r"[ \t]+", " ", raw))


def _plausible(value):
    return 0 <= value <= MAX_PLAUSIBLE_PRICE


# The model name and its description run together in the page text ("Claude
# Fable 5.1For demanding..."), so the name is matched by shape, not by line.
_ANTHROPIC_ROW = re.compile(
    r"(Claude [A-Za-z]+ \d+(?:\.\d+)?)[^\n]*\n(?:\s*\n)*\s*\|\s*\$([\d.]+) / MTok\s*\n\s*\|\s*\$([\d.]+) / MTok"
)


def parse_anthropic(text):
    """`listing` is "current" for the models in the main table and "additional" for the
    older ones under "Additional models", so the agent can prefer current models."""
    marker = text.find("Additional models")
    models, seen = [], set()
    for m in _ANTHROPIC_ROW.finditer(text):
        name = m.group(1)
        if name in seen:
            continue
        seen.add(name)
        a, b = float(m.group(2)), float(m.group(3))
        if _plausible(a) and _plausible(b):
            listing = "current" if marker == -1 or m.start() < marker else "additional"
            models.append({"vendor": "anthropic", "model": name, "input_per_1m": a, "output_per_1m": b, "listing": listing})
    return models


_TIERS = ("Standard", "Batch", "Flex", "Fast", "Ultrafast", "Priority")
_DOLLARS = re.compile(r"^\$\s*([\d.]+)$")


def parse_openai(text):
    """Header-driven: remember the latest 'Model | Input | ... | Output' header and
    tier name, then read the first Input and first Output column of each Standard row."""
    models, seen = [], set()
    tier, in_col, out_col = None, None, None
    for line in (l.strip() for l in text.split("\n")):
        if line == "All models" and models:
            break  # end of the flagship text-model table; later tables are images, audio, tools
        if line in _TIERS:
            tier = line
            continue
        cells = [c.strip() for c in line.split("|")]
        if cells and cells[0] == "Model" and "Input" in cells and "Output" in cells:
            in_col, out_col = cells.index("Input"), cells.index("Output")
            continue
        if tier != "Standard" or in_col is None or len(cells) <= max(in_col, out_col):
            continue
        if not re.fullmatch(r"[a-z][\w.\-]*", cells[0]):
            continue
        m_in, m_out = _DOLLARS.match(cells[in_col]), _DOLLARS.match(cells[out_col])
        if not (m_in and m_out) or cells[0] in seen:
            continue
        a, b = float(m_in.group(1)), float(m_out.group(1))
        if _plausible(a) and _plausible(b):
            seen.add(cells[0])
            models.append({"vendor": "openai", "model": cells[0], "input_per_1m": a, "output_per_1m": b, "listing": "current"})
    return models


# --------------------------------------------------------------------------
# Generic path: find the pricing page through the vendor's llms.txt, have a small
# model read it, and have code verify what the model claims it read.
# --------------------------------------------------------------------------

# Vendors publish llms.txt (an index of machine-readable docs) and markdown copies of
# pages. The index finds the pricing page even when its URL moves (Anthropic's did),
# and the fallback is a known markdown URL when the index doesn't list one.
SOURCES = (
    {"vendor": "anthropic", "llms_txt": "https://docs.anthropic.com/llms.txt",
     "fallback": "https://docs.anthropic.com/en/docs/about-claude/pricing.md"},
    {"vendor": "openai", "llms_txt": "https://developers.openai.com/llms.txt",
     "fallback": "https://platform.openai.com/docs/pricing.md"},
    {"vendor": "google", "llms_txt": None,
     "fallback": "https://ai.google.dev/gemini-api/docs/pricing.md.txt"},
)
EXTRACTION_MODEL = os.environ.get("PRICE_EXTRACTION_MODEL", "gpt-4.1-nano")
MAX_PAGE_CHARS = 200_000
EXTRACTION_ATTEMPTS = 3

EXTRACTION_PROMPT = """You read a vendor's pricing page (markdown) and list the prices of its text-generation models.

Return one JSON object: {"models": [ ... ]}. Each entry has exactly these fields:
- "model": the model name as the page writes it
- "input_per_1m": number, US dollars per 1 million input tokens
- "output_per_1m": number, US dollars per 1 million output tokens

Rules:
- Use the standard pay-as-you-go rate, short context, per 1 million tokens. Not batch, flex, priority, cached or cache-write prices.
- Include only text-generation models. Skip image, audio, video, embedding, moderation, transcription and tool prices.
- Only include a model if the page states both prices per million tokens (written as $ per MTok, per 1M tokens, or similar). Otherwise leave it out.
- If a price changes on a date, use the price in effect today. Today's date is {today}.
- Never estimate or calculate a price. If you are unsure, leave the model out.
- Write each price exactly as the page states it (a number, no currency symbol)."""


def _squash(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _dollar_amounts(text):
    out = []
    for raw in re.findall(r"\$\s*([\d,]*\.?\d+)", text or ""):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            pass
    return out


def _name_pattern(name):
    # "Claude Opus 4" must not match inside "Claude Opus 4.5", nor "gpt-5" inside "gpt-5.5".
    return re.compile(re.escape(name) + r"(?![\w.\-])", re.I)


BLOCK_LOOKAHEAD_LINES = 40


# The small model is told to skip image, audio and other non-text models and doesn't
# always (on the Google page it kept text-to-speech, music and embedding models), and a
# price check can't catch that, because those prices really are on the page. So code
# filters by name as well.
NON_TEXT_NAME = re.compile(r"tts|audio|speech|live|embed|image|imagen|veo|lyria|video|music|transcrib|realtime|moderation|whisper|dall|sora|davinci|babbage", re.I)


def verify_extracted(item, page_text):
    """Returns (ok, reason). The model's claims are only accepted if the page itself
    backs them. Code, not the model, finds the evidence: it locates the model's name on
    the page and checks that both prices appear as dollar amounts there. Two layouts are
    handled: one table row per model (both prices on the same line as the name), and a
    block per model (the prices on the lines that follow its heading).
    Earlier the model was asked to copy the evidence out itself, and it often copied it
    inexactly, so most correct prices were thrown away. This version doesn't depend on that.
    What it can't catch is the right number in the wrong column, or from a later table
    for the same model (a batch price read as the standard one)."""
    try:
        name = str(item["model"]).strip()
        raw_in, raw_out = item["input_per_1m"], item["output_per_1m"]
        if isinstance(raw_in, bool) or isinstance(raw_out, bool):
            return False, "price is not a number"
        a, b = float(raw_in), float(raw_out)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False, "missing or malformed field"
    if not (_plausible(a) and _plausible(b)):
        return False, "implausible price"
    if NON_TEXT_NAME.search(name):
        return False, "not a text-generation model (by name)"
    pattern = _name_pattern(name) if name else None
    page_lines = page_text.split("\n")
    hits = [i for i, line in enumerate(page_lines) if pattern and pattern.search(line)]
    if not hits:
        return False, "model name not on the page"
    row = next((i for i in hits if len(_dollar_amounts(page_lines[i])) >= 2), None)
    if row is not None:
        found = _dollar_amounts(page_lines[row])
    else:
        found = _dollar_amounts(" ".join(page_lines[hits[0]:hits[0] + BLOCK_LOOKAHEAD_LINES]))
    if not any(abs(a - v) < 1e-9 for v in found):
        return False, "input price not found with the model's name on the page"
    if not any(abs(b - v) < 1e-9 for v in found):
        return False, "output price not found with the model's name on the page"
    return True, ""



def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")


def discover_pricing_url(source):
    """The pricing page's markdown URL from the vendor's llms.txt, else the known fallback."""
    if source.get("llms_txt"):
        try:
            index = _fetch(source["llms_txt"])
            for text, url in re.findall(r"\[([^\]]*)\]\((https?://[^)\s]+)\)", index):
                if "pricing" in url.lower() or text.strip().lower() == "pricing":
                    return url
        except Exception:
            pass
    return source["fallback"]


def extract_with_llm(model_client, vendor, page_text):
    """Asks the small model to read the page, then keeps only what verifies.
    Returns (models, notes)."""
    response = model_client.chat.completions.create(
        model=EXTRACTION_MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT.replace("{today}", date.today().isoformat())},
            {"role": "user", "content": page_text[:MAX_PAGE_CHARS]},
        ],
    )
    claimed = json.loads(response.choices[0].message.content).get("models", [])
    if not isinstance(claimed, list):
        claimed = []
    models, rejected, seen = [], [], set()
    for item in claimed:
        ok, reason = verify_extracted(item, page_text)
        if not ok:
            rejected.append(f"{item.get('model') if isinstance(item, dict) else item}: {reason}")
            continue
        name = str(item["model"]).strip()
        if name in seen:
            continue
        seen.add(name)
        a, b = float(item["input_per_1m"]), float(item["output_per_1m"])
        models.append({"vendor": vendor, "model": name, "input_per_1m": a, "output_per_1m": b,
                       "listing": "current", "extraction": "llm_verified",
                       "flag": "output price is below input price; check the column" if b < a else None})
    notes = []
    if rejected:
        notes.append(f"{vendor}: rejected {len(rejected)} of {len(claimed)} extracted models ({'; '.join(rejected[:4])})")
    return models, notes


def fetch_prices_llm(model_client):
    as_of = datetime.now(timezone.utc).isoformat()
    models, problems = [], []
    for source in SOURCES:
        vendor = source["vendor"]
        try:
            url = discover_pricing_url(source)
            page = _fetch(url)
            # The reader's recall varies from run to run (the same page gave 13, 18 or 0 verified
            # models on different runs), which would make a model look newly published when it was
            # only missed before. So read the page up to three times and keep every model that
            # verified on any read; stop once a read adds nothing new. Each kept price was checked
            # against the page, so merging can't let an unchecked price in.
            by_name, notes = {}, []
            for attempt in range(EXTRACTION_ATTEMPTS):
                attempt_found, attempt_notes = extract_with_llm(model_client, vendor, page)
                added = 0
                for m in attempt_found:
                    key = normalize_model_name(m["model"])
                    if key not in by_name:
                        by_name[key] = m
                        added += 1
                    elif (by_name[key]["input_per_1m"], by_name[key]["output_per_1m"]) != (m["input_per_1m"], m["output_per_1m"]):
                        by_name[key]["flag"] = "two reads of the page gave different prices; check by hand"
                if attempt == 0:
                    notes = attempt_notes
                if attempt >= 1 and added == 0:
                    break
            found = list(by_name.values())
            problems.extend(notes)
            if len(found) < MIN_MODELS_PER_VENDOR:
                raise PriceSourceError(f"only {len(found)} models verified against the page")
            for m in found:
                m.update(source_url=url, as_of=as_of)
            models.extend(found)
        except Exception as e:
            problems.append(f"{vendor}: could not read prices ({type(e).__name__}: {e})")
    return {"models": models, "problems": problems, "as_of": as_of}


_VENDORS = (
    ("anthropic", ANTHROPIC_URL, parse_anthropic),
    ("openai", OPENAI_URL, parse_openai),
)


def fetch_prices(model_client=None):
    """Returns {"models": [...], "problems": [...], "as_of": iso}. With a model_client the
    generic path is used (llms.txt discovery, model extraction, code verification);
    without one, the hand-written parsers for Anthropic and OpenAI."""
    if model_client is not None:
        return fetch_prices_llm(model_client)
    as_of = datetime.now(timezone.utc).isoformat()
    models, problems = [], []
    for vendor, url, parser in _VENDORS:
        try:
            found = parser(_page_text(url))
            if len(found) < MIN_MODELS_PER_VENDOR:
                raise PriceSourceError(f"only {len(found)} models parsed; the page layout may have changed")
            for m in found:
                m.update(source_url=url, as_of=as_of)
            models.extend(found)
        except Exception as e:
            problems.append(f"{vendor}: could not read prices ({type(e).__name__}: {e})")
    return {"models": models, "problems": problems, "as_of": as_of}


# --------------------------------------------------------------------------
# Two completeness checks. Neither adds a price; each only raises a flag.
#   1. Compare the vendor's models page with the last refresh: a model that is newly
#      listed there, or no longer listed.
#   2. Compare with the vendor's own models page: a model the vendor lists there that
#      has no price in our list, meaning the reader missed a row or no price is published.
# --------------------------------------------------------------------------

MODELS_PAGES = {
    "anthropic": "https://docs.anthropic.com/en/docs/about-claude/models/overview.md",
    "openai": "https://platform.openai.com/docs/models.md",
    "google": "https://ai.google.dev/gemini-api/docs/models.md.txt",
}
# Names are matched by their shape, not by asking a model, so this check can't be
# wrong in the same way the price reader can.
MODEL_NAME_PATTERNS = {
    "anthropic": re.compile(r"Claude [A-Z][a-z]+ \d+(?:\.\d+)?"),
    "openai": re.compile(r"\b(?:gpt-[a-z0-9][\w.\-]*|o\d[\w\-]*)"),
    "google": re.compile(r"\bgemini-[a-z0-9][\w.\-]*"),
}


def normalize_model_name(name):
    """Lowercase, without a trailing ".md" (OpenAI's page links to model pages), a
    parenthetical like "(<272K context length)", or a dated snapshot suffix."""
    name = re.sub(r"\s*\(.*?\)", "", str(name)).strip().lower()
    name = re.sub(r"\.md$", "", name)
    name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)
    return re.sub(r"[\s.\-]+$", "", name)


def parse_listed_models(vendor, page_text):
    """Names the vendor's models page mentions, text models only."""
    names = {normalize_model_name(n) for n in MODEL_NAME_PATTERNS[vendor].findall(page_text)}
    return {n for n in names if n and not NON_TEXT_NAME.search(n)}


def fetch_listed_models():
    listed, problems = {}, []
    for vendor, url in MODELS_PAGES.items():
        try:
            listed[vendor] = parse_listed_models(vendor, _fetch(url))
        except Exception as e:
            problems.append(f"{vendor}: could not read the models page ({type(e).__name__}: {e})")
    return listed, problems


def unpriced_models(listed_by_vendor, models):
    """Models a vendor's models page lists that our price list has no price for."""
    priced = {}
    for m in models:
        priced.setdefault(m["vendor"], set()).add(normalize_model_name(m["model"]))
    return {vendor: sorted(names - priced.get(vendor, set()))
            for vendor, names in listed_by_vendor.items() if names - priced.get(vendor, set())}


def diff_listed_models(previous, listed):
    """Compares the model names on each vendor's models page with the last refresh.
    `new` means never listed in any earlier refresh. Names come from a fixed pattern on the
    page, not from the price reader, because the reader's recall varies from run to run (on
    the Google page it found 7, 7 and 12 models on three consecutive runs), and a diff of
    its output flagged old models as new. The names on the models page don't change
    between reads, so a difference is real."""
    if not previous or not previous.get("listed_models"):
        return {"new_on_models_page": [], "no_longer_listed": [], "baseline": True}
    seen = {(v, n) for v, names in previous["listed_models"].items() for n in names}
    seen |= {tuple(x) for x in previous.get("seen_listed", [])}
    last = {(v, n) for v, names in previous["listed_models"].items() for n in names}
    fresh = {(v, n) for v, names in listed.items() for n in names}
    vendors_read = set(listed)  # a models page that couldn't be read isn't "everything removed"
    return {
        "new_on_models_page": sorted(f"{v}: {n}" for v, n in fresh - seen),
        "no_longer_listed": sorted(f"{v}: {n}" for v, n in last - fresh if v in vendors_read),
        "baseline": False,
    }


def add_completeness_checks(result, previous=None):
    """Adds `changes`, `listed_but_unpriced`, `listed_models` and `seen_listed` to a fresh
    price result. Neither check changes a price; both are flags for a person."""
    listed, problems = fetch_listed_models()
    result["problems"] = list(result["problems"]) + problems
    result["changes"] = diff_listed_models(previous, listed)
    result["listed_models"] = {v: sorted(names) for v, names in listed.items()}
    seen = {(v, n) for v, names in listed.items() for n in names}
    seen |= {tuple(x) for x in (previous or {}).get("seen_listed", [])}
    seen |= {(v, n) for v, names in (previous or {}).get("listed_models", {}).items() for n in names}
    result["seen_listed"] = [list(x) for x in sorted(seen)]
    result["listed_but_unpriced"] = unpriced_models(listed, result["models"])
    return result


def get_prices(force_refresh=False, model_client=None):
    """Cached for 24 hours so one request doesn't mean two page downloads. If a fresh
    fetch fails entirely, an older cache is used and marked stale rather than guessing."""
    cached = None
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            cached = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    if cached and not force_refresh and time.time() - cached.get("fetched_epoch", 0) < CACHE_TTL_SECONDS:
        return cached
    fresh = fetch_prices(model_client)
    if fresh["models"]:
        add_completeness_checks(fresh, cached)
        fresh["fetched_epoch"] = time.time()
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(fresh, f, indent=2)
        return fresh
    if cached:
        cached["problems"] = fresh["problems"] + ["using an older saved price list; it may be out of date"]
        cached["stale"] = True
        return cached
    return fresh
