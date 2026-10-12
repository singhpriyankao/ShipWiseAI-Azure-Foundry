"""
Checks on the Competitive Analysis agent's sources, run in code after the agent
returns.

The agent writes a note and a source for each of accuracy, latency and pricing. With the v3
JSON agent the address was typed by the model (and was often invented). With v4
the addresses come from the search tool's own citations (competitive_parse.py), and a line can
cite several pages. Either way, what code can do is check the cited page itself:

1. A claim with no source URL must say so. If the URL is null and the note doesn't start
   with "No source found:", the note is replaced (the agent broke its own rule on the
   first v3 run, 2026-10-04).
2. A cited page is loaded. It must mention the vendor or product, and every price or
   percentage in the note must appear on the page. A page that loads and doesn't
   support the claim replaces the note, and so does a page that doesn't exist (404 or
   410, which the first live run found on a cited Zendesk URL, or a web address
   that doesn't exist, which the end-to-end run found on every URL it cited). A page that can't be read (blocked, timed out,
   nearly empty because it is built with JavaScript) keeps the note and is marked
   "could not be checked", because a blocked page says nothing about whether the claim
   is true.

Each competitor gets a `source_checks` dict (accuracy / latency / pricing -> status).
A replaced note is kept under `original_<field>` so nothing is lost.

URLs come from a model, so before fetching, each one is checked to be plain http(s)
and to resolve to a public address (no localhost, private or link-local addresses),
and redirects are followed by hand with the same check at every hop.
"""
import ipaddress
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

FIELDS = ("accuracy", "latency", "pricing")
NO_SOURCE_PREFIX = "No source found:"
NO_SOURCE_NOTE = NO_SOURCE_PREFIX + " the agent gave no source for this claim, so it is not shown."
UNSUPPORTED_NOTE = "No verified source: the cited page does not support this claim ({reason}), so it is not shown."

PAGE_NOT_FOUND = "page not found"
FETCH_TIMEOUT_SECONDS = 8
MAX_REDIRECTS = 3
MAX_PAGE_BYTES = 1_000_000
MIN_READABLE_CHARS = 1500  # shorter than this and the page is probably built by JavaScript
USER_AGENT = "Mozilla/5.0 (compatible; ShipWiseSourceCheck/1.0)"

_FIGURE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?\s?%")


def _resolve(host):
    """The addresses a host name points to, or None if the name doesn't exist."""
    try:
        return [info[4][0] for info in socket.getaddrinfo(host, None)]
    except socket.gaierror:
        return None


def _is_public_address(addresses):
    for addr in addresses:
        ip = ipaddress.ip_address(addr)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False
    return bool(addresses)


def _online():
    """True if a well-known host name resolves, so a name that fails to resolve is the
    cited site's fault and not this machine being offline."""
    return _resolve("example.com") is not None


def fetch_page_text(url):
    """Returns (status, text). status is "ok" or a short reason the page couldn't be read.
    Never raises."""
    try:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            parsed = urlparse(current)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                return "not a web address", ""
            addresses = _resolve(parsed.hostname)
            if addresses is None:
                # A host name that doesn't exist means the cited page doesn't exist, which
                # is evidence against the claim. If we are offline it is not.
                return (PAGE_NOT_FOUND, "") if _online() else ("no internet connection", "")
            if not _is_public_address(addresses):
                return "address not allowed", ""
            resp = requests.get(current, timeout=FETCH_TIMEOUT_SECONDS, allow_redirects=False,
                                headers={"User-Agent": USER_AGENT}, stream=True)
            if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                current = urljoin(current, resp.headers["Location"])
                continue
            if resp.status_code in (404, 410):
                return PAGE_NOT_FOUND, ""
            if resp.status_code != 200:
                return f"page returned {resp.status_code}", ""
            body = resp.raw.read(MAX_PAGE_BYTES, decode_content=True)
            html = body.decode(resp.encoding or "utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = " ".join(soup.get_text(" ").split())
            if len(text) < MIN_READABLE_CHARS:
                return "page has too little readable text", ""
            return "ok", text
        return "too many redirects", ""
    except Exception as e:  # network errors are expected; none may stop the run
        return f"could not load ({type(e).__name__})", ""


def _normalise(text):
    return text.lower().replace(",", "").replace("$ ", "$").replace(" %", "%")


def figures_in(note):
    """Prices and percentages stated in a note, e.g. ["$74", "90%"]."""
    return [_normalise(m) for m in _FIGURE.findall(note or "")]


def judge_page(competitor, note, page_text):
    """Returns None if the page supports the note, else a short reason."""
    page = _normalise(page_text)
    names = [n.lower() for n in (competitor.get("vendor"), competitor.get("product")) if n]
    if names and not any(n in page for n in names):
        return "the page does not mention the vendor or product"
    missing = [f for f in figures_in(note) if f not in page]
    if missing:
        return "the page does not contain " + ", ".join(missing)
    return None


def check_competitor_sources(output, fetch=fetch_page_text):
    """Applies the source rules to the agent's parsed output and returns it (changed in place).
    `fetch` is injectable so tests don't touch the network."""
    competitors = output.get("competitors") if isinstance(output, dict) else None
    if not competitors:
        return output

    jobs = []  # (competitor, field, [urls])
    for comp in competitors:
        comp["source_checks"] = {}
        for field in FIELDS:
            note = comp.get(f"{field}_notes") or ""
            # v4 lines can cite several pages (`_source_urls`); a v3 claim has the one `_source_url`.
            single = comp.get(f"{field}_source_url")
            urls = [u for u in (comp.get(f"{field}_source_urls") or []) if u] or ([single] if single else [])
            if not urls:
                if note.startswith(NO_SOURCE_PREFIX):
                    comp["source_checks"][field] = "no_source_stated"
                else:
                    comp[f"original_{field}_notes"] = note
                    comp[f"{field}_notes"] = NO_SOURCE_NOTE
                    comp["source_checks"][field] = "no_source_note_replaced"
            else:
                jobs.append((comp, field, urls))

    unique_urls = {url for _, _, urls in jobs for url in urls}
    with ThreadPoolExecutor(max_workers=6) as pool:
        pages = dict(zip(unique_urls, pool.map(fetch, unique_urls)))

    for comp, field, urls in jobs:
        note = comp.get(f"{field}_notes") or ""
        outcomes = []  # (url, kind, detail)
        for url in urls:
            status, text = pages[url]
            if status == PAGE_NOT_FOUND:
                outcomes.append((url, "not_found", None))
            elif status != "ok":
                outcomes.append((url, "unreadable", status))
            else:
                reason = judge_page(comp, note, text)
                outcomes.append((url, "verified" if reason is None else "unsupported", reason))

        # One cited page that supports the claim is enough; the address kept is that page's.
        supported = next((o for o in outcomes if o[1] == "verified"), None)
        if supported:
            comp[f"{field}_source_url"] = supported[0]
            comp["source_checks"][field] = "verified"
            continue
        # A page that loads and doesn't support the claim is evidence against it. So is a page that doesn't
        # exist, when every cited page is like that. Unlike a blocked page (403, timeout), these say something
        # about the claim.
        unsupported = next((o for o in outcomes if o[1] == "unsupported"), None)
        if unsupported or all(o[1] == "not_found" for o in outcomes):
            comp[f"original_{field}_notes"] = note
            comp[f"{field}_notes"] = UNSUPPORTED_NOTE.format(reason=unsupported[2] if unsupported else "the cited page does not exist")
            comp["source_checks"][field] = "page_does_not_support_claim"
        else:
            first = next(o for o in outcomes if o[1] == "unreadable")
            comp["source_checks"][field] = f"could_not_be_checked ({first[2]})"
    return output
