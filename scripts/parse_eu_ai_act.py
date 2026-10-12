"""
Parses the EU AI Act (EUR-Lex HTML, CELEX 32024R1689) into paragraph-level
chunk records ready for embedding.

All structure (article number, recital number, annex, paragraph number) is
read directly from the document's own ELI markup (div ids like "art_6",
"rct_27", "anx_III", "006.002") - nothing is inferred or guessed. This approach was chosen over
Foundry's built-in chunking or an LLM-based parser.

Usage:
    python parse_eu_ai_act.py <path-to-raw-html> <output-json-path>
"""

import sys
import re
import json
from bs4 import BeautifulSoup

CELEX_ID = "32024R1689"
OJ_REFERENCE = "OJ L, 2024/1689, 12.7.2024"
OJ_PDF_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689"
HTML_BASE_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689"

EXPECTED_ARTICLE_COUNT = 113
EXPECTED_RECITAL_COUNT = 180
EXPECTED_ANNEX_COUNT = 13

XREF_PATTERN = re.compile(
    r"Article\s+\d+(?:\(\d+\))?(?:\([a-z]\))?|Annex\s+[IVXLC]+",
    re.IGNORECASE,
)


def clean_text(el):
    """Flattens an element's text, collapsing whitespace/newlines from the
    pretty-printed source into a single readable string."""
    return " ".join(el.get_text(" ", strip=True).split())


def extract_xrefs(text):
    return sorted(set(m.strip() for m in XREF_PATTERN.findall(text)))


def build_record(**fields):
    base = {
        "id": None,
        "regulation": "EU_AI_Act",
        "celex_id": CELEX_ID,
        "oj_reference": OJ_REFERENCE,
        "article_number": None,
        "article_title": None,
        "paragraph_number": None,
        "recital_number": None,
        "is_recital": False,
        "is_annex": False,
        "annex_id": None,
        "annex_item_number": None,
        "annex_sub_point": None,
        "xrefs": [],
        "text": "",
        "parent_article_text": None,
        "source_url": OJ_PDF_URL,
    }
    base.update(fields)
    if not base["xrefs"]:
        base["xrefs"] = extract_xrefs(base["text"])
    return base


def parse_articles(soup):
    records = []
    articles = soup.find_all("div", class_="eli-subdivision", id=re.compile(r"^art_\d+$"))
    for art in articles:
        number = int(art["id"].split("_")[1])
        title_el = art.find("p", class_="oj-sti-art")
        article_title = clean_text(title_el) if title_el else ""
        full_article_text = clean_text(art)
        anchor_url = f"{HTML_BASE_URL}#{art['id']}"

        para_id_pattern = re.compile(rf"^{number:03d}\.\d+$")
        para_divs = [d for d in art.find_all("div", id=para_id_pattern, recursive=False)]

        if not para_divs:
            # short article, no numbered sub-paragraphs - whole body is one chunk
            records.append(build_record(
                id=f"art_{number}",
                article_number=number,
                article_title=article_title,
                text=full_article_text,
                source_url=anchor_url,
            ))
        else:
            for pd in para_divs:
                para_num = int(pd["id"].split(".")[1])
                records.append(build_record(
                    id=f"art_{number}_para_{para_num}",
                    article_number=number,
                    article_title=article_title,
                    paragraph_number=para_num,
                    text=clean_text(pd),
                    parent_article_text=full_article_text,
                    source_url=anchor_url,
                ))
    return records, len({r["article_number"] for r in records})


def parse_recitals(soup):
    records = []
    recitals = soup.find_all("div", class_="eli-subdivision", id=re.compile(r"^rct_\d+$"))
    for rct in recitals:
        number = int(rct["id"].split("_")[1])
        records.append(build_record(
            id=f"rct_{number}",
            recital_number=number,
            is_recital=True,
            text=clean_text(rct),
            source_url=f"{HTML_BASE_URL}#{rct['id']}",
        ))
    return records, len({r["recital_number"] for r in records})


def parse_annexes(soup):
    records = []
    annexes = soup.find_all("div", class_="eli-container", id=re.compile(r"^anx_[IVXLC]+$"))
    for anx in annexes:
        annex_id = anx["id"].split("_")[1]
        anchor_url = f"{HTML_BASE_URL}#{anx['id']}"
        top_level_tables = anx.find_all("table", recursive=False)

        if not top_level_tables:
            # short annex with no numbered items - whole body is one chunk
            records.append(build_record(
                id=f"anx_{annex_id}",
                is_annex=True,
                annex_id=annex_id,
                text=clean_text(anx),
                source_url=anchor_url,
            ))
            continue

        for table in top_level_tables:
            tbody = table.find("tbody", recursive=False)
            rows = tbody.find_all("tr", recursive=False) if tbody else []
            if not rows:
                continue
            first_row_cells = rows[0].find_all("td", recursive=False)
            if len(first_row_cells) < 2:
                continue
            item_label = clean_text(first_row_cells[0])
            item_match = re.match(r"^(\d+)\.?$", item_label)
            item_number = int(item_match.group(1)) if item_match else None
            content_cell = first_row_cells[1]

            # Only direct-child <p> tags - excludes text inside nested
            # sub-point tables, isolating the item's own "stem" sentence
            # (e.g. "Biometrics, in so far as...:") from its sub-points.
            intro_text = " ".join(clean_text(p) for p in content_cell.find_all("p", recursive=False))
            sub_tables = content_cell.find_all("table", recursive=False)

            if not sub_tables:
                # no lettered sub-points - the whole item is one coherent chunk
                records.append(build_record(
                    id=f"anx_{annex_id}_item_{item_number if item_number is not None else len(records)}",
                    is_annex=True,
                    annex_id=annex_id,
                    annex_item_number=item_number,
                    text=f"{item_label} {intro_text}",
                    source_url=anchor_url,
                ))
                continue

            # Lettered sub-points (a)/(b)/(c) are distinct, unrelated AI
            # system categories, not steps of one rule - chunk each on its
            # own so a query about one doesn't get diluted by the others.
            # Annex III items are chunked at the sub-point level.
            for sub_table in sub_tables:
                sub_tbody = sub_table.find("tbody", recursive=False)
                sub_rows = sub_tbody.find_all("tr", recursive=False) if sub_tbody else []
                if not sub_rows:
                    continue
                sub_cells = sub_rows[0].find_all("td", recursive=False)
                if len(sub_cells) < 2:
                    continue
                sub_label = clean_text(sub_cells[0])
                sub_match = re.match(r"^\(([a-z])\)$", sub_label)
                sub_letter = sub_match.group(1) if sub_match else sub_label
                sub_text = clean_text(sub_cells[1])
                records.append(build_record(
                    id=f"anx_{annex_id}_item_{item_number}_{sub_letter}",
                    is_annex=True,
                    annex_id=annex_id,
                    annex_item_number=item_number,
                    annex_sub_point=sub_letter,
                    # Deliberately bare - no item_label/intro_text prefix.
                    # Citation comes from the metadata fields above, not
                    # from this text; prepending the parent item's category
                    # (e.g. "Biometrics...") measurably hurt embedding
                    # similarity for queries about this specific sub-point.
                    # Annex III items are chunked at the sub-point level.
                    text=sub_text,
                    source_url=anchor_url,
                ))
    return records, len({r["annex_id"] for r in records})


def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_eu_ai_act.py <input.html> <output.json>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    with open(input_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    article_records, article_count = parse_articles(soup)
    recital_records, recital_count = parse_recitals(soup)
    annex_records, annex_count = parse_annexes(soup)

    all_records = article_records + recital_records + annex_records

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(all_records)} chunk records to {output_path}")
    print()
    print("Structural integrity check :")
    print(f"  Articles: {article_count} (expected {EXPECTED_ARTICLE_COUNT}) "
          f"{'PASS' if article_count == EXPECTED_ARTICLE_COUNT else 'FAIL'}")
    print(f"  Recitals: {recital_count} (expected {EXPECTED_RECITAL_COUNT}) "
          f"{'PASS' if recital_count == EXPECTED_RECITAL_COUNT else 'FAIL'}")
    print(f"  Annexes:  {annex_count} (expected {EXPECTED_ANNEX_COUNT}) "
          f"{'PASS' if annex_count == EXPECTED_ANNEX_COUNT else 'FAIL'}")

    if article_count != EXPECTED_ARTICLE_COUNT or recital_count != EXPECTED_RECITAL_COUNT or annex_count != EXPECTED_ANNEX_COUNT:
        sys.exit(1)


if __name__ == "__main__":
    main()
