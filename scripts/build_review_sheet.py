"""
Builds the review sheet a person fills in: evals/review_sheet.xlsx.

Usage:
    python build_review_sheet.py

Reads evals/golden_hhh_cases.json and the saved runs in evals/golden_runs/ (see run_golden_cases.py).

Sheet "Review": one row per case that has a saved run. Columns for the PM to fill in are the last three (verdict,
what was wrong or right, edited expected behaviour). The code check and the judge scores are deliberately NOT here,
so the reviewer judges the reply first. For Regulations replies the exact text of every provision the reply cites,
and of Article 113 when the reply gives a date, is shown from the same index the agent searched, so the answer can be
read against the Act itself. The agent's citations are what it claims to rely on; which passages the search actually
returned to it is not recorded, and the sheet says so.

Sheet "Run details": reference only (run id, time taken, date).
"""
import json
import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

import act_text
import ui_payload

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_FILE = os.path.join(HERE, "..", "evals", "golden_hhh_cases.json")
RUNS_DIR = os.path.join(HERE, "..", "evals", "golden_runs")
OUT_FILE = os.path.join(HERE, "..", "evals", "review_sheet.xlsx")
MAX_CELL = 30000  # Excel allows 32,767 characters in a cell

AGENT_LABEL = {"regulations": "Regulations", "competitive_analysis": "Competitive", "model_stack_analysis": "Model Stack"}
ARTICLE_113 = {"type": "article", "article_number": 113}


def reg_output(record):
    result = ((record.get("agent_results") or {}).get("regulations") or {})
    return result.get("output") if result.get("status") == "ok" else None


def source_text(record):
    """The Act text behind a Regulations answer, or a plain note when there is none."""
    output = reg_output(record)
    if not output:
        if "regulations" in (record.get("agents_run") or []):
            return "Regulations ran but returned no answer."
        return "Regulations did not run for this request, so nothing in the reply comes from the Act."
    citations = list(output.get("citations") or [])
    if re.search(r"Article 113", output.get("applicability_status") or ""):
        citations.append(ARTICLE_113)
    if not citations:
        return "The answer cites no provision of the Act."
    quotes = act_text.quote_citations(citations, max_chars=2500)
    if not quotes:
        return "The answer cites " + ", ".join(act_text.citation_label(c) for c in citations) + ", but the text could not be fetched."
    parts = [f"{q['label']}" + (f" ({q['title']})" if q.get("title") else "") + f"\n{q['text']}" for q in quotes]
    cited = ", ".join(act_text.citation_label(c) for c in output.get("citations") or []) or "none"
    head = f"Provisions the answer cites: {cited}. (Which passages the search returned to the agent is not recorded.)"
    return head + "\n\n" + "\n\n".join(parts)


def guardrail_text(record):
    summary = ui_payload.guardrail_summary(record.get("guardrail_events"))
    if not summary["lines"]:
        return f"{summary['checked']} checks ran; none changed anything."
    return f"{summary['checked']} checks ran; {summary['fired']} changed something:\n- " + "\n- ".join(summary["lines"])


def cell(text):
    text = "" if text is None else str(text)
    return text if len(text) <= MAX_CELL else text[:MAX_CELL] + " …(cut)"


import re  # noqa: E402  (used by source_text)


def main():
    load_dotenv(os.path.join(HERE, ".env"))
    cases = json.load(open(CASES_FILE, encoding="utf-8"))["cases"]
    wb = Workbook()
    review, details = wb.active, wb.create_sheet("Run details")
    review.title = "Review"

    headers = ["Case ID", "Agent", "Type", "What the PM typed", "What ShipWise replied", "Which agents ran",
               "Guardrails that fired", "Expected behaviour (draft, edit if wrong)", "Source check: the Act text the answer cites",
               "YOUR VERDICT (pass / fail / unsure)", "What was right or wrong (one sentence)", "Edited expected behaviour (only if you changed it)"]
    review.append(headers)
    details.append(["Case ID", "Run id", "Seconds", "Created (UTC)", "Request type", "Error", "Trace ids"])

    for case in cases:
        path = os.path.join(RUNS_DIR, f"{case['id']}.json")
        if not os.path.exists(path):
            continue
        record = json.load(open(path, encoding="utf-8"))["record"]
        expected = f"{case['reviewer_question']}\n\nWhy it matters: {case['why_it_matters']}"
        review.append([
            case["id"], AGENT_LABEL.get(case["component"], case["component"]), case["dimension"], cell(case["message"]),
            cell(record.get("reply") or f"(no reply; error: {record.get('error')})"),
            ", ".join(record.get("agents_run") or []) or "none", cell(guardrail_text(record)), cell(expected),
            cell(source_text(record)), None, None, None])
        details.append([case["id"], record.get("id"), record.get("duration_seconds"), record.get("created_at"),
                        record.get("request_type"), record.get("error"), ", ".join(record.get("trace_ids") or [])])

    widths = [18, 14, 10, 40, 80, 18, 40, 50, 80, 22, 40, 40]
    for i, w in enumerate(widths, 1):
        review.column_dimensions[get_column_letter(i)].width = w
    head_fill, input_fill = PatternFill("solid", fgColor="D9D9D9"), PatternFill("solid", fgColor="FFF2CC")
    for c in review[1]:
        c.font, c.fill = Font(bold=True), (input_fill if c.column >= 10 else head_fill)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for row in review.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if c.column >= 10:
                c.fill = input_fill
    review.freeze_panes = "B2"
    validation = DataValidation(type="list", formula1='"pass,fail,unsure"', allow_blank=True)
    review.add_data_validation(validation)
    validation.add(f"J2:J{max(review.max_row, 2)}")
    for i, w in enumerate([18, 38, 10, 28, 20, 40, 30], 1):
        details.column_dimensions[get_column_letter(i)].width = w
    for c in details[1]:
        c.font = Font(bold=True)

    wb.save(OUT_FILE)
    print(f"{review.max_row - 1} case(s) written to {os.path.abspath(OUT_FILE)}")


if __name__ == "__main__":
    main()
