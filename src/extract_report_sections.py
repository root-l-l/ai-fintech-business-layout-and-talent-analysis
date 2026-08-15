"""Extract auditable text sections from locally saved annual-report PDFs."""

from __future__ import annotations

import re
import sys
import argparse

import pdfplumber
import pandas as pd

from common import ensure_directories, read_csv, repo_path

SECTION_PATTERNS = {
    "business_overview": [r"业务概要", r"主营业务分析"],
    "core_competitiveness": [r"核心竞争力", r"核心竞争优势"],
    "r_and_d": [r"研发投入", r"研发人员"],
    "management_discussion": [r"管理层讨论与分析", r"经营情况讨论与分析"],
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_text(path: str) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append((number, text))
    return pages


def find_sections(pages: list[tuple[int, str]]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for section_name, patterns in SECTION_PATTERNS.items():
        matches = [(number, text) for number, text in pages if any(re.search(pattern, text) for pattern in patterns)]
        if not matches:
            results.append({"section": section_name, "page_start": "", "page_end": "", "text": "", "extraction_note": "heading_not_found"})
            continue
        start_page = matches[0][0]
        selected = [(number, text) for number, text in pages if start_page <= number < start_page + 12]
        results.append({"section": section_name, "page_start": str(selected[0][0]), "page_end": str(selected[-1][0]), "text": "\n".join(text for _, text in selected), "extraction_note": "windowed_extraction_review_required"})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-code", help="Extract one company only and replace only its existing rows.")
    args = parser.parse_args()
    manifest_path = repo_path("data", "reference", "annual_report_manifest.csv")
    if manifest_path.exists():
        registry = read_csv(manifest_path)
        registry = registry[registry["mapping_status"] == "mapped"].copy()
    else:
        registry = read_csv(repo_path("data", "reference", "source_registry.csv"))
        registry = registry[registry["source_type"] == "annual_report"].copy()
    rows: list[dict[str, str]] = []
    reports = registry.copy()
    if args.company_code:
        reports = reports[reports["company_code"] == args.company_code]
        if reports.empty:
            raise ValueError(f"No annual report registered for company code {args.company_code}.")
    for _, record in reports.iterrows():
        path = repo_path(*record["local_path"].split("/"))
        if not path.exists():
            print(f"SKIP missing report: {path}")
            continue
        try:
            sections = find_sections(extract_pdf_text(str(path)))
        except Exception as error:
            print(f"FAILED {path}: {error}", file=sys.stderr)
            continue
        for section in sections:
            rows.append({"company_code": record["company_code"], "source_id": record["source_id"], **section})
    columns = ["company_code", "source_id", "section", "page_start", "page_end", "text", "extraction_note"]
    output = repo_path("data", "processed", "annual_report_sections.csv")
    ensure_directories([output.parent])
    result = pd.DataFrame(rows, columns=columns)
    if output.exists():
        existing = pd.read_csv(output, dtype=str, keep_default_na=False)
        if args.company_code:
            result = pd.concat([existing[existing["company_code"] != args.company_code], result], ignore_index=True)
        else:
            result = pd.concat([existing[~existing["company_code"].isin(result["company_code"])], result], ignore_index=True)
    result.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(rows)} section row(s) to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
