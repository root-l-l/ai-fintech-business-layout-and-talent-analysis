"""Extract review-required financial metric candidates from annual-report PDFs.

This script does not claim that regex hits are audited financial values. It
retains page text snippets for human verification before any numeric comparison.
"""

from __future__ import annotations

import re
import argparse

import pdfplumber
import pandas as pd

from common import ensure_directories, repo_path


METRICS = {
    "total_assets": ["资产总额", "资产合计"],
    "revenue": ["营业收入"],
    "net_profit_attributable": ["归属于上市公司股东的净利润", "归属于母公司股东的净利润"],
}
NUMBER = re.compile(r"(?<!\d)(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?!\d)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-code", help="Extract one company only and append/replace its candidate rows.")
    args = parser.parse_args()
    manifest = pd.read_csv(repo_path("data", "reference", "annual_report_manifest.csv"), dtype=str, keep_default_na=False)
    manifest = manifest[manifest["mapping_status"] == "mapped"]
    if args.company_code:
        manifest = manifest[manifest["company_code"] == args.company_code]
        if manifest.empty:
            raise ValueError(f"No mapped annual report for {args.company_code}.")
    rows = []
    for _, record in manifest.iterrows():
        path = repo_path(*record["local_path"].split("/"))
        with pdfplumber.open(path) as document:
            for page_number, page in enumerate(document.pages, start=1):
                text = " ".join((page.extract_text() or "").split())
                for metric, labels in METRICS.items():
                    if not any(label in text for label in labels):
                        continue
                    values = NUMBER.findall(text)
                    rows.append({"company_code": record["company_code"], "company_name": record["company_name"], "metric": metric, "page_number": page_number, "matched_label": next(label for label in labels if label in text), "numeric_tokens_on_page": "|".join(values[:25]), "page_text_snippet": text[:1000], "review_status": "candidate_requires_manual_selection"})
    output = repo_path("outputs", "tables", "financial_metric_candidates.csv")
    ensure_directories([output.parent])
    result = pd.DataFrame(rows)
    if args.company_code and output.exists():
        existing = pd.read_csv(output, dtype=str, keep_default_na=False)
        result = pd.concat([existing[existing["company_code"] != args.company_code], result], ignore_index=True)
    result.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(rows)} financial metric candidates to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
