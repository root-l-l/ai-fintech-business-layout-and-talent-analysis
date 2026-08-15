"""Extract audit-friendly key financial metric lines from early annual-report pages."""

from __future__ import annotations

import re

import pdfplumber
import pandas as pd

from common import ensure_directories, repo_path


LABELS = {
    "revenue": "营业收入",
    "net_profit_attributable": "归属于上市公司股东的净利润",
    "total_assets": "资产总额",
}


def main() -> int:
    manifest = pd.read_csv(repo_path("data", "reference", "annual_report_manifest.csv"), dtype=str, keep_default_na=False)
    rows = []
    for _, record in manifest[manifest["mapping_status"] == "mapped"].iterrows():
        path = repo_path(*record["local_path"].split("/"))
        found: set[str] = set()
        with pdfplumber.open(path) as document:
            for page_number, page in enumerate(document.pages[:18], start=1):
                lines = [re.sub(r"\s+", " ", line).strip() for line in (page.extract_text() or "").splitlines()]
                for metric, label in LABELS.items():
                    if metric in found:
                        continue
                    matches = [line for line in lines if label in line]
                    if matches:
                        rows.append({"company_code": record["company_code"], "company_name": record["company_name"], "metric": metric, "page_number": page_number, "metric_line_raw": matches[0], "selection_status": "raw_line_requires_human_value_selection"})
                        found.add(metric)
    output = repo_path("outputs", "tables", "financial_summary_lines.csv")
    ensure_directories([output.parent])
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(rows)} financial summary lines to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
