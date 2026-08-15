"""Extract comparable 2025 financial metrics from annual-report summary tables.

The extractor searches only the opening pages where listed companies normally
publish their three-year key accounting data.  It retains page-level excerpts
and deliberately labels every row as machine-extracted so a researcher can
audit it against the source PDF before submission.
"""

from __future__ import annotations

import re

import pdfplumber
import pandas as pd

from common import ensure_directories, repo_path


METRICS = {
    "revenue_yuan": ("营业收入", "营业总收入"),
    # PDF table extraction often inserts numeric cells between “股东” and
    # “净利润”; the leading ownership label is stable and appears first.
    "net_profit_attributable_yuan": ("归属于上市公司股东", "归属于母公司股东", "归属于母公司所有者"),
    "total_assets_yuan": ("资产总额", "总资产", "资产总计"),
}
NUMBER = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?")


def flattened(text: str) -> str:
    """Remove PDF-inserted whitespace so split table labels can be matched."""

    return re.sub(r"\s+", "", text).strip()


def page_score(text: str) -> int:
    flat = flattened(text)
    return sum(any(alias in flat for alias in aliases) for aliases in METRICS.values())


def unit_multiplier(text: str) -> int:
    return 10_000 if "单位：万元" in text or "单位:万元" in text else 1


def compact_cell(value: object) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def metric_for_row(label: str) -> str | None:
    if ("营业收入" in label or "营业总收入" in label) and "扣除" not in label:
        return "revenue_yuan"
    if ("归属于上市公司股东" in label or "归属于母公司" in label) and "净利润" in label and "扣除" not in label:
        return "net_profit_attributable_yuan"
    if "资产总额" in label or label == "总资产" or label == "资产总计":
        return "total_assets_yuan"
    return None


def table_metrics(page: pdfplumber.page.Page) -> tuple[dict[str, tuple[float, str]], int] | None:
    """Return 2025 table cells where the PDF exposes a usable grid."""

    table = page.extract_table()
    if not table:
        return None
    rows: dict[str, tuple[float, str]] = {}
    for row in table:
        if not row:
            continue
        metric = metric_for_row(compact_cell(row[0]))
        if metric is None or len(row) < 2:
            continue
        raw_value = compact_cell(row[1])
        matches = NUMBER.findall(raw_value)
        if matches:
            rows[metric] = (float(matches[0].replace(",", "")), raw_value)
    return rows, len(rows)


def first_number_after_alias(text: str, aliases: tuple[str, ...]) -> tuple[float | None, str]:
    flat = flattened(text)
    positions = [(flat.find(alias), alias) for alias in aliases if flat.find(alias) >= 0]
    if not positions:
        return None, ""
    position, alias = min(positions)
    excerpt = flat[position : position + 220]
    numbers = NUMBER.findall(excerpt[len(alias) :])
    if not numbers:
        return None, excerpt
    return float(numbers[0].replace(",", "")), excerpt


def extract_company_metrics(record: pd.Series) -> list[dict[str, object]]:
    path = repo_path(*record["local_path"].split("/"))
    with pdfplumber.open(path) as report:
        candidates = []
        for index, page in enumerate(report.pages[:20], start=1):
            page_text = page.extract_text() or ""
            extracted_table = table_metrics(page)
            table_values, table_score = extracted_table if extracted_table else ({}, 0)
            # Tables preserve columns; use them over text when they contain all
            # three metrics. Text remains a transparent fallback for PDFs with
            # no extractable table grid.
            candidates.append((index, page_text, table_values, table_score, page_score(page_text)))
    best_page, page_text, table_values, table_score, text_score = max(candidates, key=lambda item: (item[3], item[4], -item[0]))
    score = table_score if table_score == 3 else text_score
    multiplier = unit_multiplier(page_text)
    rows: list[dict[str, object]] = []
    for metric, aliases in METRICS.items():
        if metric in table_values:
            value, excerpt = table_values[metric]
        else:
            value, excerpt = first_number_after_alias(page_text, aliases)
        value_yuan = value * multiplier if value is not None else None
        is_plausible = value_yuan is not None and (metric == "net_profit_attributable_yuan" or value_yuan >= 10_000_000)
        rows.append({
            "company_code": record["company_code"],
            "company_name": record["company_name"],
            "report_year": int(record["report_year"]),
            "metric": metric,
            "value_yuan": value_yuan,
            "source_page": best_page,
            "source_excerpt": excerpt,
            "page_metric_coverage": score,
            "extraction_method": "table_cell" if metric in table_values else "text_fallback",
            "unit_multiplier": multiplier,
            "verification_status": "machine_extracted_source_table_requires_manual_audit" if is_plausible and score == 3 else "extraction_requires_manual_review",
        })
    return rows


def main() -> int:
    manifest = pd.read_csv(repo_path("data", "reference", "annual_report_manifest.csv"), dtype=str, keep_default_na=False)
    manifest = manifest[manifest["mapping_status"] == "mapped"].copy()
    rows = [row for _, record in manifest.iterrows() for row in extract_company_metrics(record)]
    output = repo_path("outputs", "tables", "financial_metrics_2025_extracted.csv")
    ensure_directories([output.parent])
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False, encoding="utf-8")
    complete = result.groupby("company_code")["value_yuan"].apply(lambda values: values.notna().all()).sum()
    print(f"Wrote {len(result)} metric rows for {len(manifest)} companies to {output}; complete company rows: {complete}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
