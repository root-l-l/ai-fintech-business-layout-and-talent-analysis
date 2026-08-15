"""Create a provenance table for first-stage AI product-line claims.

Website snapshots establish that a company has supplied product-page material,
but a company-level snapshot is not automatically proof of every named product.
This script makes that distinction explicit.
"""

from __future__ import annotations

import re

import pandas as pd

from common import ensure_directories, repo_path


def normalized(value: str) -> str:
    return re.sub(r"[\s\-_/＋+]", "", value).lower()


def main() -> int:
    product_lines = pd.read_csv(repo_path("data", "reference", "ai_product_lines.csv"), dtype=str, keep_default_na=False)
    text_snapshots = pd.read_csv(repo_path("data", "processed", "product_page_snapshots.csv"), dtype=str, keep_default_na=False)
    screenshots = pd.read_csv(repo_path("data", "processed", "product_page_screenshots.csv"), dtype=str, keep_default_na=False)
    verification_path = repo_path("outputs", "tables", "ai_product_line_verification.csv")
    verification = pd.read_csv(verification_path, dtype=str, keep_default_na=False).set_index("company_code") if verification_path.exists() else pd.DataFrame()
    text_by_company = text_snapshots.set_index("company_code") if not text_snapshots.empty else pd.DataFrame()
    screenshot_counts = screenshots.groupby("company_code").agg(
        screenshot_count=("source_filename", "count"), screenshot_paths=("local_path", lambda values: "；".join(values))
    )
    rows: list[dict[str, object]] = []
    for _, line in product_lines.iterrows():
        code = line["company_code"]
        has_text = code in text_by_company.index
        text_match = False
        text_source = ""
        if has_text:
            snapshot = text_by_company.loc[code]
            text_source = str(snapshot["local_path"])
            text_match = normalized(line["product_line"]) in normalized(str(snapshot["snapshot_text"]))
        has_screenshots = code in screenshot_counts.index
        if text_match:
            status = "exact_product_name_found_in_text_snapshot"
        elif has_text:
            status = "website_text_snapshot_available_product_name_not_exactly_matched"
        elif has_screenshots:
            status = "website_screenshots_available_manual_product_name_review_required"
        else:
            status = "stage1_claim_only_requires_source_review"
        verification_row = verification.loc[code] if not verification.empty and code in verification.index else None
        rows.append({
            **line.to_dict(),
            "text_snapshot_path": text_source,
            "exact_product_name_text_match": text_match,
            "screenshot_count": int(screenshot_counts.loc[code, "screenshot_count"]) if has_screenshots else 0,
            "screenshot_paths": str(screenshot_counts.loc[code, "screenshot_paths"]) if has_screenshots else "",
            "evidence_status": status,
            "entity_verification_status": verification_row["verification_status"] if verification_row is not None else "verification_not_run",
            "verified_anchors": verification_row["verified_anchors"] if verification_row is not None else "",
            "annual_report_match_locations": verification_row["annual_report_match_locations"] if verification_row is not None else "",
            "entity_evidence_excerpt": verification_row["evidence_excerpt"] if verification_row is not None else "",
            "evidence_note": "官网快照覆盖只能支持来源追溯；产品名称、功能和场景需与具体页码或段落一一对应后才可标为已核验。",
        })
    output = repo_path("outputs", "tables", "ai_product_line_evidence.csv")
    ensure_directories([output.parent])
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote provenance rows for {len(rows)} AI product-line claims to {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
