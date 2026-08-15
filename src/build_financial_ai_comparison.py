"""Join audited annual-report financial metrics to AI-direction evidence.

Financial scale is used for descriptive grouping only.  The script does not
estimate an effect of assets or profitability on AI capability.
"""

from __future__ import annotations

import pandas as pd

from common import ensure_directories, repo_path


METRIC_COLUMNS = {
    "revenue_yuan": "revenue_yuan",
    "net_profit_attributable_yuan": "net_profit_attributable_yuan",
    "total_assets_yuan": "total_assets_yuan",
}


def main() -> int:
    extracted = pd.read_csv(repo_path("outputs", "tables", "financial_metrics_2025_extracted.csv"), dtype={"company_code": str}, keep_default_na=False)
    corrections = pd.read_csv(repo_path("data", "reference", "financial_metrics_2025_manual_corrections.csv"), dtype={"company_code": str}, keep_default_na=False)
    for _, correction in corrections.iterrows():
        mask = (extracted["company_code"] == correction["company_code"]) & (extracted["metric"] == correction["metric"])
        extracted.loc[mask, "value_yuan"] = float(correction["value_yuan"])
        extracted.loc[mask, "source_page"] = int(correction["source_page"])
        extracted.loc[mask, "source_excerpt"] = correction["source_excerpt"]
        extracted.loc[mask, "extraction_method"] = "manual_source_line_correction"
        extracted.loc[mask, "verification_status"] = correction["review_status"]
    duplicate_keys = extracted.duplicated(["company_code", "metric"])
    if duplicate_keys.any():
        raise ValueError("Financial metric table has duplicate company-metric rows.")
    wide = extracted.pivot(index=["company_code", "company_name"], columns="metric", values="value_yuan").reset_index()
    missing = [column for column in METRIC_COLUMNS.values() if column not in wide.columns or wide[column].isna().any()]
    if missing:
        raise ValueError(f"Missing financial values after corrections: {missing}")
    pool = pd.read_csv(repo_path("data", "reference", "final_sample_pool.csv"), dtype=str, keep_default_na=False)
    directions = pd.read_csv(repo_path("outputs", "tables", "company_core_ai_directions.csv"), dtype=str, keep_default_na=False)
    comparison = pool[["company_code", "company_name", "segment"]].merge(wide, on=["company_code", "company_name"], validate="one_to_one")
    comparison = comparison.merge(directions[["company_code", "core_ai_directions", "direction_keywords"]], on="company_code", validate="one_to_one")
    comparison["total_assets_billion_yuan"] = comparison["total_assets_yuan"] / 1_000_000_000
    comparison["revenue_billion_yuan"] = comparison["revenue_yuan"] / 1_000_000_000
    comparison["net_profit_billion_yuan"] = comparison["net_profit_attributable_yuan"] / 1_000_000_000
    comparison["net_profit_margin"] = comparison["net_profit_attributable_yuan"] / comparison["revenue_yuan"]
    comparison["asset_size_group"] = pd.qcut(
        comparison["total_assets_yuan"], q=3, labels=["低资产组", "中资产组", "高资产组"], duplicates="drop"
    ).astype(str)
    comparison = comparison.sort_values("total_assets_yuan", ascending=False)
    output = repo_path("outputs", "tables", "company_ai_financial_comparison.csv")
    ensure_directories([output.parent])
    comparison.to_csv(output, index=False, encoding="utf-8")
    group_summary = comparison.groupby("asset_size_group", observed=True).agg(
        company_count=("company_code", "count"),
        median_assets_billion_yuan=("total_assets_billion_yuan", "median"),
        median_revenue_billion_yuan=("revenue_billion_yuan", "median"),
        median_net_profit_billion_yuan=("net_profit_billion_yuan", "median"),
        profitable_company_count=("net_profit_attributable_yuan", lambda values: int((values > 0).sum())),
    ).reset_index()
    group_summary.to_csv(repo_path("outputs", "tables", "financial_ai_group_summary.csv"), index=False, encoding="utf-8")
    direction_summary = comparison.assign(ai_direction=comparison["core_ai_directions"].str.split("；")).explode("ai_direction")
    direction_summary = direction_summary.groupby(["asset_size_group", "ai_direction"], as_index=False).agg(
        company_count=("company_code", "nunique")
    )
    direction_summary.to_csv(repo_path("outputs", "tables", "financial_ai_direction_group_summary.csv"), index=False, encoding="utf-8")
    provenance = extracted[["company_code", "metric", "source_page", "source_excerpt", "extraction_method", "verification_status"]].copy()
    provenance.to_csv(repo_path("outputs", "tables", "financial_metrics_2025_provenance.csv"), index=False, encoding="utf-8")
    print(f"Wrote comparison for {len(comparison)} companies to {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
