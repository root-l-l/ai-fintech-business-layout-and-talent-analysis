"""Run a category-scale sensitivity check on saved embedding top-k matches.

The primary graph retains the observed top-k keyword associations.  Because the
full embedding matrix may be unavailable when a pretrained model is not cached,
this script standardizes mean similarity within each category using only those
saved associations.  It is a sensitivity analysis, not a replacement for a
full-matrix category calibration.
"""

from __future__ import annotations

import pandas as pd

from common import ensure_directories, repo_path


def main() -> int:
    scores = pd.read_csv(repo_path("outputs", "tables", "semantic_keyword_scores.csv"), dtype=str, keep_default_na=False)
    pool = pd.read_csv(repo_path("data", "reference", "final_sample_pool.csv"), dtype=str, keep_default_na=False)
    scores["similarity"] = pd.to_numeric(scores["similarity"], errors="coerce")
    categories = sorted(scores["category"].dropna().unique())
    base = pool[["company_code"]].assign(_key=1).merge(pd.DataFrame({"category": categories, "_key": 1}), on="_key").drop(columns="_key")
    observed = scores.groupby(["company_code", "category"], as_index=False).agg(
        retained_match_count=("keyword", "count"),
        mean_retained_similarity=("similarity", "mean"),
    )
    frame = base.merge(observed, on=["company_code", "category"], how="left")
    frame["retained_match_count"] = frame["retained_match_count"].fillna(0).astype(int)
    frame["mean_retained_similarity"] = frame["mean_retained_similarity"].fillna(0.0)
    baseline_mean = frame.groupby("category")["mean_retained_similarity"].transform("mean")
    baseline_std = frame.groupby("category")["mean_retained_similarity"].transform("std").replace(0, 1.0)
    frame["retained_category_zscore"] = ((frame["mean_retained_similarity"] - baseline_mean) / baseline_std).round(6)
    frame["sensitivity_rank"] = frame.groupby("company_code")["retained_category_zscore"].rank(method="first", ascending=False).astype(int)
    output = repo_path("outputs", "tables", "semantic_category_standardized_scores.csv")
    ensure_directories([output.parent])
    frame.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(frame)} retained-top-k category standardization rows to {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
