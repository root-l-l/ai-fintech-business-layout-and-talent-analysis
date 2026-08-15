"""Create descriptive tables and figures for Part 2 from verified pipeline outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import ensure_directories, repo_path


def main() -> int:
    directions = pd.read_csv(repo_path("outputs", "tables", "company_core_ai_directions.csv"), dtype=str, keep_default_na=False)
    scores = pd.read_csv(repo_path("outputs", "tables", "company_ai_direction_category_scores.csv"), dtype=str, keep_default_na=False)
    scores["mean_similarity"] = pd.to_numeric(scores["mean_similarity"], errors="coerce")
    summary = scores.groupby("category", as_index=False).agg(
        company_count=("company_code", "nunique"),
        mean_similarity=("mean_similarity", "mean"),
    ).sort_values("mean_similarity", ascending=False)
    output_tables = repo_path("outputs", "tables")
    output_figures = repo_path("outputs", "figures")
    ensure_directories([output_tables, output_figures])
    summary.to_csv(output_tables / "ai_direction_category_summary.csv", index=False, encoding="utf-8")

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.barh(summary["category"], summary["mean_similarity"], color="#167D9A")
    axis.invert_yaxis()
    axis.set_xlabel("Mean cosine similarity (pretrained embedding)")
    axis.set_title("AI fintech theme prominence across 15-company annual-report sections")
    for bar, count in zip(bars, summary["company_count"]):
        axis.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"n={count}", va="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_figures / "ai_direction_category_summary.png", dpi=220)
    plt.close(figure)
    print(f"Wrote category summary and figure for {len(directions)} companies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
