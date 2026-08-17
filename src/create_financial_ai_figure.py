"""Visualize financial scale and profitability alongside AI-layout categories."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import ensure_directories, repo_path


COLORS = {
    "证券与资管科技": "#197C9A",
    "银行科技": "#D36C54",
    "保险科技": "#5B8C5A",
    "泛金融AI底座": "#6D6AAE",
}


def main() -> int:
    comparison = pd.read_csv(repo_path("outputs", "tables", "company_ai_financial_comparison.csv"))
    output = repo_path("outputs", "figures", "financial_scale_profitability_ai_layout.png")
    ensure_directories([output.parent])
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)
    for segment, subset in comparison.groupby("segment"):
        axis.scatter(
            subset["total_assets_billion_yuan"],
            subset["net_profit_margin"] * 100,
            s=(subset["revenue_billion_yuan"].clip(lower=0.3) ** 0.5) * 90,
            color=COLORS.get(segment, "#666666"),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
            label=segment,
        )
    for _, row in comparison.iterrows():
        axis.annotate(row["company_name"], (row["total_assets_billion_yuan"], row["net_profit_margin"] * 100), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xscale("log")
    axis.set_xlabel("2025 年总资产（十亿元，对数刻度）")
    axis.set_ylabel("归母净利率（%）")
    axis.set_title("图 3：财务规模、盈利表现与 AI 布局的描述性比较")
    axis.grid(linestyle=":", alpha=0.4)
    axis.legend(title="样本赛道", frameon=False, fontsize=8, title_fontsize=8, loc="lower right")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote financial comparison figure to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
