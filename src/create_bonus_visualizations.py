"""Create presentation-ready figures from existing, reproducible project tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_directories, repo_path


PALETTE = {"teal": "#277A76", "coral": "#C84C3A", "gold": "#D49A28", "ink": "#202A36", "mist": "#E9F1F0"}


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.titleweight"] = "bold"


def heatmap(company_scores: pd.DataFrame, pool: pd.DataFrame, output: Path) -> None:
    names = pool.set_index("company_code")["company_name"]
    pivot = company_scores.pivot(index="company_code", columns="category", values="mean_similarity").fillna(0)
    pivot.index = pivot.index.map(names)
    pivot = pivot.sort_index()
    figure, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    image = axis.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=float(pivot.min().min()), vmax=float(pivot.max().max()))
    axis.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns, rotation=20, ha="right")
    axis.set_yticks(np.arange(pivot.shape[0]), labels=pivot.index)
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            axis.text(column, row, f"{pivot.iat[row, column]:.2f}", ha="center", va="center", fontsize=8, color=PALETTE["ink"])
    figure.colorbar(image, ax=axis, label="年报章节与技术方向的平均语义相似度")
    axis.set(title="图B-1 15家样本公司的 AI 技术方向语义布局", xlabel="AI 技术方向", ylabel="公司")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def prompt_comparison(summary: pd.DataFrame, output: Path) -> None:
    versions = summary["prompt_version"].tolist()
    x = np.arange(len(versions))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]})
    quality = [
        ("strict_macro_field_accuracy", "严格宏平均准确率", PALETTE["teal"]),
        ("macro_pairwise_agreement", "跨轮一致性", PALETTE["gold"]),
        ("macro_hallucination_rate", "操作性幻觉率", PALETTE["coral"]),
    ]
    width = 0.24
    for index, (column, label, color) in enumerate(quality):
        bars = axes[0].bar(x + (index - 1) * width, summary[column], width=width, label=label, color=color)
        for bar in bars:
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set(xticks=x, xticklabels=versions, ylim=(0, 1.1), ylabel="比例", title="Prompt 质量与稳定性")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    runtime = axes[1].bar(versions, summary["mean_runtime_seconds"], color=[PALETTE["teal"], PALETTE["gold"], PALETTE["coral"]])
    for bar in runtime:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{bar.get_height():.2f}s", ha="center", fontsize=9)
    axes[1].set(ylim=(0, float(summary["mean_runtime_seconds"].max()) + 1), ylabel="每请求平均秒数", title="运行时间")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    figure.suptitle("图B-2 三套 Prompt 的真实模型实验结果", fontsize=14, fontweight="bold")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def recruitment_skills(skills: pd.DataFrame, output: Path) -> None:
    top = skills.nlargest(12, "job_records").sort_values("job_records")
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = axis.barh(top["keyword"], top["job_records"], color=PALETTE["teal"])
    for bar, share in zip(bars, top["share_of_cleaned_records"]):
        axis.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width())} ({share:.1%})", va="center", fontsize=9)
    axis.set(xlim=(0, float(top["job_records"].max()) + 18), xlabel="包含该关键词的招聘记录数", title="图B-3 招聘文本中出现频率最高的技能与领域词")
    axis.grid(axis="x", linestyle=":", alpha=0.5)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def encoder_learning_curve(history: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", linewidth=2, color=PALETTE["coral"])
    axes[0].set(xticks=history["epoch"], xlabel="训练轮次", ylabel="训练损失", title="训练损失")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    axes[1].plot(history["epoch"], history["accuracy"], marker="o", linewidth=2, color=PALETTE["teal"], label="准确率")
    axes[1].plot(history["epoch"], history["f1"], marker="o", linewidth=2, color=PALETTE["gold"], label="F1")
    axes[1].set(xticks=history["epoch"], ylim=(0, 1), xlabel="训练轮次", ylabel="留出集指标", title="留出集表现")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    axes[1].legend(frameon=False)
    figure.suptitle("图B-4 小规模金融招聘文本编码器的实际微调过程", fontsize=14, fontweight="bold")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def knowledge_graph_overview(nodes: pd.DataFrame, edges: pd.DataFrame, output: Path) -> None:
    """Draw the observed company-product-line-direction subgraph in three layers."""

    labels = nodes.drop_duplicates("node_id").set_index("node_id")["label"].to_dict()
    company_to_line = edges[edges["relation"].eq("offers_ai_product_line")].copy()
    line_to_direction = edges[edges["relation"].eq("implements_direction")].copy()
    company_ids = sorted(company_to_line["source"].unique(), key=lambda node: str(labels[node]))
    line_ids = company_to_line["target"].tolist()
    direction_ids = sorted(line_to_direction["target"].unique(), key=lambda node: str(labels[node]))
    company_positions = {node: (0.0, len(company_ids) - index - 1) for index, node in enumerate(company_ids)}
    line_positions = {node: (1.0, company_positions[company_to_line.loc[company_to_line["target"].eq(node), "source"].iloc[0]][1]) for node in line_ids}
    direction_y = np.linspace(len(company_ids) - 1, 0, num=len(direction_ids))
    direction_positions = {node: (2.0, float(y)) for node, y in zip(direction_ids, direction_y)}
    positions = {**company_positions, **line_positions, **direction_positions}
    figure, axis = plt.subplots(figsize=(18, 10), constrained_layout=True)
    for _, edge in company_to_line.iterrows():
        start, end = positions[edge["source"]], positions[edge["target"]]
        axis.plot([start[0], end[0]], [start[1], end[1]], color="#8FA9A7", linewidth=1.2, alpha=0.85, zorder=1)
    for _, edge in line_to_direction.iterrows():
        start, end = positions[edge["source"]], positions[edge["target"]]
        axis.plot([start[0], end[0]], [start[1], end[1]], color="#D4A64A", linewidth=1.1, alpha=0.65, zorder=1)
    for node, (x, y) in company_positions.items():
        axis.scatter(x, y, s=260, color=PALETTE["teal"], zorder=2)
        axis.text(x - 0.05, y, labels[node], ha="right", va="center", fontsize=9)
    for node, (x, y) in line_positions.items():
        axis.scatter(x, y, s=130, color=PALETTE["gold"], marker="s", zorder=2)
        axis.text(x, y, labels[node], ha="center", va="bottom", fontsize=7, wrap=True)
    for node, (x, y) in direction_positions.items():
        axis.scatter(x, y, s=260, color=PALETTE["coral"], marker="D", zorder=2)
        axis.text(x + 0.05, y, labels[node], ha="left", va="center", fontsize=9)
    for x, title in ((0.0, "公司"), (1.0, "AI 产品线"), (2.0, "AI 技术方向")):
        axis.text(x, len(company_ids) + 0.15, title, ha="center", va="bottom", fontweight="bold", fontsize=12, color=PALETTE["ink"])
    axis.set(xlim=(-0.55, 2.75), ylim=(-0.8, len(company_ids) + 0.7), title="产品线知识图谱总览：公司--产品线--AI 技术方向")
    axis.axis("off")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reusable visualizations from project output tables.")
    parser.add_argument("--output-dir", type=Path, default=repo_path("outputs", "figures"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    ensure_directories([args.output_dir])
    tables = repo_path("outputs", "tables")
    heatmap(pd.read_csv(tables / "company_ai_direction_category_scores.csv"), pd.read_csv(repo_path("data", "reference", "final_sample_pool.csv")), args.output_dir / "company_ai_direction_heatmap.png")
    prompt_comparison(pd.read_csv(tables / "prompt_version_summary.csv"), args.output_dir / "prompt_quality_comparison.png")
    recruitment_skills(pd.read_csv(tables / "recruitment_skill_keyword_counts.csv"), args.output_dir / "recruitment_skill_demand.png")
    knowledge_graph_overview(pd.read_csv(repo_path("outputs", "knowledge_graph", "nodes.csv")), pd.read_csv(repo_path("outputs", "knowledge_graph", "edges.csv")), args.output_dir / "product_line_knowledge_graph.png")
    history_path = tables / "domain_encoder_training_history.csv"
    if history_path.exists():
        encoder_learning_curve(pd.read_csv(history_path), args.output_dir / "domain_encoder_learning_curve.png")
    print(f"Wrote presentation-ready figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
