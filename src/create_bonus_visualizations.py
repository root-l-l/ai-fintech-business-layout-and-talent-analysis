"""Create presentation-ready figures from existing, reproducible project tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common import ensure_directories, repo_path


PALETTE = {"teal": "#277A76", "coral": "#C84C3A", "gold": "#D49A28", "ink": "#202A36", "mist": "#E9F1F0"}
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "soft_teal",
    ["#FFFDF3", "#EEF7F5", "#D9EEEC", "#BCDCE0", "#91C5CF", "#62AAB7"],
)


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
    image = axis.imshow(
        pivot.to_numpy(),
        cmap=HEATMAP_CMAP,
        aspect="auto",
        vmin=0,
        vmax=float(pivot.max().max()),
    )
    axis.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns, rotation=20, ha="right")
    axis.set_yticks(np.arange(pivot.shape[0]), labels=pivot.index)
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            axis.text(column, row, f"{pivot.iat[row, column]:.2f}", ha="center", va="center", fontsize=8, color="#243746")
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


def _wrap_label(value: str, width: int) -> str:
    """Wrap Chinese-heavy labels deterministically for a dense network view."""

    return "\n".join(value[index : index + width] for index in range(0, len(value), width))


def knowledge_graph_overview(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    category_scores: pd.DataFrame,
    output: Path,
) -> None:
    """Draw an evidence-flow projection without encoding verification status by line style."""

    del category_scores  # The right-hand lane uses product-level technical categories.
    labels = nodes.drop_duplicates("node_id").set_index("node_id")["label"].to_dict()
    offers = edges[edges["relation"].isin([
        "offers_stage1_product_line_claim", "offers_page_verified_ai_product_entity"
    ])].copy()
    category_edges = edges[edges["relation"].eq("maps_to_technical_category")].copy()
    product_ids = offers["target"].drop_duplicates().tolist()
    company_ids = offers["source"].drop_duplicates().tolist()
    category_by_product = category_edges.drop_duplicates("source").set_index("source")["target"].to_dict()

    # Stack product entities by company.  This makes the source -> entity ->
    # direction evidence path readable even after screenshot-level expansion.
    products_by_company = {
        company_id: sorted(
            offers.loc[offers["source"].eq(company_id), "target"].drop_duplicates(),
            key=lambda product_id: str(labels[product_id]),
        )
        for company_id in company_ids
    }
    product_y: dict[str, float] = {}
    cursor = float(len(product_ids))
    for company_id in company_ids:
        for product_id in products_by_company[company_id]:
            product_y[product_id] = cursor
            cursor -= 1.0
    company_y = {
        company_id: float(np.mean([product_y[product_id] for product_id in products]))
        for company_id, products in products_by_company.items()
    }
    category_ids = sorted(set(category_by_product.values()), key=lambda category_id: str(labels[category_id]))
    category_y = {
        category_id: float(np.mean([
            product_y[product_id] for product_id, mapped_id in category_by_product.items() if mapped_id == category_id
        ]))
        for category_id in category_ids
    }

    height = max(10.5, len(product_ids) * 0.37 + 2.4)
    figure, axis = plt.subplots(figsize=(16.2, height), constrained_layout=True)
    for company_id, products in products_by_company.items():
        for product_id in products:
            axis.plot([0.10, 0.48], [company_y[company_id], product_y[product_id]], color="#BBC4C9", linewidth=0.75, alpha=0.72, zorder=1)
            category_id = category_by_product[product_id]
            axis.plot([0.52, 0.91], [product_y[product_id], category_y[category_id]], color="#BBC4C9", linewidth=0.75, alpha=0.72, zorder=1)

    for company_id, y_value in company_y.items():
        axis.scatter(0.08, y_value, s=205, color="#21869A", edgecolor="white", linewidth=0.8, zorder=3)
        axis.text(0.045, y_value, labels[company_id], ha="right", va="center", fontsize=8.0, fontweight="bold", color="#1D2933")
    for product_id, y_value in product_y.items():
        axis.scatter(0.50, y_value, s=38, color="#D49A28", edgecolor="white", linewidth=0.55, zorder=3)
        axis.text(0.516, y_value, _wrap_label(str(labels[product_id]), 13), ha="left", va="center", fontsize=6.1, color="#263238")
    for category_id, y_value in category_y.items():
        axis.scatter(0.94, y_value, s=238, color="#D95A5A", edgecolor="white", linewidth=0.8, zorder=3)
        axis.text(0.965, y_value, _wrap_label(str(labels[category_id]), 7), ha="left", va="center", fontsize=7.2, fontweight="bold", color="#1D2933")

    top_y = len(product_ids) + 1.2
    axis.text(0.08, top_y, "样本公司", ha="center", va="bottom", fontsize=10, fontweight="bold", color=PALETTE["ink"])
    axis.text(0.50, top_y, "产品线 / 产品实体", ha="center", va="bottom", fontsize=10, fontweight="bold", color=PALETTE["ink"])
    axis.text(0.94, top_y, "技术方向", ha="center", va="bottom", fontsize=10, fontweight="bold", color=PALETTE["ink"])
    legend_items = [
        plt.Line2D([0], [0], marker="o", color="w", label=f"公司（{len(company_ids)}）", markerfacecolor="#21869A", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label=f"产品线 / 实体（{len(product_ids)}）", markerfacecolor="#D49A28", markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", label=f"技术方向（{len(category_ids)}）", markerfacecolor="#D95A5A", markersize=8),
    ]
    axis.legend(handles=legend_items, loc="lower left", frameon=True, framealpha=0.94, fontsize=8)
    axis.set_title("AI 金融科技上市公司—产品线—技术方向知识图谱（证据流图）", fontsize=14, fontweight="bold", pad=18)
    axis.text(
        0.50,
        -0.025,
        "产品层包含 15 条第一阶段公司级主张与 22 个官网截图逐页确认的金融 AI 实体；37 张截图的逐页结论见 screenshot_page_mapping.csv。连线仅表示材料中的归属与证据关系，不表示商业合作、技术依赖或因果关系。",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#52616B",
    )
    axis.set(xlim=(-0.17, 1.18), ylim=(0.0, top_y + 0.75))
    axis.axis("off")
    figure.savefig(output, dpi=260, bbox_inches="tight", facecolor="white")
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
    knowledge_graph_overview(
        pd.read_csv(repo_path("outputs", "knowledge_graph", "nodes.csv")),
        pd.read_csv(repo_path("outputs", "knowledge_graph", "edges.csv")),
        pd.read_csv(tables / "company_ai_direction_category_scores.csv"),
        args.output_dir / "product_line_knowledge_graph.png",
    )
    history_path = tables / "domain_encoder_training_history.csv"
    if history_path.exists():
        encoder_learning_curve(pd.read_csv(history_path), args.output_dir / "domain_encoder_learning_curve.png")
    print(f"Wrote presentation-ready figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
