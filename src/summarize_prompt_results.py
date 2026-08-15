"""Create a report-ready summary from field-level quality and stability outputs.

The summary uses the pre-specified strict field comparison in evaluate_llm.py:
after normalization, single-value fields must match exactly and list fields are
compared as unordered exact sets. It does not silently substitute a more lenient
metric after results are observed.
"""

from __future__ import annotations

import argparse

import pandas as pd

from common import ensure_directories, repo_path


ADVICE = {
    "P1": "作为 zero-shot 基线保留；增加字段定义与证据约束以降低软技能误填。",
    "P2": "跨轮一致性最高；保留 Schema 与证据约束，并优先优化 AI 技术栈的术语归一化。",
    "P3": "严格字段准确率最高；保留 one-shot 与自检，但针对 AI 技术栈增加受控术语表或逐项证据校验。",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quality-run-id", default="FULL_R1")
    parser.add_argument("--stability-run-ids", nargs=2, default=["FULL_R1", "FULL_R2"], metavar=("RUN_A", "RUN_B"))
    args = parser.parse_args()

    quality = pd.read_csv(repo_path("outputs", "tables", "prompt_evaluation.csv"))
    stability = pd.read_csv(repo_path("outputs", "tables", "prompt_stability.csv"))
    if quality.empty or stability.empty:
        raise SystemExit("Quality or stability output is empty; run the corresponding evaluators first.")

    quality_summary = quality.groupby("prompt_version", as_index=False).agg(
        strict_macro_field_accuracy=("accuracy", "mean"),
        macro_hallucination_rate=("hallucination_rate", "mean"),
        mean_runtime_seconds=("mean_runtime_seconds", "mean"),
        quality_sample_n=("n", "max"),
    )
    stability_summary = stability.groupby("prompt_version", as_index=False).agg(
        macro_pairwise_agreement=("pairwise_agreement", "mean"),
        stability_sample_n=("jobs_with_repeats", "max"),
        minimum_field_agreement=("pairwise_agreement", "min"),
    )
    summary = quality_summary.merge(stability_summary, on="prompt_version", how="left")
    summary.insert(1, "model_id", args.model_id)
    summary.insert(2, "quality_run_id", args.quality_run_id)
    summary.insert(3, "stability_run_ids", " / ".join(args.stability_run_ids))
    summary["strict_macro_field_accuracy"] = summary["strict_macro_field_accuracy"].round(4)
    summary["macro_hallucination_rate"] = summary["macro_hallucination_rate"].round(4)
    summary["mean_runtime_seconds"] = summary["mean_runtime_seconds"].round(3)
    summary["macro_pairwise_agreement"] = summary["macro_pairwise_agreement"].round(4)
    summary["minimum_field_agreement"] = summary["minimum_field_agreement"].round(4)
    summary["optimization_advice"] = summary["prompt_version"].map(ADVICE)
    summary["metric_note"] = (
        "严格宏平均：8字段归一化后完全一致率；技能列表按无序集合完全一致比较。"
        "运行时间为本次并发调用的单请求端到端耗时均值。"
    )
    output = repo_path("outputs", "tables", "prompt_version_summary.csv")
    ensure_directories([output.parent])
    summary.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(summary)} report-ready prompt-version summaries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
