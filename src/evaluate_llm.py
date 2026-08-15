"""Evaluate LLM field extraction using human gold annotations, never invented labels."""

from __future__ import annotations

import json
import re
import argparse
from pathlib import Path

import pandas as pd

from common import ensure_directories, repo_path


FIELDS = ["job_title", "salary_range", "work_location", "hard_skills", "soft_skills", "education", "experience", "ai_tech_stack"]
LIST_FIELDS = {"hard_skills", "soft_skills", "ai_tech_stack"}
EMPTY_MARKERS = {"", "无", "无明确要求", "未提及", "未公示", "nan", "none", "null", "[]"}


def norm(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        values = [norm(item) for item in value]
        return "|".join(sorted(set(item for item in values if item)))
    text = " ".join(str(value).strip().lower().split())
    return "" if text in EMPTY_MARKERS else text


def norm_field(value: object, field: str) -> str:
    """Normalize lists as unordered sets; gold CSV uses `|` as its separator."""
    if field not in LIST_FIELDS:
        return norm(value)
    if isinstance(value, list):
        return norm(value)
    parts = re.split(r"[|]", str(value))
    return "|".join(sorted(set(item for item in (norm(part) for part in parts) if item)))


def load_predictions(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame(columns=["job_id", "prompt_version", *FIELDS])
    rows = []
    for _, row in runs.iterrows():
        json_path = repo_path(*row["parsed_json_path"].split("/"))
        if not json_path.exists():
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append({"job_id": row["job_id"], "prompt_version": row["prompt_version"], **{field: payload.get(field) for field in FIELDS}})
    return pd.DataFrame(rows, columns=["job_id", "prompt_version", *FIELDS])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="Evaluate only one actual model; never mix model outputs.")
    parser.add_argument("--run-id", default="R1", help="Run used for quality comparison, default: R1")
    args = parser.parse_args()
    gold_path = repo_path("data", "reference", "gold_job_annotations.csv")
    run_path = repo_path("data", "reference", "llm_outputs.csv")
    output = repo_path("outputs", "tables", "prompt_evaluation.csv")
    ensure_directories([output.parent])
    if not gold_path.exists() or not run_path.exists():
        pd.DataFrame(columns=["prompt_version", "field", "n", "accuracy", "hallucination_rate", "mean_runtime_seconds"]).to_csv(output, index=False, encoding="utf-8")
        print(f"Gold annotations or LLM runs not yet supplied; wrote empty schema: {output}")
        return 0
    gold = pd.read_csv(gold_path, dtype=str, keep_default_na=False)
    run_meta = pd.read_csv(run_path, dtype=str, keep_default_na=False)
    selected_runs = run_meta.loc[
        run_meta["model_id"].eq(args.model_id) & run_meta["run_id"].eq(args.run_id)
    ].copy()
    predictions = load_predictions(selected_runs)
    if gold.empty or predictions.empty:
        pd.DataFrame(columns=["prompt_version", "field", "n", "accuracy", "hallucination_rate", "mean_runtime_seconds"]).to_csv(output, index=False, encoding="utf-8")
        print(f"No comparable gold labels and predictions for model={args.model_id}, run={args.run_id}; wrote empty schema: {output}")
        return 0
    gold = gold.groupby("job_id", as_index=False)[FIELDS].agg(lambda s: s.iloc[0])
    merged = predictions.merge(gold, on="job_id", suffixes=("_pred", "_gold"), how="inner")
    selected_runs["runtime_seconds"] = (
        pd.to_datetime(selected_runs["finished_at"], errors="coerce")
        - pd.to_datetime(selected_runs["started_at"], errors="coerce")
    ).dt.total_seconds()
    merged = merged.merge(
        selected_runs[["job_id", "prompt_version", "runtime_seconds"]],
        on=["job_id", "prompt_version"],
        how="left",
    )
    result = []
    for prompt, group in merged.groupby("prompt_version"):
        for field in FIELDS:
            pred = group[f"{field}_pred"].map(lambda value: norm_field(value, field))
            gold_value = group[f"{field}_gold"].map(lambda value: norm_field(value, field))
            result.append({"prompt_version": prompt, "field": field, "n": len(group), "accuracy": round(float((pred == gold_value).mean()), 4), "hallucination_rate": round(float(((pred != "") & (gold_value == "")).mean()), 4), "mean_runtime_seconds": round(float(group["runtime_seconds"].mean()), 3) if group["runtime_seconds"].notna().any() else None})
    pd.DataFrame(result).to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(result)} prompt-evaluation rows for model={args.model_id}, run={args.run_id} to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
