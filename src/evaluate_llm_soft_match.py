"""Add transparent lexical soft-match metrics for structured skill extraction.

Strict set equality remains the primary reproducible accuracy metric.  This
script reports Jaccard and token F1 for the three list fields so that partial
lexical overlap is not discarded when gold annotations use long composite
phrases and model outputs use atomic skills.  It is not a semantic-equivalence
test and does not replace manual review.
"""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from common import ensure_directories, repo_path


LIST_FIELDS = ["hard_skills", "soft_skills", "ai_tech_stack"]


def tokenize(value: object) -> set[str]:
    """Use technical Latin tokens and overlapping Han-character bigrams."""
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    text = str(value or "").lower()
    english = re.findall(r"[a-z][a-z0-9+.#/_-]*", text)
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", text)
    chinese_bigrams = [chunk[index : index + 2] for chunk in chinese_chunks for index in range(max(0, len(chunk) - 1))]
    chinese_singletons = [chunk for chunk in chinese_chunks if len(chunk) == 1]
    return set(english + chinese_bigrams + chinese_singletons)


def load_predictions(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, run in runs.iterrows():
        path = repo_path(*run["parsed_json_path"].split("/"))
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append({"job_id": run["job_id"], "prompt_version": run["prompt_version"], **{field: payload.get(field, []) for field in LIST_FIELDS}})
    return pd.DataFrame(rows)


def scores(prediction: object, gold: object) -> tuple[float, float]:
    predicted_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not predicted_tokens and not gold_tokens:
        return 1.0, 1.0
    intersection = len(predicted_tokens & gold_tokens)
    union = len(predicted_tokens | gold_tokens)
    jaccard = intersection / union if union else 0.0
    f1 = 2 * intersection / (len(predicted_tokens) + len(gold_tokens)) if predicted_tokens or gold_tokens else 1.0
    return jaccard, f1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-id", default="FULL_R1")
    args = parser.parse_args()
    gold_path = repo_path("data", "reference", "gold_job_annotations.csv")
    runs_path = repo_path("data", "reference", "llm_outputs.csv")
    output_dir = repo_path("outputs", "tables")
    ensure_directories([output_dir])
    if not gold_path.exists() or not runs_path.exists():
        raise FileNotFoundError("Gold annotations and LLM output metadata are required.")
    gold = pd.read_csv(gold_path, dtype=str, keep_default_na=False).groupby("job_id", as_index=False)[LIST_FIELDS].first()
    runs = pd.read_csv(runs_path, dtype=str, keep_default_na=False)
    runs = runs.loc[runs["model_id"].eq(args.model_id) & runs["run_id"].eq(args.run_id)].copy()
    predictions = load_predictions(runs)
    merged = predictions.merge(gold, on="job_id", suffixes=("_pred", "_gold"), how="inner")
    detail_rows = []
    for _, row in merged.iterrows():
        for field in LIST_FIELDS:
            jaccard, f1 = scores(row[f"{field}_pred"], row[f"{field}_gold"])
            detail_rows.append({
                "job_id": row["job_id"],
                "prompt_version": row["prompt_version"],
                "field": field,
                "jaccard": round(jaccard, 6),
                "token_f1": round(f1, 6),
            })
    detail = pd.DataFrame(detail_rows)
    summary = detail.groupby(["prompt_version", "field"], as_index=False).agg(
        n=("job_id", "count"), mean_jaccard=("jaccard", "mean"), mean_token_f1=("token_f1", "mean")
    )
    summary["mean_jaccard"] = summary["mean_jaccard"].round(4)
    summary["mean_token_f1"] = summary["mean_token_f1"].round(4)
    macro = summary.groupby("prompt_version", as_index=False).agg(
        fields=("field", "count"), macro_jaccard=("mean_jaccard", "mean"), macro_token_f1=("mean_token_f1", "mean")
    )
    macro["macro_jaccard"] = macro["macro_jaccard"].round(4)
    macro["macro_token_f1"] = macro["macro_token_f1"].round(4)
    detail.to_csv(output_dir / "prompt_skill_soft_match_detail.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / "prompt_skill_soft_match_summary.csv", index=False, encoding="utf-8")
    macro.to_csv(output_dir / "prompt_skill_soft_match_macro.csv", index=False, encoding="utf-8")
    print(f"Wrote {len(detail)} soft-match comparisons and {len(summary)} prompt-field summaries to {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
