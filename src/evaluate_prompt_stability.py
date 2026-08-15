"""Measure repeated-run consistency for each prompt and extraction field."""

from __future__ import annotations

import json
import argparse
from itertools import combinations

import pandas as pd

from common import ensure_directories, repo_path
from evaluate_llm import FIELDS, norm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="Compare repeated calls from one actual model only.")
    parser.add_argument("--run-ids", nargs=2, default=["R1", "R2"], metavar=("RUN_A", "RUN_B"))
    parser.add_argument("--sample-file", default="data/interim/prompt_stability_sample.csv")
    args = parser.parse_args()
    runs_path = repo_path("data", "reference", "llm_outputs.csv")
    output = repo_path("outputs", "tables", "prompt_stability.csv")
    ensure_directories([output.parent])
    if not runs_path.exists():
        pd.DataFrame(columns=["prompt_version", "field", "jobs_with_repeats", "pairwise_agreement"]).to_csv(output, index=False, encoding="utf-8")
        print("No LLM run metadata available; wrote empty stability schema.")
        return 0
    sample = pd.read_csv(repo_path(*args.sample_file.split("/")), dtype=str, keep_default_na=False)
    selected_job_ids = set(sample["job_id"])
    runs = pd.read_csv(runs_path, dtype=str, keep_default_na=False)
    runs = runs.loc[
        runs["model_id"].eq(args.model_id)
        & runs["run_id"].isin(args.run_ids)
        & runs["job_id"].isin(selected_job_ids)
    ]
    rows = []
    for _, run in runs.iterrows():
        path = repo_path(*run["parsed_json_path"].split("/"))
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append({"job_id": run["job_id"], "prompt_version": run["prompt_version"], "run_id": run["run_id"], **{field: norm(payload.get(field)) for field in FIELDS}})
    predictions = pd.DataFrame(rows)
    result = []
    if not predictions.empty:
        for (prompt, job_id), group in predictions.groupby(["prompt_version", "job_id"]):
            if set(group["run_id"]) != set(args.run_ids):
                continue
            for field in FIELDS:
                values = group[field].tolist()
                for left, right in combinations(values, 2):
                    result.append({"prompt_version": prompt, "job_id": job_id, "field": field, "agreement": int(left == right)})
    frame = pd.DataFrame(result)
    if frame.empty:
        summary = pd.DataFrame(columns=["prompt_version", "field", "jobs_with_repeats", "pairwise_agreement"])
    else:
        summary = frame.groupby(["prompt_version", "field"], as_index=False).agg(jobs_with_repeats=("job_id", "nunique"), pairwise_agreement=("agreement", "mean"))
        summary["pairwise_agreement"] = summary["pairwise_agreement"].round(4)
    summary.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(summary)} stability rows for model={args.model_id}, runs={args.run_ids} to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
