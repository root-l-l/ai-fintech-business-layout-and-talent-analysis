"""Validate manually or API-produced LLM extraction files before evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import ensure_directories, repo_path, write_json


REQUIRED = {"job_title", "salary_range", "work_location", "hard_skills", "soft_skills", "education", "experience", "ai_tech_stack"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-file", default="data/reference/llm_outputs.csv")
    parser.add_argument("--model-id", help="Optional model filter for a single auditable experiment")
    parser.add_argument("--run-ids", nargs="+", help="Optional run-id filter, for example FULL_R1 FULL_R2")
    parser.add_argument("--output", default="outputs/tables/llm_output_validation.json")
    args = parser.parse_args()
    runs_path = repo_path(*args.runs_file.split("/"))
    report_path = repo_path(*args.output.split("/"))
    ensure_directories([report_path.parent])
    if not runs_path.exists():
        write_json(report_path, {"status": "no_runs_file", "checked": 0, "errors": []})
        print(f"No run file found: {runs_path}")
        return 0
    runs = pd.read_csv(runs_path, dtype=str, keep_default_na=False)
    if args.model_id:
        runs = runs.loc[runs["model_id"].eq(args.model_id)].copy()
    if args.run_ids:
        runs = runs.loc[runs["run_id"].isin(args.run_ids)].copy()
    errors = []
    checked = 0
    for _, row in runs.iterrows():
        path = repo_path(*row["parsed_json_path"].split("/"))
        if not path.exists():
            errors.append({"job_id": row["job_id"], "prompt_version": row["prompt_version"], "error": "missing_parsed_json"})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append({"job_id": row["job_id"], "prompt_version": row["prompt_version"], "error": "invalid_json"})
            continue
        missing = sorted(REQUIRED - set(payload))
        if missing:
            errors.append({"job_id": row["job_id"], "prompt_version": row["prompt_version"], "error": f"missing_fields:{','.join(missing)}"})
        for field in ["hard_skills", "soft_skills", "ai_tech_stack"]:
            if field in payload and not isinstance(payload[field], list):
                errors.append({"job_id": row["job_id"], "prompt_version": row["prompt_version"], "error": f"{field}_not_list"})
        checked += 1
    report = {
        "status": "passed" if not errors else "failed",
        "checked": checked,
        "error_count": len(errors),
        "model_id": args.model_id,
        "run_ids": args.run_ids,
        "errors": errors,
    }
    write_json(report_path, report)
    print(f"Validation {report['status']}: checked {checked} run(s), {len(errors)} issue(s).")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
