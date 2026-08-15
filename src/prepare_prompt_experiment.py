"""Create a stratified, auditable job sample for the three-prompt experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ensure_directories, repo_path


FIELDS = ["job_title", "salary_range", "work_location", "hard_skills", "soft_skills", "education", "experience", "ai_tech_stack"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-company", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    jobs = pd.read_csv(repo_path("data", "processed", "jobs_clean.csv"), dtype=str, keep_default_na=False)
    sampled = (jobs.groupby("company_code", group_keys=False)
               .apply(lambda group: group.sample(n=min(args.per_company, len(group)), random_state=args.seed))
               .reset_index(drop=True))
    columns = ["job_id", "company_code", "company_name", "platform", "job_title", "salary_raw", "location_raw", "education_raw", "experience_raw", "skills_raw", "ai_tech_stack_raw", "job_description_raw", "source_workbook", "source_sheet", "source_row"]
    sampled = sampled[[column for column in columns if column in sampled.columns]]
    interim = repo_path("data", "interim")
    reference = repo_path("data", "reference")
    ensure_directories([interim, reference])
    sampled.to_csv(interim / "prompt_experiment_sample.csv", index=False, encoding="utf-8")
    gold = pd.DataFrame({"job_id": sampled["job_id"], "annotator": "", **{field: "" for field in FIELDS}, "annotation_notes": ""})
    gold.to_csv(reference / "gold_job_annotations.csv", index=False, encoding="utf-8")
    print(f"Wrote {len(sampled)} stratified sampled jobs and blank gold annotation sheet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
