"""Clean job records without inferring missing values."""

from __future__ import annotations

import re

import pandas as pd

from common import ensure_directories, repo_path


INPUT = repo_path("data", "raw", "jobs", "jobs_raw.csv")
OUTPUT = repo_path("data", "processed", "jobs_clean.csv")
COLUMNS = ["company_code", "job_id", "platform", "job_title", "salary_raw", "location_raw", "education_raw", "experience_raw", "job_description_raw", "job_url", "publish_time_raw", "crawl_date", "source_id"]


def normalise_space(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def main() -> int:
    ensure_directories([INPUT.parent, OUTPUT.parent])
    if not INPUT.exists():
        pd.DataFrame(columns=COLUMNS + ["dedup_key", "cleaning_note"]).to_csv(OUTPUT, index=False, encoding="utf-8")
        print(f"No raw job file found. Created empty schema: {OUTPUT}")
        return 0
    jobs = pd.read_csv(INPUT, dtype=str, keep_default_na=False)
    missing = set(COLUMNS) - set(jobs.columns)
    if missing:
        raise ValueError(f"jobs_raw.csv missing columns: {sorted(missing)}")
    for column in ["job_title", "location_raw", "education_raw", "experience_raw", "job_description_raw"]:
        jobs[column] = jobs[column].map(normalise_space)
    jobs["dedup_key"] = (jobs["company_code"] + "|" + jobs["job_title"].str.lower() + "|" + jobs["location_raw"] + "|" + jobs["job_description_raw"].str[:500]).map(normalise_space)
    before = len(jobs)
    jobs = jobs.drop_duplicates(subset=["dedup_key"], keep="first").copy()
    jobs["company_code_missing"] = jobs["company_code"].eq("")
    jobs["cleaning_note"] = "Whitespace normalized; duplicates removed by company/title/location/description-prefix key. Company code is mapped only from final_sample_pool.csv."
    jobs.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"Wrote {len(jobs)} jobs to {OUTPUT}; removed {before - len(jobs)} duplicate(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
