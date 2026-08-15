"""Produce descriptive recruitment tables from the user-provided job workbook.

The script reports observed records only. "AI-related" is a transparent lexical
flag used for descriptive grouping, not a claim that a job is an AI position.
"""

from __future__ import annotations

import re

import pandas as pd

from common import ensure_directories, repo_path


AI_PATTERN = re.compile(r"(?:\bAI\b|人工智能|大模型|模型|智能体|agent|机器学习|深度学习|NLP|自然语言|知识图谱|计算机视觉|算法|风控)", re.IGNORECASE)
EMPTY_MARKERS = {"", "无", "无明确要求", "无相关要求", "nan", "暂无"}


def observed_ai_reference(value: object) -> bool:
    text = str(value or "").strip()
    return text.lower() not in EMPTY_MARKERS and bool(AI_PATTERN.search(text))


def first_location(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "未披露"
    return re.split(r"[、，,/\s]+", text)[0]


def main() -> int:
    jobs = pd.read_csv(repo_path("data", "processed", "jobs_clean.csv"), dtype=str, keep_default_na=False)
    pool = pd.read_csv(repo_path("data", "reference", "final_sample_pool.csv"), dtype=str, keep_default_na=False)
    jobs["ai_reference_flag"] = jobs["ai_tech_stack_raw"].map(observed_ai_reference)
    jobs["primary_location"] = jobs["location_raw"].map(first_location)
    company_counts = jobs.groupby(["company_code", "company_name"], dropna=False).agg(
        job_records=("job_id", "count"),
        ai_reference_records=("ai_reference_flag", "sum"),
        platform_count=("platform", lambda x: x.replace("", pd.NA).nunique()),
        missing_salary=("salary_raw", lambda x: int(x.eq("").sum())),
        missing_location=("location_raw", lambda x: int(x.eq("").sum())),
        missing_education=("education_raw", lambda x: int(x.eq("").sum())),
        missing_experience=("experience_raw", lambda x: int(x.eq("").sum())),
        missing_ai_stack=("ai_tech_stack_raw", lambda x: int(x.eq("").sum())),
    ).reset_index()
    company_counts["ai_reference_share"] = (company_counts["ai_reference_records"] / company_counts["job_records"]).round(4)
    coverage = pool.merge(company_counts, on=["company_code", "company_name"], how="left")
    coverage["job_records"] = coverage["job_records"].fillna(0).astype(int)
    coverage["ai_reference_records"] = coverage["ai_reference_records"].fillna(0).astype(int)
    coverage["coverage_status"] = coverage["job_records"].map(lambda n: "有公开岗位记录" if n > 0 else "本次导入数据未覆盖")
    locations = jobs.groupby("primary_location").size().reset_index(name="job_records").sort_values("job_records", ascending=False)
    education = jobs.assign(education_group=jobs["education_raw"].replace("", "未披露")).groupby("education_group").size().reset_index(name="job_records").sort_values("job_records", ascending=False)
    quality = pd.DataFrame([{
        "raw_imported_records": int(pd.read_csv(repo_path("data", "raw", "jobs", "jobs_raw.csv"), dtype=str, keep_default_na=False).shape[0]),
        "deduplicated_records": int(len(jobs)),
        "duplicates_removed": int(pd.read_csv(repo_path("data", "raw", "jobs", "jobs_raw.csv"), dtype=str, keep_default_na=False).shape[0] - len(jobs)),
        "covered_companies": int((coverage["job_records"] > 0).sum()),
        "sample_pool_companies": int(len(pool)),
        "uncovered_companies": int((coverage["job_records"] == 0).sum()),
        "ai_reference_records": int(jobs["ai_reference_flag"].sum()),
        "ai_reference_definition": "AI技术栈字段含AI/人工智能/大模型/智能体/机器学习/深度学习/NLP/自然语言/知识图谱/计算机视觉/算法/风控等词，且不为无相关表述。",
    }])
    output = repo_path("outputs", "tables")
    ensure_directories([output])
    coverage.to_csv(output / "recruitment_company_coverage.csv", index=False, encoding="utf-8")
    locations.to_csv(output / "recruitment_location_distribution.csv", index=False, encoding="utf-8")
    education.to_csv(output / "recruitment_education_distribution.csv", index=False, encoding="utf-8")
    quality.to_csv(output / "recruitment_data_quality.csv", index=False, encoding="utf-8")
    print(f"Wrote recruitment coverage and quality tables to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
