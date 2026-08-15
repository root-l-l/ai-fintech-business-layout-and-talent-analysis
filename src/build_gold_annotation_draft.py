"""Build an auditable gold draft from the user-provided structured job table.

The workbook was manually structured by the research team.  Its explicit field
values are therefore adopted as the annotation source for this experiment;
``无`` and blank cells mean that the source did not mention that field.  The
draft retains provenance so a later reviewer can still correct an individual
record without losing the original source trail.
"""

from __future__ import annotations

import re

import pandas as pd

from common import ensure_directories, repo_path


EMPTY_MARKERS = {"", "无", "无明确要求", "未提及", "nan", "none", "null"}


def clean_source(value: object) -> str:
    """Return an empty value for an explicit source non-mention."""
    text = str(value).strip() if value is not None else ""
    return "" if text.lower() in EMPTY_MARKERS else text


def split_skills(value: object) -> tuple[str, str]:
    """Split the workbook's explicitly labelled hard and soft skill fields."""
    text = clean_source(value)
    if not text:
        return "", ""
    hard = ""
    soft = ""
    hard_match = re.search(r"硬技能[:：]\s*(.*?)(?=\s*软技能[:：]|$)", text, flags=re.S)
    soft_match = re.search(r"软技能[:：]\s*(.*)$", text, flags=re.S)
    if hard_match:
        hard = hard_match.group(1).strip()
    if soft_match:
        soft = soft_match.group(1).strip()
    # Some job rows have an unlabelled skill description. Keep it visible for
    # human review rather than guessing how to classify it.
    if not hard and not soft:
        hard = text
    return hard, soft


def main() -> int:
    sample_path = repo_path("data", "interim", "prompt_experiment_sample.csv")
    sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
    skills = sample["skills_raw"].map(split_skills)
    draft = pd.DataFrame({
        "job_id": sample["job_id"],
        "annotator": "user_provided_recruitment_workbook",
        "job_title": sample["job_title"].map(clean_source),
        "salary_range": sample["salary_raw"].map(clean_source),
        "work_location": sample["location_raw"].map(clean_source),
        "hard_skills": skills.map(lambda item: item[0]),
        "soft_skills": skills.map(lambda item: item[1]),
        "education": sample["education_raw"].map(clean_source),
        "experience": sample["experience_raw"].map(clean_source),
        "ai_tech_stack": sample["ai_tech_stack_raw"].map(clean_source),
        "review_status": "user_provided_structured_annotation",
        "annotation_notes": (
            "金标准直接采用用户提供的结构化招聘表；空白、无、无明确要求均按原文未提及处理。"
            "来源：" + sample["source_workbook"].astype(str) + "/" + sample["source_sheet"].astype(str)
            + "，行 " + sample["source_row"].astype(str)
        ),
    })
    output = repo_path("data", "interim", "gold_annotation_draft.csv")
    ensure_directories([output.parent])
    draft.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(draft)} reviewable gold-annotation drafts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
