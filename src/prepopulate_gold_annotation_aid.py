"""Create a source-field preannotation aid, explicitly not a human gold standard."""

from __future__ import annotations

import pandas as pd

from common import ensure_directories, repo_path


def main() -> int:
    sample = pd.read_csv(repo_path("data", "interim", "prompt_experiment_sample.csv"), dtype=str, keep_default_na=False)
    aid = pd.DataFrame({
        "job_id": sample["job_id"],
        "company_name": sample["company_name"],
        "job_title_source": sample["job_title"],
        "salary_range_source": sample["salary_raw"],
        "work_location_source": sample["location_raw"],
        "education_source": sample["education_raw"],
        "experience_source": sample["experience_raw"],
        "skills_source": sample["skills_raw"],
        "ai_tech_stack_source": sample["ai_tech_stack_raw"],
        "review_status": "not_human_validated",
        "instruction": "Copy only after checking against job_description_raw; split hard/soft skills and normalize AI stack manually.",
    })
    output = repo_path("data", "interim", "gold_annotation_aid.csv")
    ensure_directories([output.parent])
    aid.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(aid)} source-field preannotation aids to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
