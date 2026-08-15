"""Create evaluation gold from a human-confirmed or user-provided annotation."""

from __future__ import annotations

import pandas as pd

from common import ensure_directories, repo_path


FIELDS = [
    "job_id", "annotator", "job_title", "salary_range", "work_location",
    "hard_skills", "soft_skills", "education", "experience", "ai_tech_stack",
    "annotation_notes",
]
ACCEPTED_STATUSES = {"human_confirmed", "user_provided_structured_annotation"}


def main() -> int:
    draft_path = repo_path("data", "interim", "gold_annotation_draft.csv")
    draft = pd.read_csv(draft_path, dtype=str, keep_default_na=False)
    confirmed = draft.loc[draft["review_status"].isin(ACCEPTED_STATUSES)].copy()
    if confirmed.empty:
        raise SystemExit("No accepted annotation rows. Confirm rows or provide a structured annotation source before finalizing.")
    if not confirmed["annotator"].str.strip().any():
        raise SystemExit("Confirmed rows require a non-empty annotator field.")
    output = repo_path("data", "reference", "gold_job_annotations.csv")
    ensure_directories([output.parent])
    confirmed[FIELDS].to_csv(output, index=False, encoding="utf-8")
    print(f"Finalized {len(confirmed)} accepted gold rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
