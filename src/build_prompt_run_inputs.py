"""Build JSONL prompt inputs for each sampled job and prompt version.

This script does not call an LLM. It produces the exact input payloads that
must be supplied to an LLM, together with identifiers needed for evaluation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from common import ensure_directories, repo_path
import pandas as pd


PROMPT_FILES = {"P1": "prompt_v1.md", "P2": "prompt_v2.md", "P3": "prompt_v3.md"}


def field_line(label: str, value: object) -> str:
    """Render an observed job-posting header field without filling blanks."""
    return f"{label}：{str(value).strip()}"


def build_full_posting_text(row: pd.Series) -> str:
    """Reconstruct a source posting from its header and description fields.

    The user-provided workbook stores title, stated salary and location in
    columns separate from the description body. They are observed recruitment
    fields, not gold labels. Including them makes every evaluated field
    available to the LLM.
    """
    header = [
        field_line("职位名称", row["job_title"]),
        field_line("薪资范围", row["salary_raw"]),
        field_line("工作地点", row["location_raw"]),
        "职位描述：",
    ]
    return "\n".join(header + [str(row["job_description_raw"]).strip()])


def main() -> int:
    sample = pd.read_csv(repo_path("data", "interim", "prompt_experiment_sample.csv"), dtype=str, keep_default_na=False)
    output_dir = repo_path("data", "interim", "prompt_inputs")
    ensure_directories([output_dir])
    for version, filename in PROMPT_FILES.items():
        template = (repo_path("docs", "prompts", filename)).read_text(encoding="utf-8")
        path = output_dir / f"{version.lower()}_inputs.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for _, row in sample.iterrows():
                prompt = template.replace("{job_description_raw}", build_full_posting_text(row))
                payload = {
                    "job_id": row["job_id"],
                    "prompt_version": version,
                    "company_code": row["company_code"],
                    "input_profile": "full_posting_header_and_description_v1",
                    "model_input": prompt,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"Wrote {len(sample)} inputs to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
