"""Copy saved LLM JSON outputs into model-specific folders and update the ledger.

Early batch runs used paths containing only prompt, run and job ID.  That can
overwrite a previous provider's output when the same job is rerun with another
model.  This one-time migration preserves every existing JSON before future
calls use a model-specific destination.
"""

from __future__ import annotations

import shutil

import pandas as pd

from common import ensure_directories, repo_path
from run_prompt_batch import safe_model_id


def main() -> int:
    ledger_path = repo_path("data", "reference", "llm_outputs.csv")
    if not ledger_path.exists():
        raise SystemExit(f"No LLM ledger found: {ledger_path}")
    runs = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
    if runs.empty:
        print("LLM ledger is empty; no paths migrated.")
        return 0
    backup_path = repo_path("data", "reference", "llm_outputs_before_model_path_migration.csv")
    if not backup_path.exists():
        shutil.copy2(ledger_path, backup_path)
    changed = 0
    for index, row in runs.iterrows():
        source = repo_path(*row["parsed_json_path"].split("/"))
        if not source.exists():
            raise FileNotFoundError(f"Missing JSON for ledger row {index}: {source}")
        destination = repo_path(
            "data",
            "interim",
            "llm_outputs",
            safe_model_id(row["model_id"]),
            row["prompt_version"],
            row["run_id"],
            f"{row['job_id']}.json",
        )
        ensure_directories([destination.parent])
        if not destination.exists():
            shutil.copy2(source, destination)
        relative_destination = str(destination.relative_to(repo_path()))
        if row["parsed_json_path"] != relative_destination:
            runs.at[index, "parsed_json_path"] = relative_destination
            changed += 1
    runs.to_csv(ledger_path, index=False, encoding="utf-8")
    print(f"Migrated {changed} ledger paths; original ledger backed up at {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
