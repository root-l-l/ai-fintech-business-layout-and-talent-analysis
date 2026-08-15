"""Create a small stratified subset for a second prompt run.

The first run evaluates extraction quality on all 43 sampled jobs.  A second
run on one reproducibly selected job per company is sufficient to report a
transparent, lower-cost stability check; it is not presented as full-sample
stability.
"""

from __future__ import annotations

import argparse

import pandas as pd

from common import ensure_directories, repo_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    sample = pd.read_csv(repo_path("data", "interim", "prompt_experiment_sample.csv"), dtype=str, keep_default_na=False)
    rows = []
    for _, group in sample.groupby("company_code", sort=True):
        rows.append(group.sample(n=1, random_state=args.seed))
    stability = pd.concat(rows, ignore_index=True).sort_values(["company_code", "job_id"])
    output = repo_path("data", "interim", "prompt_stability_sample.csv")
    ensure_directories([output.parent])
    stability.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(stability)} one-job-per-company stability records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
