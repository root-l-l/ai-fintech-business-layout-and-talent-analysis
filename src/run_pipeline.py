"""Run the offline-safe stages of the project pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from common import ROOT


STEPS = ["validate_sources.py", "build_keyword_lexicon.py", "clean_jobs.py", "semantic_analysis.py", "build_knowledge_graph.py", "prepare_prompts.py", "evaluate_llm.py", "score_automation_risk.py"]


def main() -> int:
    for script in STEPS:
        command = [sys.executable, str(Path(__file__).with_name(script))]
        print("Running", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)
    print("Offline-safe pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
