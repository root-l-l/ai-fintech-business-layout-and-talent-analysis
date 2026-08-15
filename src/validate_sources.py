"""Validate research tables before data collection or analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from common import read_csv, repo_path, write_json


REQUIRED_SCREENING = {"company_code", "company_name", "segment", "screening_status", "evidence_url", "evidence_access_date"}
REQUIRED_SOURCES = {"source_id", "company_code", "source_type", "title", "url", "access_date", "local_path", "status"}


def nonempty_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_dates(frame: pd.DataFrame, column: str, label: str, errors: list[str]) -> None:
    values = frame[column].replace("", pd.NA).dropna()
    bad = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce").isna()
    if bad.any():
        errors.append(f"{label}.{column} has {bad.sum()} invalid ISO date(s).")


def main() -> int:
    screening = read_csv(repo_path("data", "reference", "company_screening.csv"))
    sources = read_csv(repo_path("data", "reference", "source_registry.csv"))
    errors: list[str] = []
    for required, frame, name in ((REQUIRED_SCREENING, screening, "company_screening"), (REQUIRED_SOURCES, sources, "source_registry")):
        missing = required - set(frame.columns)
        if missing:
            errors.append(f"{name} missing columns: {sorted(missing)}")
    if not errors:
        if screening["company_code"].duplicated().any():
            errors.append("company_screening has duplicate company_code values.")
        invalid_status = set(screening["screening_status"]) - {"verified", "pending", "excluded"}
        if invalid_status:
            errors.append(f"Unknown screening status: {sorted(invalid_status)}")
        for name, frame, column in (("company_screening", screening, "evidence_url"), ("source_registry", sources, "url")):
            if not frame[column].map(nonempty_url).all():
                errors.append(f"{name}.{column} includes malformed URL(s).")
        validate_dates(screening, "evidence_access_date", "company_screening", errors)
        validate_dates(sources, "access_date", "source_registry", errors)
        unknown_codes = set(sources["company_code"]) - set(screening["company_code"])
        if unknown_codes:
            errors.append(f"source_registry references unknown company code(s): {sorted(unknown_codes)}")
    report = {
        "status": "passed" if not errors else "failed",
        "verified_company_count": int((screening.get("screening_status", pd.Series(dtype=str)) == "verified").sum()),
        "source_count": int(len(sources)),
        "errors": errors,
    }
    output = repo_path("outputs", "tables", "source_validation.json")
    write_json(output, report)
    print(f"Validation {report['status']}: {output}")
    for error in errors:
        print(f"ERROR: {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
