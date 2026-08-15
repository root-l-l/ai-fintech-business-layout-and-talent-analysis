"""Normalize stated job salaries to RMB numeric ranges without inventing values.

The original salary wording remains authoritative.  This parser only converts
explicit numeric amounts: ``1K`` is treated as 1,000 RMB.  It never converts
``面议`` or ``未公示`` into a number, and keeps daily and monthly pay separate.
"""

from __future__ import annotations

import re

import pandas as pd

from common import ensure_directories, repo_path


NO_NUMERIC_DISCLOSURE = {"", "面议", "未公示", "无", "无明确要求"}


def parse_salary_rmb(value: object) -> dict[str, object]:
    """Parse explicit salary amounts while retaining pay period and 13/14/16薪."""
    raw = str(value).strip() if value is not None else ""
    result: dict[str, object] = {
        "salary_raw": raw,
        "disclosure_status": "numeric_disclosed",
        "pay_period": "unknown",
        "rmb_min": None,
        "rmb_max": None,
        "salary_months": None,
        "parse_note": "",
    }
    if raw in NO_NUMERIC_DISCLOSURE:
        result.update({"disclosure_status": "not_disclosed", "parse_note": "No numeric salary disclosed in source."})
        return result

    compact = re.sub(r"\s+", "", raw.lower())
    if "元/天" in compact or "元／天" in compact:
        result["pay_period"] = "day"
    elif "元/月" in compact or "元／月" in compact or "k" in compact:
        result["pay_period"] = "month"

    months = re.search(r"[·・]?\s*(\d{2})\s*薪", raw)
    if months:
        result["salary_months"] = int(months.group(1))

    # Explicit K ranges are monthly RMB unless another period is stated.
    # Job boards commonly write both `15K-25K` and the abbreviated `15-25K`.
    k_match = re.search(r"(\d+(?:\.\d+)?)\s*[kK]?\s*[-~至]\s*(\d+(?:\.\d+)?)\s*[kK]", raw)
    if k_match:
        result["rmb_min"] = int(float(k_match.group(1)) * 1000)
        result["rmb_max"] = int(float(k_match.group(2)) * 1000)
        if result["pay_period"] == "unknown":
            result["pay_period"] = "month"
        return result

    # Covers explicit RMB ranges such as 3000-5000 元/月 and 200-250元/天.
    rmb_match = re.search(r"(\d+(?:\.\d+)?)\s*[-~至]\s*(\d+(?:\.\d+)?)\s*元", raw)
    if rmb_match:
        result["rmb_min"] = int(float(rmb_match.group(1)))
        result["rmb_max"] = int(float(rmb_match.group(2)))
        return result

    result.update({"disclosure_status": "unparsed_mixed_expression", "parse_note": "Mixed or non-standard salary expression; preserve source text and review manually."})
    return result


def main() -> int:
    jobs = pd.read_csv(repo_path("data", "processed", "jobs_clean.csv"), dtype=str, keep_default_na=False)
    normalized = pd.DataFrame([parse_salary_rmb(value) for value in jobs["salary_raw"]])
    output = pd.concat([jobs[["job_id", "company_name", "job_title"]].reset_index(drop=True), normalized], axis=1)
    output_path = repo_path("outputs", "tables", "job_salary_normalization.csv")
    ensure_directories([output_path.parent])
    output.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Wrote {len(output)} salary normalization rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
