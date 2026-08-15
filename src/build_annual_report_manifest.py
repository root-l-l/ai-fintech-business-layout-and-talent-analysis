"""Map user-provided annual-report filenames to the author-defined 15-company pool."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import ensure_directories, repo_path, sha256_file


FILENAME_TOKENS = {
    "恒生电子": "恒生电子", "同花顺": "同花顺", "东方财富": "东方财富", "金证股份": "金证股份", "赢时胜": "赢时胜",
    "宇信科技": "宇信科技", "神州信息": "神州信息", "长亮科技": "长亮科技", "天阳科技": "天阳科技", "科蓝软件": "科蓝软件",
    "京北方": "京北方", "中科软": "中科软", "新致软件": "新致软件", "科大讯飞": "科大讯飞", "拓尔思": "拓尔思",
}


def main() -> int:
    pool = pd.read_csv(repo_path("data", "reference", "final_sample_pool.csv"), dtype=str, keep_default_na=False)
    code_by_company = dict(zip(pool["company_name"], pool["company_code"]))
    report_dir = repo_path("data", "raw", "annual_reports")
    rows = []
    for path in sorted(report_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        matched = [company for token, company in FILENAME_TOKENS.items() if token in path.name]
        if len(matched) != 1:
            rows.append({"source_id": "", "company_code": "", "company_name": "", "report_year": "2025", "local_path": str(path.relative_to(repo_path())), "sha256": sha256_file(path), "mapping_status": "unmapped_or_ambiguous", "mapping_note": f"Matches: {matched}"})
            continue
        company = matched[0]
        code = code_by_company[company]
        rows.append({"source_id": f"AR_{code.replace('.', '_')}_2025", "company_code": code, "company_name": company, "report_year": "2025", "local_path": str(path.relative_to(repo_path())), "sha256": sha256_file(path), "mapping_status": "mapped", "mapping_note": "Mapped from user-provided filename token; original file retained."})
    result = pd.DataFrame(rows)
    output = repo_path("data", "reference", "annual_report_manifest.csv")
    ensure_directories([output.parent])
    result.to_csv(output, index=False, encoding="utf-8")
    mapped = int((result["mapping_status"] == "mapped").sum()) if not result.empty else 0
    print(f"Wrote {len(result)} manifest row(s), {mapped} mapped, to {output}")
    return 0 if mapped == len(pool) else 1


if __name__ == "__main__":
    raise SystemExit(main())
