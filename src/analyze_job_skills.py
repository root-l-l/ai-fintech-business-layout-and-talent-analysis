"""Create transparent keyword counts for observed recruitment text."""

from __future__ import annotations

import re

import pandas as pd

from common import ensure_directories, repo_path


SKILL_TERMS = ["Python", "Java", "C++", "SQL", "大模型", "智能体", "RAG", "NLP", "知识图谱", "机器学习", "深度学习", "算法", "风控", "数据分析", "云原生", "微服务", "Linux", "测试", "金融"]


def main() -> int:
    jobs = pd.read_csv(repo_path("data", "processed", "jobs_clean.csv"), dtype=str, keep_default_na=False)
    text = (jobs["job_title"] + " " + jobs["job_description_raw"] + " " + jobs.get("ai_tech_stack_raw", "")).fillna("")
    rows = []
    for term in SKILL_TERMS:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        mask = text.map(lambda value: bool(pattern.search(value)))
        rows.append({"keyword": term, "job_records": int(mask.sum()), "share_of_cleaned_records": round(float(mask.mean()), 4), "definition": "职位名称、技能要求、学历/经验或AI技术栈原文中的字面匹配；同一职位对同一关键词至多计一次。"})
    output = repo_path("outputs", "tables", "recruitment_skill_keyword_counts.csv")
    ensure_directories([output.parent])
    pd.DataFrame(rows).sort_values("job_records", ascending=False).to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote {len(rows)} observed keyword counts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
