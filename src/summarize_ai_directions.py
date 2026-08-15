"""Aggregate pretrained-embedding matches into transparent company AI directions."""

from __future__ import annotations

import pandas as pd

from common import ensure_directories, repo_path


def main() -> int:
    scores = pd.read_csv(repo_path("outputs", "tables", "semantic_keyword_scores.csv"), dtype=str, keep_default_na=False)
    pool = pd.read_csv(repo_path("data", "reference", "final_sample_pool.csv"), dtype=str, keep_default_na=False)
    scores["similarity"] = pd.to_numeric(scores["similarity"], errors="coerce")
    category = scores.groupby(["company_code", "category"], as_index=False).agg(
        mean_similarity=("similarity", "mean"),
        matched_keywords=("keyword", lambda value: "、".join(dict.fromkeys(value))),
        evidence_sections=("section", lambda value: "、".join(dict.fromkeys(value))),
    )
    category["category_rank"] = category.groupby("company_code")["mean_similarity"].rank(method="first", ascending=False).astype(int)
    directions = category[category["category_rank"] <= 3].sort_values(["company_code", "category_rank"])
    company = directions.groupby("company_code", as_index=False).agg(
        core_ai_directions=("category", lambda value: "；".join(value)),
        direction_keywords=("matched_keywords", lambda value: "；".join(value)),
        evidence_sections=("evidence_sections", lambda value: "；".join(value)),
    )
    company = pool.merge(company, on="company_code", how="left")
    company["interpretation_note"] = "核心方向由预训练句向量模型在四类年报章节中与研究词库的平均语义相似度前3类别确定；仅表示文本布局重点，需结合年报原文页码及产品页复核，不等同于收入占比或技术性能。"
    output = repo_path("outputs", "tables")
    ensure_directories([output])
    category.sort_values(["company_code", "category_rank"]).to_csv(output / "company_ai_direction_category_scores.csv", index=False, encoding="utf-8")
    company.to_csv(output / "company_core_ai_directions.csv", index=False, encoding="utf-8")
    print(f"Wrote {len(category)} category score rows and {len(company)} company direction rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
