"""Create node and edge tables from observed semantic-analysis results."""

from __future__ import annotations

import pandas as pd

from common import ensure_directories, repo_path


def main() -> int:
    pool = pd.read_csv(repo_path("data", "reference", "final_sample_pool.csv"), dtype=str, keep_default_na=False)
    products = pd.read_csv(repo_path("data", "reference", "product_pages.csv"), dtype=str, keep_default_na=False)
    product_lines = pd.read_csv(repo_path("data", "reference", "ai_product_lines.csv"), dtype=str, keep_default_na=False)
    page_mapping_path = repo_path("data", "processed", "screenshot_page_mapping.csv")
    screenshot_entities_path = repo_path("data", "processed", "screenshot_ai_product_entities.csv")
    stage_categories_path = repo_path("data", "processed", "stage1_product_line_technical_categories.csv")
    required_paths = [page_mapping_path, screenshot_entities_path, stage_categories_path]
    missing = [str(path.relative_to(repo_path())) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run `python src/map_screenshot_product_entities.py` before building the graph. Missing: "
            + ", ".join(missing)
        )
    page_mapping = pd.read_csv(page_mapping_path, dtype=str, keep_default_na=False)
    screenshot_entities = pd.read_csv(screenshot_entities_path, dtype=str, keep_default_na=False)
    stage_categories = pd.read_csv(stage_categories_path, dtype=str, keep_default_na=False).set_index("company_code")["technical_category"].to_dict()
    product_evidence_path = repo_path("outputs", "tables", "ai_product_line_evidence.csv")
    product_evidence = pd.read_csv(product_evidence_path, dtype=str, keep_default_na=False) if product_evidence_path.exists() else pd.DataFrame()
    financial_path = repo_path("outputs", "tables", "company_ai_financial_comparison.csv")
    financial = pd.read_csv(financial_path, dtype=str, keep_default_na=False) if financial_path.exists() else pd.DataFrame()
    scores_path = repo_path("outputs", "tables", "semantic_keyword_scores.csv")
    scores = pd.read_csv(scores_path, dtype=str, keep_default_na=False) if scores_path.exists() else pd.DataFrame(columns=["company_code", "keyword", "category", "similarity"])
    nodes = []
    for _, row in pool.iterrows():
        nodes.append({"node_id": f"company:{row['company_code']}", "node_type": "company", "label": row["company_name"], "evidence": row["source_document"]})
    edges = []
    for _, row in scores.iterrows():
        keyword_id = f"keyword:{row['keyword']}"
        nodes.append({"node_id": keyword_id, "node_type": "keyword", "label": row["keyword"], "evidence": "ai_fintech_keywords.csv"})
        edges.append({"source": f"company:{row['company_code']}", "target": keyword_id, "relation": "semantic_association", "weight": row["similarity"], "evidence": row.get("source_id", "")})
    for _, row in products.iterrows():
        matched = pool[pool["company_name"] == row["company_name"]]
        if matched.empty or not row["product_url"]:
            continue
        code = matched.iloc[0]["company_code"]
        product_id = f"product_page:{code}"
        nodes.append({"node_id": product_id, "node_type": "product_page", "label": row["product_page_title"], "evidence": row["product_url"]})
        edges.append({"source": f"company:{code}", "target": product_id, "relation": "has_product_page", "weight": "", "evidence": row["product_url"], "verification_status": "link_provided"})
    for _, row in product_lines.iterrows():
        line_id = f"product_line:{row['company_code']}:{row['product_line']}"
        category = stage_categories[row["company_code"]]
        direction_id = f"technical_category:{category}"
        nodes.append({"node_id": line_id, "node_type": "stage1_product_line_claim", "label": row["product_line"], "evidence": row["source_reference"]})
        nodes.append({"node_id": direction_id, "node_type": "technical_category", "label": category, "evidence": "stage1_product_line_technical_categories.csv"})
        status = row["verification_status"]
        matching_evidence = product_evidence[product_evidence["company_code"].eq(row["company_code"])] if not product_evidence.empty else pd.DataFrame()
        if not matching_evidence.empty:
            status = matching_evidence.iloc[0].get("entity_verification_status", matching_evidence.iloc[0]["evidence_status"])
        edges.append({"source": f"company:{row['company_code']}", "target": line_id, "relation": "offers_stage1_product_line_claim", "weight": "", "evidence": row["source_reference"], "verification_status": status})
        edges.append({"source": line_id, "target": direction_id, "relation": "maps_to_technical_category", "weight": "", "evidence": row["application_scenario"], "verification_status": status})
    for _, page in page_mapping.iterrows():
        page_id = f"product_screenshot_page:{page['company_code']}:{page['screenshot_page']}"
        nodes.append({"node_id": page_id, "node_type": "product_screenshot_page", "label": f"官网截图 p{page['screenshot_page']}：{page['page_title']}", "evidence": page["local_path"]})
        edges.append({"source": f"company:{page['company_code']}", "target": page_id, "relation": "has_product_screenshot_page", "weight": "", "evidence": page["local_path"], "verification_status": "user_provided_product_page_screenshot"})
    for _, entity in screenshot_entities.iterrows():
        entity_id = f"product_line:{entity['company_code']}:{entity['product_line']}"
        direction_id = f"technical_category:{entity['technical_category']}"
        page_id = f"product_screenshot_page:{entity['company_code']}:{entity['screenshot_page']}"
        nodes.append({"node_id": entity_id, "node_type": "page_verified_ai_product_entity", "label": entity["product_line"], "evidence": entity["source_path"]})
        nodes.append({"node_id": direction_id, "node_type": "technical_category", "label": entity["technical_category"], "evidence": "screenshot_ai_product_entities.csv"})
        edges.append({"source": f"company:{entity['company_code']}", "target": entity_id, "relation": "offers_page_verified_ai_product_entity", "weight": "", "evidence": entity["evidence_text"], "verification_status": entity["verification_status"]})
        edges.append({"source": entity_id, "target": direction_id, "relation": "maps_to_technical_category", "weight": "", "evidence": entity["application_scenario"], "verification_status": entity["verification_status"]})
        edges.append({"source": page_id, "target": entity_id, "relation": "documents_product_entity", "weight": "", "evidence": entity["evidence_text"], "verification_status": entity["verification_status"]})
    if not product_evidence.empty:
        for _, evidence in product_evidence.iterrows():
            code = evidence["company_code"]
            if evidence["text_snapshot_path"]:
                evidence_id = f"product_snapshot:{code}"
                nodes.append({"node_id": evidence_id, "node_type": "product_snapshot", "label": "官网文字快照", "evidence": evidence["text_snapshot_path"]})
                edges.append({"source": f"company:{code}", "target": evidence_id, "relation": "has_product_evidence", "weight": "", "evidence": evidence["text_snapshot_path"], "verification_status": "raw_text_snapshot"})
            if int(evidence["screenshot_count"] or 0) > 0:
                evidence_id = f"product_screenshots:{code}"
                nodes.append({"node_id": evidence_id, "node_type": "product_screenshots", "label": f"官网截图（{evidence['screenshot_count']}张）", "evidence": evidence["screenshot_paths"]})
                edges.append({"source": f"company:{code}", "target": evidence_id, "relation": "has_product_evidence", "weight": "", "evidence": evidence["screenshot_paths"], "verification_status": "raw_image_snapshot"})
    if not financial.empty:
        for _, profile in financial.iterrows():
            group_id = f"asset_group:{profile['asset_size_group']}"
            nodes.append({"node_id": group_id, "node_type": "asset_size_group", "label": profile["asset_size_group"], "evidence": "company_ai_financial_comparison.csv"})
            edges.append({"source": f"company:{profile['company_code']}", "target": group_id, "relation": "belongs_to_asset_size_group", "weight": profile["total_assets_yuan"], "evidence": "2025 annual-report financial summary table", "verification_status": "descriptive_grouping"})
    output_dir = repo_path("outputs", "knowledge_graph")
    ensure_directories([output_dir])
    pd.DataFrame(nodes, columns=["node_id", "node_type", "label", "evidence"]).drop_duplicates("node_id").to_csv(output_dir / "nodes.csv", index=False, encoding="utf-8")
    pd.DataFrame(edges, columns=["source", "target", "relation", "weight", "evidence", "verification_status"]).to_csv(output_dir / "edges.csv", index=False, encoding="utf-8")
    print(f"Wrote {len(nodes)} node observations and {len(edges)} edges to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
