"""Turn the user-provided product-link workbook into auditable product evidence tables."""

from __future__ import annotations

import pandas as pd

from common import ensure_directories, repo_path


def main() -> int:
    products = pd.read_csv(repo_path("data", "reference", "product_pages.csv"), dtype=str, keep_default_na=False)
    pool = pd.read_csv(repo_path("data", "reference", "final_sample_pool.csv"), dtype=str, keep_default_na=False)
    result = pool.merge(products, on="company_name", how="left", validate="one_to_one")
    result["product_link_status"] = result["product_url"].map(lambda x: "已提供" if x else "未提供")
    output = repo_path("outputs", "tables", "product_page_coverage.csv")
    ensure_directories([output.parent])
    result.to_csv(output, index=False, encoding="utf-8")
    print(f"Wrote product-link coverage for {len(result)} sample companies to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
