"""Verify first-stage AI product-line claims against supplied primary materials.

Verification is intentionally entity-level. A company-level product-page
snapshot does not prove a named product; a claim is fully text-verified only
when every configured distinctive product anchor is found in the same company's
specified annual-report corpus or supplied website text snapshot. Screenshot-
only evidence is retained for manual review, not OCR-derived inference.
"""

from __future__ import annotations

import re

import pandas as pd

from common import ensure_directories, repo_path


ANCHORS = {
    "600570.SH": ("LightGPT", "WarrenQ"),
    "300033.SZ": ("问财金融大模型", "同创智能体平台"),
    "300059.SZ": ("妙想金融大模型", "妙想投研助理"),
    "600446.SH": ("K-GPT", "AI-KOCA"),
    "300377.SZ": ("AI-Agent专业服务器",),
    "300674.SZ": ("星睿智调", "ChatBI"),
    "000555.SZ": ("企业级智能体中台", "Skillbase Agent OS"),
    "300348.SZ": ("分布式AI核心系统",),
    "300872.SZ": ("天策", "天元金融大模型"),
    "300663.SZ": ("魔聚企业级智能体底座",),
    "002987.SZ": ("金融知识智能体", "数字员工"),
    "603927.SH": ("中科文澜大模型", "云原生寿险核心系统"),
    "688590.SH": ("新知大模型", "AI一体机"),
    "002230.SZ": ("星火大模型金融解决方案",),
    "300229.SZ": ("拓天金融大模型",),
}


def normalize(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]", "", value).lower()


def matching_sections(sections: pd.DataFrame, anchor: str) -> pd.DataFrame:
    term = normalize(anchor)
    return sections[sections["text"].map(lambda text: term and term in normalize(text))]


def context(text: str, anchor: str) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    term = normalize(anchor)
    position = normalize(flat).find(term)
    # Normalization changes offsets around punctuation. A short source page
    # excerpt remains auditable even if the exact offset cannot be restored.
    return flat[max(0, position - 80) : position + 260] if position >= 0 else flat[:340]


def main() -> int:
    claims = pd.read_csv(repo_path("data", "reference", "ai_product_lines.csv"), dtype=str, keep_default_na=False)
    sections = pd.read_csv(repo_path("data", "interim", "annual_report_sections_final15.csv"), dtype=str, keep_default_na=False)
    text_snapshots = pd.read_csv(repo_path("data", "processed", "product_page_snapshots.csv"), dtype=str, keep_default_na=False).set_index("company_code")
    screenshots = pd.read_csv(repo_path("data", "processed", "product_page_screenshots.csv"), dtype=str, keep_default_na=False)
    screenshot_counts = screenshots.groupby("company_code").size().to_dict()
    rows: list[dict[str, object]] = []
    for _, claim in claims.iterrows():
        code = claim["company_code"]
        anchors = ANCHORS[code]
        company_sections = sections[sections["company_code"].eq(code)]
        snapshot_text = text_snapshots.loc[code, "snapshot_text"] if code in text_snapshots.index else ""
        annual_matches = {anchor: matching_sections(company_sections, anchor) for anchor in anchors}
        snapshot_matches = {anchor: normalize(anchor) in normalize(snapshot_text) for anchor in anchors}
        found = [anchor for anchor in anchors if not annual_matches[anchor].empty or snapshot_matches[anchor]]
        if len(found) == len(anchors):
            status = "all_distinctive_product_entities_text_verified"
        elif found:
            status = "partial_distinctive_product_entity_text_verification"
        else:
            status = "no_distinctive_product_entity_text_match_found"
        page_notes = []
        excerpts = []
        for anchor in anchors:
            sections_found = annual_matches[anchor]
            if not sections_found.empty:
                first_section = sections_found.iloc[0]
                page_notes.append(f"{anchor}:{first_section['section']} p{first_section['page_start']}-{first_section['page_end']}")
                excerpts.append(f"{anchor} [{first_section['section']} p.{first_section['page_start']}-{first_section['page_end']}] {context(first_section['text'], anchor)}")
            elif snapshot_matches[anchor]:
                page_notes.append(f"{anchor}:website_text_snapshot")
                excerpts.append(f"{anchor} [website text snapshot] {context(snapshot_text, anchor)}")
        rows.append({
            **claim.to_dict(),
            "distinctive_product_anchors": "；".join(anchors),
            "verified_anchors": "；".join(found),
            "annual_report_match_locations": "；".join(page_notes),
            "evidence_excerpt": "\n".join(excerpts),
            "website_text_snapshot_available": code in text_snapshots.index,
            "website_screenshot_count": int(screenshot_counts.get(code, 0)),
            "verification_status": status,
            "verification_scope": "仅验证产品实体锚点是否在用户提供的指定年报章节语料或官网文字快照中出现；应用场景、功能和商业化信息仍需逐页或段落对应核验。",
        })
    result = pd.DataFrame(rows)
    output = repo_path("outputs", "tables", "ai_product_line_verification.csv")
    ensure_directories([output.parent])
    result.to_csv(output, index=False, encoding="utf-8")
    print(f"Verified {len(result)} product-line claims against the specified annual-report corpus and supplied text snapshots: {result['verification_status'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
