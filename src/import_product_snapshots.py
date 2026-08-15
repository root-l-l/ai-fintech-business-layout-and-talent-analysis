"""Import user-provided website-product text snapshots with provenance.

Website snapshots are supplementary product evidence. They are deliberately not
merged into the annual-report semantic corpus, which remains limited to the
specified annual-report sections. The importer copies supplied DOCX snapshots,
extracts paragraph-level text with Python's standard library, and records the
source file and SHA-256 for every company section.
"""

from __future__ import annotations

import argparse
import re
import shutil
import xml.etree.ElementTree as element_tree
import zipfile
from pathlib import Path

import pandas as pd

from common import ensure_directories, repo_path, sha256_file


WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
EXPECTED_COMPANIES = {
    "1-4 产品文字版.docx": ("恒生电子", "同花顺", "东方财富", "金证股份"),
    "5-8产品全部.docx": ("赢时胜", "宇信科技", "神州信息", "长亮科技"),
    "13-15产品.docx": ("新致软件", "科大讯飞", "拓尔思"),
}


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def docx_paragraphs(path: Path) -> list[str]:
    """Read visible paragraph text from a DOCX without adding a platform dependency."""

    with zipfile.ZipFile(path) as archive:
        document = archive.read("word/document.xml")
    root = element_tree.fromstring(document)
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", WORD_NAMESPACE):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", WORD_NAMESPACE))
        text = compact(text)
        if text:
            paragraphs.append(text)
    return paragraphs


def is_heading_for_company(paragraph: str, company_name: str) -> bool:
    """Identify short company headings while avoiding ordinary product mentions."""

    if company_name not in paragraph or len(paragraph) > 100:
        return False
    normalized = re.sub(r"^[（(]?\s*(?:第?[0-9一二三四五六七八九十]+[.、．）)]\s*)?", "", paragraph)
    return normalized.startswith(company_name)


def split_company_sections(paragraphs: list[str], expected_companies: tuple[str, ...]) -> dict[str, list[str]]:
    starts: list[tuple[int, str]] = []
    for index, paragraph in enumerate(paragraphs):
        for company_name in expected_companies:
            if is_heading_for_company(paragraph, company_name):
                starts.append((index, company_name))
                break
    # A company can appear in a page title and a numbered heading. Keep the
    # first heading only, preserving source order for non-overlapping sections.
    first_starts: list[tuple[int, str]] = []
    seen: set[str] = set()
    for index, company_name in starts:
        if company_name not in seen:
            first_starts.append((index, company_name))
            seen.add(company_name)
    sections: dict[str, list[str]] = {}
    for position, (start, company_name) in enumerate(first_starts):
        end = first_starts[position + 1][0] if position + 1 < len(first_starts) else len(paragraphs)
        sections[company_name] = paragraphs[start:end]
    return sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import DOCX website-product snapshots and retain their provenance.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="One or more user-provided DOCX product snapshot files.")
    parser.add_argument("--raw-dir", type=Path, default=repo_path("data", "raw", "product_page_snapshots"))
    parser.add_argument("--processed-path", type=Path, default=repo_path("data", "processed", "product_page_snapshots.csv"))
    parser.add_argument("--manifest-path", type=Path, default=repo_path("data", "reference", "product_page_snapshot_manifest.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [str(path) for path in args.inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Product snapshot input(s) not found: {missing}")
    pool = pd.read_csv(repo_path("data", "reference", "final_sample_pool.csv"), dtype=str, keep_default_na=False)
    code_map = dict(zip(pool["company_name"], pool["company_code"]))
    ensure_directories([args.raw_dir, args.processed_path.parent, args.manifest_path.parent, repo_path("outputs", "tables")])
    snapshots: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for source in args.inputs:
        expected = EXPECTED_COMPANIES.get(source.name, ())
        if not expected:
            raise ValueError(f"No expected-company mapping is configured for {source.name}; add an explicit mapping before import.")
        destination = args.raw_dir / source.name
        if not destination.exists() or sha256_file(destination) != sha256_file(source):
            shutil.copy2(source, destination)
        paragraphs = docx_paragraphs(destination)
        sections = split_company_sections(paragraphs, expected)
        manifest.append({
            "source_file": source.name,
            "local_path": str(destination.relative_to(repo_path())),
            "sha256": sha256_file(destination),
            "expected_companies": "、".join(expected),
            "extracted_companies": "、".join(sections),
            "paragraph_count": len(paragraphs),
            "status": "segmentation_complete" if set(sections) == set(expected) else "segmentation_needs_review",
        })
        for company_name, section_paragraphs in sections.items():
            snapshots.append({
                "company_code": code_map[company_name],
                "company_name": company_name,
                "source_file": source.name,
                "local_path": str(destination.relative_to(repo_path())),
                "source_sha256": sha256_file(destination),
                "snapshot_text": "\n".join(section_paragraphs),
                "paragraph_count": len(section_paragraphs),
                "evidence_status": "user_provided_product_text_snapshot",
                "citation_note": "引用具体产品、能力或客户案例前，应回到该快照的原始段落或对应官网页面复核。",
            })
    snapshot_frame = pd.DataFrame(snapshots)
    manifest_frame = pd.DataFrame(manifest)
    snapshot_frame.to_csv(args.processed_path, index=False, encoding="utf-8")
    manifest_frame.to_csv(args.manifest_path, index=False, encoding="utf-8")
    coverage = pool[["company_code", "company_name", "segment"]].merge(
        snapshot_frame[["company_code", "source_file", "paragraph_count"]], on="company_code", how="left"
    )
    coverage["snapshot_status"] = coverage["source_file"].map(lambda value: "已导入文字快照" if isinstance(value, str) and value else "本次未提供可导入文字快照")
    coverage.to_csv(repo_path("outputs", "tables", "product_snapshot_coverage.csv"), index=False, encoding="utf-8")
    print(f"Imported {len(snapshot_frame)} company snapshot section(s) from {len(manifest_frame)} DOCX file(s).")
    print(f"Wrote evidence table to {args.processed_path} and coverage table to outputs/tables/product_snapshot_coverage.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
