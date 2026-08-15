"""Archive user-provided website-product screenshots with traceable provenance.

The screenshots are supplementary product evidence. This script intentionally
does not infer products, functions, or customers from images: those claims need
either a reviewed transcription or a direct citation to the original page.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from common import ensure_directories, repo_path, sha256_file


COMPANY_PATTERNS = {
    "9.天阳科技-截图": "天阳科技",
    "10.科蓝软件-截图": "科蓝软件",
    "11.京北方-截图": "京北方",
    "12.中科软-截图": "中科软",
}
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}


def screenshot_page_number(path: Path) -> int:
    """Return the numbered screenshot suffix; unnumbered files are page 1."""

    match = re.search(r"截图(\d+)$", path.stem)
    return int(match.group(1)) if match else 1


def company_for_screenshot(path: Path) -> str | None:
    for prefix, company_name in COMPANY_PATTERNS.items():
        if path.stem.startswith(prefix):
            return company_name
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive website-product screenshot evidence without making image-derived claims.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing original screenshot image files.")
    parser.add_argument("--raw-dir", type=Path, default=repo_path("data", "raw", "product_page_screenshots"))
    parser.add_argument("--output-path", type=Path, default=repo_path("data", "processed", "product_page_screenshots.csv"))
    parser.add_argument("--coverage-path", type=Path, default=repo_path("outputs", "tables", "product_screenshot_coverage.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Screenshot directory not found: {args.input_dir}")
    pool = pd.read_csv(repo_path("data", "reference", "final_sample_pool.csv"), dtype=str, keep_default_na=False)
    code_map = dict(zip(pool["company_name"], pool["company_code"]))
    images = sorted(
        (path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: (company_for_screenshot(path) or "", screenshot_page_number(path), path.name),
    )
    ensure_directories([args.raw_dir, args.output_path.parent, args.coverage_path.parent])
    rows: list[dict[str, object]] = []
    unmatched: list[str] = []
    for image in images:
        company_name = company_for_screenshot(image)
        if company_name is None:
            unmatched.append(image.name)
            continue
        destination_dir = args.raw_dir / code_map[company_name]
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / image.name
        if not destination.exists() or sha256_file(destination) != sha256_file(image):
            shutil.copy2(image, destination)
        with Image.open(destination) as picture:
            width, height = picture.size
            image_format = picture.format
        rows.append({
            "company_code": code_map[company_name],
            "company_name": company_name,
            "screenshot_page": screenshot_page_number(image),
            "source_filename": image.name,
            "source_path": str(image),
            "local_path": str(destination.relative_to(repo_path())),
            "sha256": sha256_file(destination),
            "pixel_width": width,
            "pixel_height": height,
            "image_format": image_format,
            "evidence_status": "user_provided_product_page_screenshot",
            "citation_note": "引用具体产品、能力、案例或数值前，应回到本截图可见位置或对应官网页面人工复核。",
        })
    if unmatched:
        raise ValueError(f"Unmatched screenshot filenames: {unmatched}")
    frame = pd.DataFrame(rows).sort_values(["company_code", "screenshot_page", "source_filename"])
    frame.to_csv(args.output_path, index=False, encoding="utf-8")
    summary = frame.groupby(["company_code", "company_name"], as_index=False).agg(
        screenshot_count=("source_filename", "count"),
        total_pixels=("pixel_width", lambda values: int((frame.loc[values.index, "pixel_width"] * frame.loc[values.index, "pixel_height"]).sum())),
    )
    coverage = pool[["company_code", "company_name", "segment"]].merge(summary, on=["company_code", "company_name"], how="left")
    coverage["screenshot_count"] = coverage["screenshot_count"].fillna(0).astype(int)
    coverage["screenshot_status"] = coverage["screenshot_count"].map(lambda count: "已导入原始截图" if count else "未提供截图")
    coverage.to_csv(args.coverage_path, index=False, encoding="utf-8")
    print(f"Archived {len(frame)} screenshot(s) for {frame['company_name'].nunique()} company/companies.")
    print(f"Wrote screenshot manifest to {args.output_path} and coverage table to {args.coverage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
