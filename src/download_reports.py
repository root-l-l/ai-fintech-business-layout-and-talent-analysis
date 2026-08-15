"""Download only annual reports explicitly registered in the source registry."""

from __future__ import annotations

import argparse
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from common import read_csv, repo_path, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    registry_path = repo_path("data", "reference", "source_registry.csv")
    registry = read_csv(registry_path)
    reports = registry[registry["source_type"] == "annual_report"].copy()
    if reports.empty:
        print("No annual-report rows registered; nothing downloaded.")
        return 0
    changed = False
    for index, row in reports.iterrows():
        destination = repo_path(*row["local_path"].split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not args.overwrite:
            registry.loc[index, "sha256"] = sha256_file(destination)
            registry.loc[index, "status"] = "downloaded"
            continue
        try:
            request = Request(row["url"], headers={"User-Agent": "Mozilla/5.0 (academic reproducible research)"})
            with urlopen(request, timeout=args.timeout) as response:
                content = response.read()
            if not content.startswith(b"%PDF"):
                raise ValueError("Response is not a PDF; registry URL may be invalid or blocked.")
            destination.write_bytes(content)
            registry.loc[index, "sha256"] = sha256_file(destination)
            registry.loc[index, "status"] = "downloaded"
            changed = True
            print(f"Downloaded {row['source_id']} -> {destination}")
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            registry.loc[index, "status"] = "download_failed"
            print(f"FAILED {row['source_id']}: {error}", file=sys.stderr)
    if changed or (registry["status"] != read_csv(registry_path)["status"]).any():
        registry.to_csv(registry_path, index=False, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
