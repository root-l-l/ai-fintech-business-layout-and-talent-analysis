"""Run prompt inputs through an OpenAI-compatible chat-completions API.

This script sends only the prepared prompt text, never the gold-standard file.
It saves the provider's raw answer, parsed JSON, request timestamps and model
identifier so evaluation remains reproducible.  API keys must be supplied via
an environment variable and are never written to the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from common import ensure_directories, repo_path


PROMPT_FILES = {"P1": "p1_inputs.jsonl", "P2": "p2_inputs.jsonl", "P3": "p3_inputs.jsonl"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_model_id(model_id: str) -> str:
    """Make a stable folder component without conflating provider model names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._") or "unknown_model"


def parse_json_reply(raw: str) -> dict[str, Any]:
    """Accept a JSON object, optionally enclosed in a Markdown code fence."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in provider response")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Provider response JSON is not an object")
    return payload


def call_model(item: dict[str, str], args: argparse.Namespace, api_key: str) -> dict[str, str]:
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": item["model_input"]}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.thinking_mode == "siliconflow":
        body["enable_thinking"] = False
    elif args.thinking_mode == "deepseek":
        body["thinking"] = {"type": "disabled"}
    if not args.no_json_mode:
        body["response_format"] = {"type": "json_object"}
    last_error = ""
    started_at = utc_now()
    for attempt in range(1, args.retries + 1):
        try:
            response = requests.post(endpoint, headers=headers, json=body, timeout=args.timeout_seconds)
            response.raise_for_status()
            api_response = response.json()
            raw = api_response["choices"][0]["message"]["content"]
            if not isinstance(raw, str):
                raise ValueError("Provider returned non-text content")
            parsed = parse_json_reply(raw)
            finished_at = utc_now()
            json_path = repo_path(
                "data",
                "interim",
                "llm_outputs",
                safe_model_id(args.model),
                item["prompt_version"],
                args.run_id,
                f"{item['job_id']}.json",
            )
            ensure_directories([json_path.parent])
            json_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
            result = {
                "job_id": item["job_id"],
                "prompt_version": item["prompt_version"],
                "run_id": args.run_id,
                "model_id": args.model,
                "started_at": started_at,
                "finished_at": finished_at,
                "raw_model_output": raw,
                "parsed_json_path": str(json_path.relative_to(repo_path())),
            }
            # With one worker, this creates a real inter-request interval for
            # lower-quota providers such as free-tier Gemini accounts.
            if args.request_interval_seconds:
                time.sleep(args.request_interval_seconds)
            return result
        except requests.HTTPError as error:
            status_code = error.response.status_code if error.response is not None else None
            # Authentication and billing failures cannot succeed by retrying.
            if status_code in {401, 402, 403}:
                detail = error.response.text[:500] if error.response is not None else str(error)
                raise RuntimeError(
                    f"{item['job_id']} {item['prompt_version']} stopped: HTTP {status_code}. "
                    f"Check API key permissions and account billing. Provider detail: {detail}"
                ) from error
            last_error = f"attempt {attempt}: {error}"
            if attempt < args.retries:
                time.sleep(min(2**attempt, 8))
        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as error:
            last_error = f"attempt {attempt}: {error}"
            if attempt < args.retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"{item['job_id']} {item['prompt_version']} failed after {args.retries} attempts: {last_error}")


def load_items(versions: list[str], job_ids_file: str | None) -> list[dict[str, str]]:
    selected_ids: set[str] | None = None
    if job_ids_file:
        frame = pd_read_csv(repo_path(*job_ids_file.split("/")))
        if not frame or "job_id" not in frame[0]:
            raise ValueError(f"{job_ids_file} has no job_id column")
        selected_ids = {str(row["job_id"]) for row in frame}
    items: list[dict[str, str]] = []
    for version in versions:
        path = repo_path("data", "interim", "prompt_inputs", PROMPT_FILES[version])
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if selected_ids is None or item["job_id"] in selected_ids:
                items.append(item)
    return items


def exclude_completed(items: list[dict[str, str]], run_id: str, model_id: str) -> list[dict[str, str]]:
    """Avoid duplicate calls only for the same model, job, prompt and run."""
    output = repo_path("data", "reference", "llm_outputs.csv")
    if not output.exists():
        return items
    with output.open(encoding="utf-8", newline="") as handle:
        completed = {
            (row["job_id"], row["prompt_version"], row["run_id"])
            for row in csv.DictReader(handle)
            if row.get("model_id") == model_id
        }
    return [item for item in items if (item["job_id"], item["prompt_version"], run_id) not in completed]


def pd_read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_results(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    output = repo_path("data", "reference", "llm_outputs.csv")
    ensure_directories([output.parent])
    exists = output.exists()
    fields = ["job_id", "prompt_version", "run_id", "model_id", "started_at", "finished_at", "raw_model_output", "parsed_json_path"]
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="API base URL without /chat/completions")
    parser.add_argument("--model", required=True, help="Actual provider model identifier")
    parser.add_argument("--run-id", required=True, help="For example R1 or R2")
    parser.add_argument("--versions", nargs="+", choices=sorted(PROMPT_FILES), default=sorted(PROMPT_FILES))
    parser.add_argument("--job-ids-file", help="Optional CSV with a job_id column, for the stability subset")
    parser.add_argument("--api-key-env", default="LLM_API_KEY", help="Environment-variable name containing the API key")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200, help="Output-token cap; 1,200 is sufficient for one job's extraction JSON")
    parser.add_argument(
        "--thinking-mode",
        choices=["none", "siliconflow", "deepseek"],
        default="none",
        help="Disable thinking using the provider-specific request format when supported",
    )
    parser.add_argument("--no-json-mode", action="store_true", help="Use only if the selected provider rejects response_format=json_object")
    parser.add_argument("--workers", type=int, default=2, help="Keep low to respect provider rate limits")
    parser.add_argument("--request-interval-seconds", type=float, default=0.0, help="Delay after each successful request; use workers=1 for a global interval")
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} in your terminal before running this script; never put an API key in a file.")
    items = exclude_completed(load_items(args.versions, args.job_ids_file), args.run_id, args.model)
    if not items:
        print("No pending prompt inputs selected; all requested job/prompt/run combinations are already saved.")
        return 0
    print(
        f"Submitting {len(items)} jobs using {args.model}, run {args.run_id}; "
        f"temperature={args.temperature}; max_tokens={args.max_tokens}; thinking_mode={args.thinking_mode}."
    )
    successful: list[dict[str, str]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(call_model, item, args, api_key) for item in items]
        for future in as_completed(futures):
            try:
                result = future.result()
                # Persist immediately. A Ctrl-C, quota error, or network loss
                # therefore never discards already-completed real API calls.
                append_results([result])
                successful.append(result)
                print(f"Completed {len(successful)}/{len(items)}")
            except RuntimeError as error:
                failures.append(str(error))
                print(f"FAILED: {error}")
    if failures:
        failure_path = repo_path("outputs", "tables", f"prompt_batch_failures_{args.run_id}.txt")
        ensure_directories([failure_path.parent])
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"Saved {len(failures)} failure(s) to {failure_path}. Re-run only those jobs after resolving the provider error.")
        return 1
    print(f"Saved {len(successful)} real model outputs to data/reference/llm_outputs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
