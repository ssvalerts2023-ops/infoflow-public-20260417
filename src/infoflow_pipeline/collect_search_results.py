from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

from clientdcf.client import RestClient
from infoflow_pipeline.config import WorkflowConfig, load_default_config

API_BASE_URL = "https://api.dataforseo.com"
API_PATH = "/v3/serp/google/organic/live/regular"
API_RETRIES = 3
EXPECTED_LANGUAGE_CODE = "en"
EXPECTED_LOCATION_CODE = 2826
DEFAULT_DEPTH = 50
CSV_DELIMITER = ";"

# Input: one row = one API run (keyword(s), language, location, optionally depth)
INPUT_CSV = Path("master_query_list.csv")

# Output: deduped URL ledger across all runs
OUTPUT_URL_CSV = Path("master_url_list.csv")

# Output: deduped query-to-URL provenance ledger across all runs
QUERY_URL_MAP_CSV = Path("query_to_url_map.csv")


def configure(config: WorkflowConfig) -> None:
    global API_BASE_URL, API_PATH, API_RETRIES, DEFAULT_DEPTH
    global INPUT_CSV, OUTPUT_URL_CSV, QUERY_URL_MAP_CSV

    API_BASE_URL = config.search.api_base_url
    API_PATH = config.search.api_path
    API_RETRIES = config.search.api_retries
    DEFAULT_DEPTH = config.search.default_depth
    INPUT_CSV = config.paths.master_query_csv
    OUTPUT_URL_CSV = config.paths.master_url_csv
    QUERY_URL_MAP_CSV = config.paths.query_url_map_csv


def extract_urls(response: dict) -> list[str]:
    items = response["tasks"][0]["result"][0]["items"]
    return [it["url"] for it in items if isinstance(it, dict) and it.get("url")]


@dataclass(frozen=True)
class QueryParams:
    language_code: str
    location_code: int
    depth: int
    keyword: str

    def to_compact_json(self) -> str:
        return json.dumps(
            {
                "language_code": self.language_code,
                "location_code": self.location_code,
                "depth": self.depth,
                "keyword": self.keyword,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def ensure_url_csv_header(csv_path: Path) -> None:
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=CSV_DELIMITER, lineterminator="\n")
        writer.writerow(["query_parameters", "url", "generated_at"])


def ensure_query_url_map_header(csv_path: Path) -> None:
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=CSV_DELIMITER, lineterminator="\n")
        writer.writerow(["query_parameters", "url", "seen_at"])


def load_existing_urls(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    existing: set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=CSV_DELIMITER)
        if not reader.fieldnames or "url" not in reader.fieldnames:
            raise ValueError(f"URL CSV exists but has no 'url' column: {csv_path}")
        for row in reader:
            url = (row.get("url") or "").strip()
            if url:
                existing.add(url)
    return existing


def load_existing_query_url_pairs(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()

    existing: set[tuple[str, str]] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=CSV_DELIMITER)
        required = {"query_parameters", "url"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Query-to-URL CSV exists but is missing required columns {sorted(required)}: {csv_path}"
            )
        for row in reader:
            qp = (row.get("query_parameters") or "").strip()
            url = (row.get("url") or "").strip()
            if qp and url:
                existing.add((qp, url))
    return existing


def append_url_rows(csv_path: Path, rows: Iterable[tuple[str, str, str]]) -> int:
    ensure_url_csv_header(csv_path)
    count = 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=CSV_DELIMITER, lineterminator="\n")
        for qp, url, ts in rows:
            writer.writerow([qp, url, ts])
            count += 1
    return count


def append_query_url_map_rows(csv_path: Path, rows: Iterable[tuple[str, str, str]]) -> int:
    ensure_query_url_map_header(csv_path)
    count = 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=CSV_DELIMITER, lineterminator="\n")
        for qp, url, ts in rows:
            writer.writerow([qp, url, ts])
            count += 1
    return count


def _norm(s: Any) -> str:
    return str(s).strip() if s is not None else ""


def _effective_status(row: dict[str, str]) -> str:
    status = _norm(row.get("status"))
    if status:
        return status

    # Backward compatibility for rows processed before explicit statuses existed.
    if _norm(row.get("processed_at")):
        return "completed"

    return "pending"


def read_tasks(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}\n"
            "Create it with headers like: keyword;language_code;location_code[;depth]\n"
            "Optional columns: processed_at,result_count,status,attempt_count,last_attempt_at,last_error,completed_at"
        )

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=CSV_DELIMITER)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header row: {csv_path}")
        rows = [dict((k, _norm(v)) for k, v in r.items()) for r in reader]
        return rows, list(reader.fieldnames)


def write_tasks(csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=csv_path.parent,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        writer = csv.DictWriter(
            tmp,
            fieldnames=fieldnames,
            extrasaction="ignore",
            delimiter=CSV_DELIMITER,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, csv_path)


def _set_row_status(
    row: dict[str, str],
    *,
    status: str,
    last_attempt_at: str | None = None,
    completed_at: str | None = None,
    result_count: int | None = None,
    last_error: str | None = None,
    attempt_count_increment: int = 0,
) -> None:
    current_attempts = _norm(row.get("attempt_count"))
    try:
        attempt_total = int(current_attempts) if current_attempts else 0
    except ValueError:
        attempt_total = 0

    row["status"] = status
    row["attempt_count"] = str(attempt_total + attempt_count_increment)
    row["last_attempt_at"] = last_attempt_at or _norm(row.get("last_attempt_at"))
    row["last_error"] = last_error or ""
    row["completed_at"] = completed_at or ""

    if status == "completed":
        row["processed_at"] = completed_at or row.get("processed_at", "")
        row["result_count"] = str(result_count if result_count is not None else 0)
    elif result_count is not None:
        row["result_count"] = str(result_count)


def _post_with_retries(
    client: RestClient,
    post_data: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any] | None, int, str | None]:
    attempts = 0
    last_error: str | None = None

    for _ in range(API_RETRIES + 1):
        attempts += 1
        try:
            response = client.post(API_PATH, post_data)
            status_code = response.get("status_code")
            status_message = response.get("status_message")
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue

        if status_code != 20000:
            last_error = f"API status {status_code} {status_message}".strip()
            continue

        return response, attempts, None

    return None, attempts, last_error


def main(config: WorkflowConfig | None = None) -> None:
    if config is None:
        config = load_default_config()
    configure(config)

    login = config.providers.dataforseo_login.resolve("DataForSEO login")
    password = config.providers.dataforseo_password.resolve("DataForSEO password")
    client = RestClient(login, password, api_base_url=API_BASE_URL)

    task_rows, task_fieldnames = read_tasks(INPUT_CSV)

    # Ensure the output columns exist in the task CSV
    if "processed_at" not in task_fieldnames:
        task_fieldnames.append("processed_at")
    if "result_count" not in task_fieldnames:
        task_fieldnames.append("result_count")
    if "status" not in task_fieldnames:
        task_fieldnames.append("status")
    if "attempt_count" not in task_fieldnames:
        task_fieldnames.append("attempt_count")
    if "last_attempt_at" not in task_fieldnames:
        task_fieldnames.append("last_attempt_at")
    if "last_error" not in task_fieldnames:
        task_fieldnames.append("last_error")
    if "completed_at" not in task_fieldnames:
        task_fieldnames.append("completed_at")

    existing_urls = load_existing_urls(OUTPUT_URL_CSV)
    existing_query_url_pairs = load_existing_query_url_pairs(QUERY_URL_MAP_CSV)
    seen_urls_this_run: set[str] = set()
    seen_query_url_pairs_this_run: set[tuple[str, str]] = set()

    processed_tasks = 0
    skipped_tasks = 0
    failed_tasks = 0
    added_urls = 0
    added_query_url_mappings = 0

    for row in task_rows:
        status = _effective_status(row)
        if status in {"completed", "invalid_input"}:
            skipped_tasks += 1
            continue

        # Support either "keyword" or "keywords" column name
        keyword = _norm(row.get("keyword")) or _norm(row.get("keywords"))
        language_code = _norm(row.get("language_code")) or EXPECTED_LANGUAGE_CODE
        location_code_raw = _norm(row.get("location_code")) or str(EXPECTED_LOCATION_CODE)
        depth_raw = _norm(row.get("depth"))
        row_now_iso = datetime.now(timezone.utc).isoformat()

        if not keyword:
            failed_tasks += 1
            _set_row_status(
                row,
                status="invalid_input",
                last_attempt_at=row_now_iso,
                result_count=0,
                last_error="Missing required field: keyword",
                attempt_count_increment=1,
            )
            write_tasks(INPUT_CSV, task_rows, task_fieldnames)
            print("Skipping row due to missing required field: keyword")
            continue

        if language_code != EXPECTED_LANGUAGE_CODE:
            failed_tasks += 1
            _set_row_status(
                row,
                status="invalid_input",
                last_attempt_at=row_now_iso,
                result_count=0,
                last_error=f"Invalid language_code: {language_code}",
                attempt_count_increment=1,
            )
            write_tasks(INPUT_CSV, task_rows, task_fieldnames)
            print(f"Skipping row due to invalid language_code: {language_code}")
            continue

        try:
            location_code = int(location_code_raw)
        except ValueError:
            failed_tasks += 1
            _set_row_status(
                row,
                status="invalid_input",
                last_attempt_at=row_now_iso,
                result_count=0,
                last_error=f"Invalid location_code: {location_code_raw}",
                attempt_count_increment=1,
            )
            write_tasks(INPUT_CSV, task_rows, task_fieldnames)
            print(f"Skipping row due to invalid location_code: {location_code_raw}")
            continue

        if location_code != EXPECTED_LOCATION_CODE:
            failed_tasks += 1
            _set_row_status(
                row,
                status="invalid_input",
                last_attempt_at=row_now_iso,
                result_count=0,
                last_error=f"Invalid location_code: {location_code}",
                attempt_count_increment=1,
            )
            write_tasks(INPUT_CSV, task_rows, task_fieldnames)
            print(f"Skipping row due to invalid location_code: {location_code}")
            continue

        depth = DEFAULT_DEPTH
        if depth_raw:
            try:
                depth = int(depth_raw)
            except ValueError:
                depth = DEFAULT_DEPTH

        params = QueryParams(
            language_code=language_code,
            location_code=location_code,
            depth=depth,
            keyword=keyword,
        )

        post_data = {
            0: {
                "language_code": params.language_code,
                "location_code": params.location_code,
                "depth": params.depth,
                "keyword": params.keyword,
            }
        }

        response, api_attempts, post_error = _post_with_retries(client, post_data)
        now_iso = datetime.now(timezone.utc).isoformat()

        if response is None:
            failed_tasks += 1
            _set_row_status(
                row,
                status="retryable_error",
                last_attempt_at=now_iso,
                result_count=0,
                last_error=post_error or "Unknown API error",
                attempt_count_increment=api_attempts,
            )
            write_tasks(INPUT_CSV, task_rows, task_fieldnames)
            print(f"loc={location_code} kw={keyword} -> error after {api_attempts} attempts: {post_error}")
            continue

        try:
            urls = extract_urls(response)
        except (KeyError, TypeError):
            failed_tasks += 1
            _set_row_status(
                row,
                status="retryable_error",
                last_attempt_at=now_iso,
                result_count=0,
                last_error="Unexpected response shape",
                attempt_count_increment=api_attempts,
            )
            write_tasks(INPUT_CSV, task_rows, task_fieldnames)
            print(f"loc={location_code} kw={keyword} -> unexpected response shape after {api_attempts} attempts")
            continue

        print(f"loc={location_code} kw={keyword} -> {len(urls)} urls")

        qp_str = params.to_compact_json()
        new_query_url_rows: list[tuple[str, str, str]] = []
        new_url_rows: list[tuple[str, str, str]] = []
        for url in urls:
            pair = (qp_str, url)
            if pair not in existing_query_url_pairs and pair not in seen_query_url_pairs_this_run:
                seen_query_url_pairs_this_run.add(pair)
                new_query_url_rows.append((qp_str, url, now_iso))

            if url not in existing_urls and url not in seen_urls_this_run:
                seen_urls_this_run.add(url)
                new_url_rows.append((qp_str, url, now_iso))

        added_query_url_mappings += append_query_url_map_rows(QUERY_URL_MAP_CSV, new_query_url_rows)
        existing_query_url_pairs.update((qp, url) for qp, url, _ in new_query_url_rows)

        added_urls += append_url_rows(OUTPUT_URL_CSV, new_url_rows)
        existing_urls.update(url for _, url, _ in new_url_rows)

        _set_row_status(
            row,
            status="completed",
            last_attempt_at=now_iso,
            completed_at=now_iso,
            result_count=len(urls),
            last_error=None,
            attempt_count_increment=api_attempts,
        )
        write_tasks(INPUT_CSV, task_rows, task_fieldnames)
        processed_tasks += 1

    total_urls_after = len(existing_urls)

    print(f"Processed tasks: {processed_tasks}")
    print(f"Skipped tasks (already processed): {skipped_tasks}")
    print(f"Failed/invalid tasks: {failed_tasks}")
    print(f"Added new URLs: {added_urls}")
    print(f"Added query-to-URL mappings: {added_query_url_mappings}")
    print(f"Total unique URLs in URL CSV: {total_urls_after}")
    print(f"Task CSV path: {INPUT_CSV}")
    print(f"URL CSV path: {OUTPUT_URL_CSV}")
    print(f"Query-to-URL CSV path: {QUERY_URL_MAP_CSV}")


if __name__ == "__main__":
    main()
