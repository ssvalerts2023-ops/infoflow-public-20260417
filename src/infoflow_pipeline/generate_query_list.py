from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from openai import OpenAI

from infoflow_pipeline.config import WorkflowConfig, load_default_config


PROMPT_FILE = Path("prompts/01_search_queries_generation_prompt.txt")
SEED_FILE = Path("background_source.txt")
OUTPUT_CSV = Path("master_query_list.csv")
MODEL = "gpt-5"
MAX_ATTEMPTS = 2
TARGET_NAME = ""
ALIASES: list[str] = []
CSV_DELIMITER = ";"

EXPECTED_HEADER = ["keyword", "language_code", "location_code", "depth"]
EXPECTED_ROW_COUNT = 100
EXPECTED_LANGUAGE_CODE = "en"
EXPECTED_LOCATION_CODE = "2826"
EXPECTED_DEPTH = "50"


def configure(config: WorkflowConfig) -> None:
    global PROMPT_FILE, SEED_FILE, OUTPUT_CSV, MODEL, MAX_ATTEMPTS
    global TARGET_NAME, ALIASES

    PROMPT_FILE = config.prompts.query_generation
    SEED_FILE = config.paths.query_seed_file
    OUTPUT_CSV = config.paths.master_query_csv
    MODEL = config.models.query_generation
    MAX_ATTEMPTS = max(1, config.analysis.query_generation_max_attempts)
    TARGET_NAME = config.target_profile.target_name
    ALIASES = list(config.target_profile.aliases)


def load_prompt_template() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def build_prompt(template: str, seed_text: str) -> str:
    return (
        template.replace("{TARGET_NAME}", TARGET_NAME)
        .replace("{ALIASES_JSON}", json.dumps(ALIASES, ensure_ascii=False))
        .replace("{SEED_FILE_PATH}", str(SEED_FILE))
        .replace("{SEED_FILE_TEXT}", seed_text)
    )


def call_model(client: OpenAI, prompt: str) -> str:
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        truncation="disabled",
    )
    return getattr(response, "output_text", "") or ""


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return cleaned


def extract_csv_text(raw_output: str) -> str:
    cleaned = strip_code_fences(raw_output)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    header_line = CSV_DELIMITER.join(EXPECTED_HEADER)

    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == header_line)
    except StopIteration as exc:
        raise ValueError("Model output does not contain the required CSV header.") from exc

    csv_lines = [line for line in lines[start:] if line.strip()]
    return "\n".join(csv_lines).strip() + "\n"


def parse_csv_rows(csv_text: str) -> list[list[str]]:
    reader = csv.reader(io.StringIO(csv_text), delimiter=CSV_DELIMITER)
    rows = [row for row in reader if any((cell or "").strip() for cell in row)]
    if not rows:
        raise ValueError("Model output did not contain any CSV rows.")
    return rows


def normalize_and_validate_rows(rows: list[list[str]]) -> list[list[str]]:
    header = rows[0]
    if header != EXPECTED_HEADER:
        raise ValueError(f"Unexpected CSV header: {header}")

    data_rows = rows[1:]
    if len(data_rows) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} data rows, received {len(data_rows)}.")

    normalized_rows = [EXPECTED_HEADER]
    seen_keywords: set[str] = set()
    for index, row in enumerate(data_rows, start=1):
        if len(row) != 4:
            raise ValueError(f"Row {index} does not have exactly 4 columns: {row}")

        keyword, language_code, location_code, depth = [cell.strip() for cell in row]
        if not keyword:
            raise ValueError(f"Row {index} has an empty keyword.")
        keyword_key = keyword.casefold()
        if keyword_key in seen_keywords:
            raise ValueError(f"Duplicate keyword detected on row {index}: {keyword}")
        seen_keywords.add(keyword_key)

        if language_code != EXPECTED_LANGUAGE_CODE:
            raise ValueError(
                f"Row {index} has invalid language_code {language_code!r}; expected {EXPECTED_LANGUAGE_CODE!r}."
            )
        if location_code != EXPECTED_LOCATION_CODE:
            raise ValueError(
                f"Row {index} has invalid location_code {location_code!r}; expected {EXPECTED_LOCATION_CODE!r}."
            )
        if depth != EXPECTED_DEPTH:
            raise ValueError(f"Row {index} has invalid depth {depth!r}; expected {EXPECTED_DEPTH!r}.")

        normalized_rows.append([keyword, language_code, location_code, depth])

    return normalized_rows


def write_master_query_csv(rows: list[list[str]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=CSV_DELIMITER, lineterminator="\n")
        writer.writerows(rows)


def main(config: WorkflowConfig | None = None) -> None:
    if config is None:
        config = load_default_config()
    configure(config)

    if not SEED_FILE.is_file():
        raise FileNotFoundError(f"Query seed/background file not found: {SEED_FILE}")

    prompt_template = load_prompt_template()
    seed_text = SEED_FILE.read_text(encoding="utf-8")
    prompt = build_prompt(prompt_template, seed_text)

    api_key = config.providers.openai_api_key.resolve_optional()
    base_url = config.providers.openai_base_url
    client = OpenAI(api_key=api_key, base_url=base_url) if api_key else OpenAI(base_url=base_url)

    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw_output = call_model(client, prompt)
        try:
            csv_text = extract_csv_text(raw_output)
            rows = parse_csv_rows(csv_text)
            normalized_rows = normalize_and_validate_rows(rows)
            write_master_query_csv(normalized_rows)
            print(f"Wrote {len(normalized_rows) - 1} query rows to {OUTPUT_CSV}")
            return
        except Exception as exc:
            last_error = str(exc)
            prompt = (
                build_prompt(prompt_template, seed_text)
                + "\n\nVALIDATION REMINDER\n"
                + f"Your previous answer failed validation: {last_error}\n"
                + "Return only a valid CSV with the exact required header and exactly 100 data rows."
            )

    raise ValueError(f"Failed to generate valid master_query_list.csv after {MAX_ATTEMPTS} attempt(s): {last_error}")


if __name__ == "__main__":
    main()
