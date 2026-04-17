# Batch-analyse TXT files with an OpenAI prompt template and save prettified JSON results.

from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
from openai import OpenAI
from infoflow_pipeline.config import WorkflowConfig, load_default_config

PROMPT_FILE = Path("prompts/04_file_level_extractions_prompt.txt")
SOURCE_DIR = Path("serp_downloads")
CSV_PATH = Path("master_url_list.csv")
OUT_PATH = Path("file_review.json")
CHECKPOINT_PATH = OUT_PATH.with_suffix(".checkpoint.json")
CSV_DELIMITER = ";"

MODEL = "gpt-5-mini"
TARGET_NAME = ""
ALIASES: list[str] = []
MAX_FULL_FILE_BYTES = 70 * 1024
CHUNK_RADIUS = 5_000
TARGET_PATTERN = re.compile("|".join(re.escape(value) for value in [TARGET_NAME, *ALIASES]), re.IGNORECASE)
LOW_COST_REVIEW_PROMPT = """Decide whether this file may refer to the target company ({TARGET_NAME} and any of the aliases / variants, including {ALIASES} and any other similar names), including misspellings, encoding corruption, aliases, subsidiaries, or indirect references.

If yes, or unsure, return exactly: yes
If no, or very likely no, return exactly: no

Do not extract entities or transactions.
Return only yes or no.

File ID: {FILE_ID}
Source URL: {SOURCE_URL}

FILE TEXT:
{FILE_TEXT}
"""


def _compile_target_pattern(target_name: str, aliases: list[str]) -> re.Pattern[str]:
    values = [target_name, *aliases]
    return re.compile("|".join(re.escape(value) for value in values), re.IGNORECASE)


def configure(config: WorkflowConfig) -> None:
    global PROMPT_FILE, SOURCE_DIR, CSV_PATH, OUT_PATH, CHECKPOINT_PATH
    global MODEL, TARGET_NAME, ALIASES, MAX_FULL_FILE_BYTES, CHUNK_RADIUS, TARGET_PATTERN

    PROMPT_FILE = config.prompts.file_analysis
    SOURCE_DIR = config.paths.downloads_dir
    CSV_PATH = config.paths.master_url_csv
    OUT_PATH = config.paths.file_review_json
    CHECKPOINT_PATH = OUT_PATH.with_suffix(".checkpoint.json")
    MODEL = config.models.file_analysis
    TARGET_NAME = config.target_profile.target_name
    ALIASES = list(config.target_profile.aliases)
    MAX_FULL_FILE_BYTES = config.analysis.max_full_file_bytes
    CHUNK_RADIUS = config.analysis.chunk_radius
    TARGET_PATTERN = _compile_target_pattern(TARGET_NAME, ALIASES)


def load_prompt_template() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def build_url_map(csv_path: Path) -> dict[str, str]:
    df = pd.read_csv(csv_path, sep=CSV_DELIMITER)
    if "download_file" not in df.columns or "url" not in df.columns:
        raise ValueError("CSV must contain columns: 'download_file' and 'url'")

    subset = df.loc[:, ["download_file", "url"]].dropna(subset=["download_file", "url"])
    stems = subset["download_file"].astype(str).map(lambda value: Path(value).stem)
    urls = subset["url"].astype(str)
    return dict(zip(stems, urls))


def fill_prompt(template: str, file_id: str, source_url: str, file_text: str) -> str:
    return (
        template.replace("{TARGET_NAME}", TARGET_NAME)
        .replace("{ALIASES}", json.dumps(ALIASES, ensure_ascii=False))
        .replace("{FILE_ID}", file_id)
        .replace("{SOURCE_URL}", source_url or "")
        .replace("{FILE_TEXT}", file_text)
    )


def extract_json(text: str):
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}\s*$", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)), None
            except json.JSONDecodeError as e:
                return None, f"JSON decode failed after fallback: {e}"
        return None, "No valid JSON object found in model output"


def build_fallback_result(file_id: str, source_url: str, reason: str, note: str) -> dict:
    return {
        "file_id": file_id,
        "source_url": source_url,
        "target_company": TARGET_NAME,
        "date_last_updated": "unknown",
        "relevant_country": "unknown",
        "presort": {
            "category": "PARSING_ERROR",
            "reason": reason,
        },
        "entities": [],
        "file_summary": "",
        "parser_notes": [note],
    }


def build_not_in_text_result(file_id: str, source_url: str, reason: str) -> dict:
    return {
        "file_id": file_id,
        "source_url": source_url,
        "target_company": TARGET_NAME,
        "date_last_updated": "unknown",
        "relevant_country": "unknown",
        "presort": {
            "category": "NOT_IN_TEXT",
            "reason": reason,
        },
        "entities": [],
        "file_summary": "",
        "parser_notes": [],
    }


def collect_target_chunks(file_text: str) -> list[str]:
    intervals = []
    for match in TARGET_PATTERN.finditer(file_text):
        start = max(0, match.start() - CHUNK_RADIUS)
        end = min(len(file_text), match.end() + CHUNK_RADIUS)
        intervals.append((start, end))

    if not intervals:
        return []

    intervals.sort()
    merged_intervals = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_intervals.append((current_start, current_end))
            current_start, current_end = start, end
    merged_intervals.append((current_start, current_end))

    return [file_text[start:end] for start, end in merged_intervals]


def merge_presort(items: list[dict]) -> dict:
    priority = {
        "RELEVANT": 5,
        "DATABASE": 4,
        "LIST": 3,
        "IRRELEVANT": 2,
        "NOT_RELEVANT": 2,
        "NOT_IN_TEXT": 1,
        "PARSING_ERROR": 0,
    }
    best_category = "PARSING_ERROR"
    best_score = priority[best_category]
    reasons = []

    for item in items:
        presort = item.get("presort", {})
        category = str(presort.get("category", "PARSING_ERROR")).upper()
        score = priority.get(category, -1)
        if score > best_score:
            best_category = category
            best_score = score
        reason = presort.get("reason", "")
        if reason and reason not in reasons:
            reasons.append(reason)

    return {
        "category": best_category,
        "reason": " | ".join(reasons),
    }


def merge_entities(items: list[dict]) -> list[dict]:
    merged = {}
    for item in items:
        for entity in item.get("entities", []):
            key = (
                entity.get("entity_name_raw", ""),
                entity.get("entity_type_guess", ""),
                entity.get("claim_substance", ""),
            )
            if key not in merged:
                merged[key] = {
                    "entity_name_raw": entity.get("entity_name_raw", ""),
                    "entity_type_guess": entity.get("entity_type_guess", "unknown"),
                    "claim_substance": entity.get("claim_substance", ""),
                    "evidence_snippets": [],
                    "confidence": entity.get("confidence", ""),
                }

            current = merged[key]
            for snippet in entity.get("evidence_snippets", []):
                if snippet and snippet not in current["evidence_snippets"]:
                    current["evidence_snippets"].append(snippet)

            try:
                current_conf = int(current.get("confidence", 0) or 0)
                next_conf = int(entity.get("confidence", 0) or 0)
                if next_conf > current_conf:
                    current["confidence"] = entity.get("confidence", "")
            except (TypeError, ValueError):
                if not current.get("confidence"):
                    current["confidence"] = entity.get("confidence", "")

    return list(merged.values())


def merge_chunk_results(file_id: str, source_url: str, items: list[dict], chunk_errors: list[str]) -> dict:
    target_company = TARGET_NAME
    date_last_updated = "unknown"
    relevant_country = "unknown"
    summaries = []
    parser_notes = []

    for item in items:
        if item.get("target_company"):
            target_company = item["target_company"]
        if date_last_updated == "unknown" and item.get("date_last_updated"):
            date_last_updated = item["date_last_updated"]
        if relevant_country == "unknown" and item.get("relevant_country"):
            relevant_country = item["relevant_country"]

        summary = item.get("file_summary", "")
        if summary and summary not in summaries:
            summaries.append(summary)

        for note in item.get("parser_notes", []):
            if note and note not in parser_notes:
                parser_notes.append(note)

    for note in chunk_errors:
        if note and note not in parser_notes:
            parser_notes.append(note)

    return {
        "file_id": file_id,
        "source_url": source_url,
        "target_company": target_company,
        "date_last_updated": date_last_updated,
        "relevant_country": relevant_country,
        "presort": merge_presort(items),
        "entities": merge_entities(items),
        "file_summary": " | ".join(summaries),
        "parser_notes": parser_notes,
    }


def analyze_prompt(client: OpenAI, prompt: str):
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        truncation="disabled",
    )
    raw_text = getattr(response, "output_text", "") or ""
    return extract_json(raw_text)


def low_cost_name_review(client: OpenAI, file_id: str, source_url: str, file_text: str) -> bool:
    prompt = LOW_COST_REVIEW_PROMPT.format(
        TARGET_NAME=TARGET_NAME,
        ALIASES=json.dumps(ALIASES, ensure_ascii=False),
        FILE_ID=file_id,
        SOURCE_URL=source_url or "",
        FILE_TEXT=file_text,
    )
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        truncation="disabled",
    )
    raw_text = (getattr(response, "output_text", "") or "").strip().lower()
    return raw_text != "no"


def analyze_file(
    client: OpenAI,
    prompt_template: str,
    file_id: str,
    source_url: str,
    file_text: str,
    file_size: int,
) -> dict:
    try:
        has_target_match = TARGET_PATTERN.search(file_text) is not None
        if not has_target_match:
            if file_size > MAX_FULL_FILE_BYTES:
                return build_not_in_text_result(file_id, source_url, "BIG")

            low_cost_hit = low_cost_name_review(client, file_id, source_url, file_text)
            if not low_cost_hit:
                return build_not_in_text_result(file_id, source_url, "SMALL")

        if file_size <= MAX_FULL_FILE_BYTES:
            prompt = fill_prompt(prompt_template, file_id=file_id, source_url=source_url, file_text=file_text)
            parsed, parse_err = analyze_prompt(client, prompt)
            if parsed is None:
                return build_fallback_result(
                    file_id,
                    source_url,
                    "Model output could not be parsed as valid JSON.",
                    f"parse_json_error: {parse_err}",
                )
            return parsed

        chunks = collect_target_chunks(file_text)
        if not chunks:
            return build_fallback_result(
                file_id,
                source_url,
                "No target-name matches were found in the oversized file.",
                "chunking_error: oversized file exceeded 70 KB and no TARGET_NAME or ALIASES matches were found.",
            )

        chunk_results = []
        chunk_errors = []
        for index, chunk in enumerate(chunks, start=1):
            prompt = fill_prompt(prompt_template, file_id=file_id, source_url=source_url, file_text=chunk)
            parsed, parse_err = analyze_prompt(client, prompt)
            if parsed is None:
                chunk_errors.append(f"chunk_{index}_parse_json_error: {parse_err}")
                continue
            chunk_results.append(parsed)

        if not chunk_results:
            return build_fallback_result(
                file_id,
                source_url,
                "All chunk responses failed to parse as valid JSON.",
                " | ".join(chunk_errors),
            )

        return merge_chunk_results(file_id, source_url, chunk_results, chunk_errors)
    except Exception as e:
        return build_fallback_result(
            file_id,
            source_url,
            "API or file processing error prevented analysis.",
            f"api_or_io_error: {str(e)}",
        )


def write_results(results: list[dict]) -> None:
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    CHECKPOINT_PATH.write_text(payload, encoding="utf-8")
    OUT_PATH.write_text(payload, encoding="utf-8")


def main(config: WorkflowConfig | None = None):
    if config is None:
        config = load_default_config()
    configure(config)

    api_key = config.providers.openai_api_key.resolve_optional()
    base_url = config.providers.openai_base_url
    client = OpenAI(api_key=api_key, base_url=base_url) if api_key else OpenAI(base_url=base_url)
    prompt_template = load_prompt_template()
    url_map = build_url_map(CSV_PATH)

    txt_files = sorted([p for p in SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    results = []
    total = len(txt_files)

    for i, txt_path in enumerate(txt_files, start=1):
        file_id = txt_path.stem
        source_url = url_map.get(file_id, "")
        print(f"[{i}/{total}] Analysing {txt_path.name}")

        try:
            file_size = txt_path.stat().st_size
            file_text = txt_path.read_text(encoding="utf-8", errors="replace")
            has_target_match = TARGET_PATTERN.search(file_text) is not None
            if has_target_match:
                print("  Direct target-name match found in text; proceeding to full analysis.")
            elif file_size > MAX_FULL_FILE_BYTES:
                print("  No direct target-name match found and file exceeds 70 KB; returning NOT_IN_TEXT (BIG).")
            else:
                print("  No direct target-name match found; running low-cost review before full analysis.")
            result = analyze_file(client, prompt_template, file_id, source_url, file_text, file_size)
            if (
                not has_target_match
                and result.get("presort", {}).get("category") != "NOT_IN_TEXT"
            ):
                print("  Low-cost review returned yes/unsure; proceeding with full analysis.")
            results.append(result)
        except Exception as e:
            results.append(
                build_fallback_result(
                    file_id,
                    source_url,
                    "API or file I/O error prevented analysis.",
                    f"api_or_io_error: {str(e)}",
                )
            )

        write_results(results)

    print(f"\nDone. Output written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
