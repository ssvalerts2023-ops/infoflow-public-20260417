"""
Apply summarized findings to a background narrative in canonical-name blocks.

Workflow:
  1. Load the prompt template, example background, and reviewed findings JSON.
  2. Split findings into blocks of up to 10 items without splitting a canonical_name group.
  3. Run one model pass per block, feeding the current background plus the block.
  4. Parse the amended background and per-item change log from each model response.
  5. Write the final amended background and a compiled CSV change log.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from difflib import HtmlDiff
from pathlib import Path

from openai import OpenAI
from infoflow_pipeline.config import WorkflowConfig, load_default_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "prompts" / "07_updated_background_report_prompt.txt"
DEFAULT_BACKGROUND_FILE = PROJECT_ROOT / "sample_data" / "00_seed_background.txt"
DEFAULT_FINDINGS_FILE = PROJECT_ROOT / "06_relationship_findings.json"
DEFAULT_OUTPUT_BACKGROUND = PROJECT_ROOT / "07_updated_background_report.txt"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "07_background_change_log.csv"
DEFAULT_OUTPUT_DIFF = PROJECT_ROOT / "07_background_diff.html"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "07_background_update_artifacts"

DEFAULT_MODEL = os.getenv("BACKGROUND_UPDATE_MODEL", "gpt-5")
DEFAULT_MAX_ATTEMPTS = max(1, int(os.getenv("BACKGROUND_UPDATE_MAX_ATTEMPTS", "2")))
DEFAULT_BLOCK_SIZE = max(1, int(os.getenv("BACKGROUND_UPDATE_BLOCK_SIZE", "10")))
HTML_DIFF_WRAPCOLUMN = 50

BACKGROUND_START = "===BACKGROUND_NEXT==="
BACKGROUND_END = "===END_BACKGROUND_NEXT==="
CHANGE_LOG_START = "===BLOCK_CHANGE_LOG_JSON==="
CHANGE_LOG_END = "===END_BLOCK_CHANGE_LOG_JSON==="
SECTION_ID_PATTERN = re.compile(r"^(none|[a-z0-9_]+)$")


def configure(config: WorkflowConfig) -> None:
    global BASE_DIR, DEFAULT_PROMPT_FILE, DEFAULT_BACKGROUND_FILE, DEFAULT_FINDINGS_FILE
    global DEFAULT_OUTPUT_BACKGROUND, DEFAULT_OUTPUT_CSV, DEFAULT_OUTPUT_DIFF, DEFAULT_ARTIFACT_DIR
    global DEFAULT_MODEL, DEFAULT_MAX_ATTEMPTS, DEFAULT_BLOCK_SIZE

    BASE_DIR = config.base_dir
    DEFAULT_PROMPT_FILE = config.prompts.background_update
    DEFAULT_BACKGROUND_FILE = config.paths.background_file
    DEFAULT_FINDINGS_FILE = config.paths.findings_json
    DEFAULT_OUTPUT_BACKGROUND = config.paths.background_output_text
    DEFAULT_OUTPUT_CSV = config.paths.background_output_csv
    DEFAULT_OUTPUT_DIFF = config.paths.background_output_diff
    DEFAULT_ARTIFACT_DIR = config.paths.background_artifacts_dir
    DEFAULT_MODEL = config.models.background_update
    DEFAULT_MAX_ATTEMPTS = max(1, config.analysis.background_update_max_attempts)
    DEFAULT_BLOCK_SIZE = max(1, config.analysis.background_block_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run blockwise background amendments.")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--background-file", type=Path, default=DEFAULT_BACKGROUND_FILE)
    parser.add_argument("--findings-file", type=Path, default=DEFAULT_FINDINGS_FILE)
    parser.add_argument("--output-background", type=Path, default=DEFAULT_OUTPUT_BACKGROUND)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-diff", type=Path, default=DEFAULT_OUTPUT_DIFF)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_findings(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return payload


def canonical_name_for_item(item: dict) -> str:
    canonical_name = str(item.get("canonical_name", "") or "").strip()
    if canonical_name:
        return canonical_name
    return str(item.get("entity_name_raw", "") or "").strip()


def file_id_for_item(item: dict) -> str:
    source_narrative = item.get("source_narrative", {})
    if not isinstance(source_narrative, dict):
        return ""
    return str(source_narrative.get("file_id", "") or "").strip()


def group_findings(items: list[dict]) -> list[tuple[str, list[dict]]]:
    groups: list[tuple[str, list[dict]]] = []
    group_index: dict[str, int] = {}

    for item in items:
        canonical_name = canonical_name_for_item(item) or "<missing_canonical_name>"
        index = group_index.get(canonical_name)
        if index is None:
            group_index[canonical_name] = len(groups)
            groups.append((canonical_name, [item]))
        else:
            groups[index][1].append(item)

    return groups


def build_blocks(items: list[dict], block_size: int) -> list[list[dict]]:
    groups = group_findings(items)
    blocks: list[list[dict]] = []
    current_block: list[dict] = []

    for _, group_items in groups:
        if current_block and len(current_block) + len(group_items) > block_size:
            blocks.append(current_block)
            current_block = []

        if not current_block and len(group_items) > block_size:
            blocks.append(list(group_items))
            continue

        current_block.extend(group_items)

    if current_block:
        blocks.append(current_block)

    return blocks


def build_block_manifest(blocks: list[list[dict]]) -> list[dict]:
    manifest = []
    total_blocks = len(blocks)
    for block_no, block_items in enumerate(blocks, start=1):
        canonical_names = []
        seen = set()
        for item in block_items:
            canonical_name = canonical_name_for_item(item)
            if canonical_name not in seen:
                seen.add(canonical_name)
                canonical_names.append(canonical_name)
        manifest.append(
            {
                "block_no": block_no,
                "total_blocks": total_blocks,
                "item_count": len(block_items),
                "canonical_names": canonical_names,
            }
        )
    return manifest


def build_prompt(
    prompt_template: str,
    background_text: str,
    findings_block: list[dict],
    block_no: int,
    total_blocks: int,
    is_final_block: bool,
    reminder: str = "",
) -> str:
    parts = [
        prompt_template.strip(),
        "PASS INPUTS",
        "background_current.txt",
        background_text,
        "findings_block.json",
        json.dumps(findings_block, ensure_ascii=False, indent=2),
        "block_no",
        str(block_no),
        "total_blocks",
        str(total_blocks),
        "is_final_block",
        "true" if is_final_block else "false",
    ]
    if reminder:
        parts.extend(
            [
                "VALIDATION REMINDER",
                reminder,
            ]
        )
    return "\n\n".join(parts)


def call_model(client: OpenAI, prompt: str, model: str) -> str:
    response = client.responses.create(
        model=model,
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


def extract_tagged_block(text: str, start_tag: str, end_tag: str) -> str:
    pattern = re.compile(
        re.escape(start_tag) + r"\s*(.*?)\s*" + re.escape(end_tag),
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Missing tagged block {start_tag} ... {end_tag}.")
    return match.group(1).strip()


def parse_json_array(text: str) -> list[dict]:
    cleaned = strip_code_fences(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])

    if not isinstance(payload, list):
        raise ValueError("Change log block is not a JSON array.")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("Each change log row must be a JSON object.")
    return payload


def expected_item_counter(block_items: list[dict]) -> Counter:
    return Counter(
        (
            str(item.get("entity_name_raw", "") or "").strip(),
            file_id_for_item(item),
        )
        for item in block_items
    )


def normalize_log_row(row: dict, block_no: int) -> dict:
    entity_name_raw = str(row.get("entity_name_raw", "") or "").strip()
    file_id = str(row.get("file_id", "") or "").strip()
    what_has_been_done = str(row.get("what_has_been_done", "") or "").strip()
    section_id = str(row.get("section_id", "") or "").strip()

    raw_block_no = row.get("block_no", block_no)
    try:
        parsed_block_no = int(raw_block_no)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid block_no in change log row: {raw_block_no!r}") from exc

    if not entity_name_raw:
        raise ValueError("Change log row is missing entity_name_raw.")
    if not file_id:
        raise ValueError("Change log row is missing file_id.")
    if not what_has_been_done:
        raise ValueError("Change log row is missing what_has_been_done.")
    if not section_id:
        raise ValueError("Change log row is missing section_id.")
    if parsed_block_no != block_no:
        raise ValueError(f"Change log row has block_no={parsed_block_no}, expected {block_no}.")
    if not SECTION_ID_PATTERN.match(section_id):
        raise ValueError(f"Change log row has invalid section_id={section_id!r}.")

    return {
        "entity_name_raw": entity_name_raw,
        "file_id": file_id,
        "what_has_been_done": what_has_been_done,
        "block_no": parsed_block_no,
        "section_id": section_id,
    }


def parse_and_validate_response(raw_output: str, block_items: list[dict], block_no: int) -> tuple[str, list[dict]]:
    background_text = extract_tagged_block(raw_output, BACKGROUND_START, BACKGROUND_END)
    change_log_text = extract_tagged_block(raw_output, CHANGE_LOG_START, CHANGE_LOG_END)
    rows = [normalize_log_row(row, block_no) for row in parse_json_array(change_log_text)]

    if not background_text:
        raise ValueError("BACKGROUND_NEXT block is empty.")
    if len(rows) != len(block_items):
        raise ValueError(
            f"Change log row count mismatch for block {block_no}: got {len(rows)}, expected {len(block_items)}."
        )

    actual_counter = Counter((row["entity_name_raw"], row["file_id"]) for row in rows)
    expected_counter = expected_item_counter(block_items)
    if actual_counter != expected_counter:
        raise ValueError(
            "Change log rows do not match the input items for this block. "
            f"Expected {expected_counter}, got {actual_counter}."
        )

    return background_text, rows


def write_change_log_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["entity_name_raw", "file_id", "what_has_been_done", "block_no", "section_id"],
            delimiter=";",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_html_diff(path: Path, original_background: str, amended_background: str) -> None:
    diff_html = HtmlDiff(tabsize=2, wrapcolumn=HTML_DIFF_WRAPCOLUMN).make_file(
        original_background.splitlines(),
        amended_background.splitlines(),
        fromdesc="background_before.txt",
        todesc="background_after.txt",
        context=True,
        numlines=3,
        charset="utf-8",
    )
    diff_html = re.sub(
        r"\n\s*table\.diff \{width:100%; table-layout:fixed\}"
        r"\n\s*table\.diff td \{white-space:normal !important; overflow-wrap:anywhere; word-break:break-word\}",
        "",
        diff_html,
        count=1,
    )
    write_text(path, diff_html)


def run_block_pass(
    client: OpenAI,
    prompt_template: str,
    background_text: str,
    block_items: list[dict],
    block_no: int,
    total_blocks: int,
    model: str,
    max_attempts: int,
) -> tuple[str, list[dict], int]:
    reminder = ""
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        prompt = build_prompt(
            prompt_template=prompt_template,
            background_text=background_text,
            findings_block=block_items,
            block_no=block_no,
            total_blocks=total_blocks,
            is_final_block=block_no == total_blocks,
            reminder=reminder,
        )
        raw_output = call_model(client, prompt, model=model)

        try:
            next_background, rows = parse_and_validate_response(raw_output, block_items, block_no)
            return next_background, rows, attempt
        except Exception as exc:
            last_error = str(exc)
            reminder = (
                "Your previous answer failed validation. "
                f"Problem: {last_error}. "
                "Return exactly the required tagged blocks, include one JSON row per input item, "
                "and do not output any extra prose."
            )

    raise ValueError(f"Block {block_no} failed after {max_attempts} attempt(s): {last_error}")


def ensure_required_files(args: argparse.Namespace) -> None:
    for path in [args.prompt_file, args.background_file, args.findings_file]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")


def main(config: WorkflowConfig | None = None) -> None:
    if config is None:
        config = load_default_config()
    configure(config)

    args = argparse.Namespace(
        prompt_file=DEFAULT_PROMPT_FILE,
        background_file=DEFAULT_BACKGROUND_FILE,
        findings_file=DEFAULT_FINDINGS_FILE,
        output_background=DEFAULT_OUTPUT_BACKGROUND,
        output_csv=DEFAULT_OUTPUT_CSV,
        output_diff=DEFAULT_OUTPUT_DIFF,
        artifact_dir=DEFAULT_ARTIFACT_DIR,
        model=DEFAULT_MODEL,
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        block_size=DEFAULT_BLOCK_SIZE,
    )
    ensure_required_files(args)

    prompt_template = read_text(args.prompt_file)
    original_background = read_text(args.background_file)
    findings = load_findings(args.findings_file)
    blocks = build_blocks(findings, block_size=max(1, args.block_size))

    artifact_dir = args.artifact_dir
    blocks_dir = artifact_dir / "blocks"
    passes_dir = artifact_dir / "passes"
    report_path = artifact_dir / "background_update_run_report.json"
    manifest_path = artifact_dir / "update_block_manifest.json"

    for directory in [artifact_dir, blocks_dir, passes_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    manifest = build_block_manifest(blocks)
    write_json(manifest_path, manifest)

    width = max(2, len(str(max(len(blocks), 1))))
    for index, block_items in enumerate(blocks, start=1):
        write_json(blocks_dir / f"update_block_{index:0{width}d}.json", block_items)

    if not blocks:
        write_text(args.output_background, original_background)
        write_change_log_csv(args.output_csv, [])
        write_html_diff(args.output_diff, original_background, original_background)
        write_json(
            report_path,
            {
                "status": "ok",
                "message": "No findings were available. Wrote the original background unchanged.",
                "prompt_file": str(args.prompt_file),
                "background_file": str(args.background_file),
                "findings_file": str(args.findings_file),
                "output_background": str(args.output_background),
                "output_csv": str(args.output_csv),
                "output_diff": str(args.output_diff),
            },
        )
        print(f"Wrote unchanged background to {args.output_background}")
        print(f"Wrote empty change log CSV to {args.output_csv}")
        print(f"Wrote HTML diff to {args.output_diff}")
        return

    api_key = config.providers.openai_api_key.resolve_optional()
    base_url = config.providers.openai_base_url
    client = OpenAI(api_key=api_key, base_url=base_url) if api_key else OpenAI(base_url=base_url)
    current_background = original_background
    all_log_rows: list[dict] = []
    report_rows = []

    total_blocks = len(blocks)
    for block_no, block_items in enumerate(blocks, start=1):
        print(f"[{block_no}/{total_blocks}] Applying block with {len(block_items)} item(s)")

        next_background, block_rows, attempts_used = run_block_pass(
            client=client,
            prompt_template=prompt_template,
            background_text=current_background,
            block_items=block_items,
            block_no=block_no,
            total_blocks=total_blocks,
            model=args.model,
            max_attempts=max(1, args.max_attempts),
        )

        current_background = next_background
        all_log_rows.extend(block_rows)
        write_text(passes_dir / f"background_after_pass_{block_no:0{width}d}.txt", current_background)

        report_rows.append(
            {
                "block_no": block_no,
                "item_count": len(block_items),
                "canonical_names": build_block_manifest([block_items])[0]["canonical_names"],
                "attempts_used": attempts_used,
            }
        )

    write_text(args.output_background, current_background)
    write_change_log_csv(args.output_csv, all_log_rows)
    write_html_diff(args.output_diff, original_background, current_background)
    write_json(
        report_path,
        {
            "status": "ok",
            "prompt_file": str(args.prompt_file),
            "background_file": str(args.background_file),
            "findings_file": str(args.findings_file),
            "output_background": str(args.output_background),
            "output_csv": str(args.output_csv),
            "output_diff": str(args.output_diff),
            "artifact_dir": str(args.artifact_dir),
            "model": args.model,
            "block_size": max(1, args.block_size),
            "block_count": total_blocks,
            "blocks": report_rows,
        },
    )

    print(f"\nWrote final amended background to {args.output_background}")
    print(f"Wrote compiled change log CSV to {args.output_csv}")
    print(f"Wrote HTML diff to {args.output_diff}")
    print(f"Run report written to {report_path}")


if __name__ == "__main__":
    main()
