"""
Flatten file-level JSON findings into CSV outputs for downstream review steps.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from infoflow_pipeline.config import WorkflowConfig, load_default_config


INPUT_JSON = Path("file_review.json")
SERP_DOWNLOADS_DIRS = [Path("serp_downloads")]
TARGET_TERMS: list[tuple[str, str]] = []

OUTPUT_RELEVANT = Path("relevant.csv")
OUTPUT_OTHER = Path("other.csv")

CSV_DELIMITER = ";"
CSV_WRITER_KWARGS = {
    "delimiter": CSV_DELIMITER,
    "lineterminator": "\n",
}


def configure(config: WorkflowConfig) -> None:
    global INPUT_JSON, SERP_DOWNLOADS_DIRS, OUTPUT_RELEVANT, OUTPUT_OTHER, TARGET_TERMS

    INPUT_JSON = config.paths.file_review_json
    SERP_DOWNLOADS_DIRS = list(config.paths.download_lookup_dirs)
    OUTPUT_RELEVANT = config.paths.relevant_csv
    OUTPUT_OTHER = config.paths.other_csv

    target_terms = [
        config.target_profile.target_name,
        *config.target_profile.aliases,
    ]
    TARGET_TERMS = []
    for term in target_terms:
        normalized_term = str(term or "").strip()
        casefolded = normalized_term.casefold()
        if casefolded:
            TARGET_TERMS.append((normalized_term, casefolded))


def sanitize(value) -> str:
    if value is None:
        return ""
    return str(value).replace(";", ",")


def join_list(values) -> str:
    if not values:
        return ""
    return sanitize(" ||| ".join(str(value) for value in values))


def normalize_category(category: str) -> str:
    if category is None:
        return ""
    return str(category).strip().upper().replace(" ", "_")


def target_term_match(file_id: str) -> dict[str, str]:
    for source_dir in SERP_DOWNLOADS_DIRS:
        txt_path = source_dir / f"{file_id}.txt"
        if not txt_path.is_file():
            continue
        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace").casefold()
            for display_term, casefolded_term in TARGET_TERMS:
                if casefolded_term in text:
                    return {
                        "matched": "true",
                        "postprocess_flag": "true",
                        "postprocess_rule": "target_term_match",
                        "postprocess_detail": (
                            "Promoted from NOT_IN_TEXT because a configured target term "
                            f"was found in source text ({display_term}) at {txt_path}."
                        ),
                        "matched_term": display_term,
                        "source_text_path": str(txt_path),
                    }
        except Exception as exc:
            print(f"Warning: Could not read {txt_path}: {exc}", file=sys.stderr)
            return {
                "matched": "false",
                "postprocess_flag": "",
                "postprocess_rule": "",
                "postprocess_detail": "",
                "matched_term": "",
                "source_text_path": str(txt_path),
            }
    return {
        "matched": "false",
        "postprocess_flag": "",
        "postprocess_rule": "",
        "postprocess_detail": "",
        "matched_term": "",
        "source_text_path": "",
    }


def main(config: WorkflowConfig | None = None) -> None:
    if config is None:
        config = load_default_config()
    configure(config)

    with INPUT_JSON.open(encoding="utf-8") as handle:
        data = json.load(handle)

    print(f"Loaded {len(data)} items from {INPUT_JSON}")

    relevant_fields = [
        "confidence",
        "file_id",
        "date_last_updated",
        "relevant_country",
        "target_company",
        "entity_name_raw",
        "entity_type_guess",
        "claim_substance",
        "evidence_snippets",
        "postprocess_flag",
        "postprocess_rule",
        "postprocess_detail",
    ]

    other_fields = [
        "file_id",
        "presort_category",
        "presort_reason",
        "file_summary",
        "postprocess_flag",
        "postprocess_rule",
        "postprocess_detail",
    ]

    other_categories = {"LIST", "IRRELEVANT", "NOT_RELEVANT", "PARSING_ERROR"}
    relevant_count = 0
    other_count = 0
    rescued_count = 0

    OUTPUT_RELEVANT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_OTHER.parent.mkdir(parents=True, exist_ok=True)

    with (
        OUTPUT_RELEVANT.open("w", newline="", encoding="utf-8") as relevant_handle,
        OUTPUT_OTHER.open("w", newline="", encoding="utf-8") as other_handle,
    ):
        relevant_writer = csv.DictWriter(
            relevant_handle,
            fieldnames=relevant_fields,
            **CSV_WRITER_KWARGS,
        )
        other_writer = csv.DictWriter(
            other_handle,
            fieldnames=other_fields,
            **CSV_WRITER_KWARGS,
        )

        relevant_writer.writeheader()
        other_writer.writeheader()

        def write_relevant_rows(
            item: dict,
            *,
            postprocess_flag: str = "",
            postprocess_rule: str = "",
            postprocess_detail: str = "",
        ) -> int:
            head = {
                "file_id": sanitize(item["file_id"]),
                "date_last_updated": sanitize(item.get("date_last_updated", "")),
                "relevant_country": sanitize(item.get("relevant_country", "")),
                "target_company": sanitize(item.get("target_company", "")),
                "postprocess_flag": sanitize(postprocess_flag),
                "postprocess_rule": sanitize(postprocess_rule),
                "postprocess_detail": sanitize(postprocess_detail),
            }
            entities = item.get("entities", [])
            if not entities:
                relevant_writer.writerow(
                    {
                        "confidence": "",
                        **head,
                        "entity_name_raw": "",
                        "entity_type_guess": "",
                        "claim_substance": "",
                        "evidence_snippets": "",
                    }
                )
                return 1

            row_count = 0
            for entity in entities:
                relevant_writer.writerow(
                    {
                        "confidence": sanitize(entity.get("confidence", "")),
                        **head,
                        "entity_name_raw": sanitize(entity.get("entity_name_raw", "")),
                        "entity_type_guess": sanitize(entity.get("entity_type_guess", "")),
                        "claim_substance": sanitize(entity.get("claim_substance", "")),
                        "evidence_snippets": join_list(entity.get("evidence_snippets", [])),
                    }
                )
                row_count += 1
            return row_count

        for item in data:
            category_raw = item["presort"]["category"]
            category_norm = normalize_category(category_raw)

            if category_norm == "RELEVANT":
                relevant_count += write_relevant_rows(item)
                continue

            if category_norm == "DATABASE":
                continue

            if category_norm in other_categories:
                other_writer.writerow(
                    {
                        "file_id": sanitize(item["file_id"]),
                        "presort_category": sanitize(category_raw),
                        "presort_reason": sanitize(item["presort"].get("reason", "")),
                        "file_summary": sanitize(item.get("file_summary", "")),
                        "postprocess_flag": "",
                        "postprocess_rule": "",
                        "postprocess_detail": "",
                    }
                )
                other_count += 1
                continue

            if category_norm == "NOT_IN_TEXT":
                match = target_term_match(item["file_id"])
                if match["matched"] == "true":
                    relevant_count += write_relevant_rows(
                        item,
                        postprocess_flag=match["postprocess_flag"],
                        postprocess_rule=match["postprocess_rule"],
                        postprocess_detail=match["postprocess_detail"],
                    )
                    rescued_count += 1
                else:
                    other_writer.writerow(
                        {
                            "file_id": sanitize(item["file_id"]),
                            "presort_category": sanitize(category_raw),
                            "presort_reason": sanitize(item["presort"].get("reason", "")),
                            "file_summary": sanitize(item.get("file_summary", "")),
                            "postprocess_flag": "",
                            "postprocess_rule": "",
                            "postprocess_detail": "",
                        }
                    )
                    other_count += 1
                continue

            print(
                f"Warning: Unhandled category for file_id {item.get('file_id', '')}: {category_raw}",
                file=sys.stderr,
            )

    print(f"Done: {OUTPUT_RELEVANT} ({relevant_count} rows)")
    print(f"Done: {OUTPUT_OTHER} ({other_count} rows)")
    print(f"Rescued via post-processing rule target_term_match: {rescued_count}")
    print("All done.")


if __name__ == "__main__":
    main()
