"""
Generate JSON findings for each row in relevant.csv and aggregate them.

Workflow:
  1. Read relevant.csv.
  2. Match each row to the corresponding file item and entity in file_review.json.
  3. Load the full source text for the file_id.
  4. Prompt the full OpenAI model with:
     - INPUT A: entity_under_review
     - INPUT B: row-specific JSON extract
     - INPUT C: full source text
  5. Validate the JSON response, assign canonical_name using entity_name_raw-only
     deduplication, and write one sorted JSON file plus a canonical-name review CSV.
"""

import csv
import json
import os
import re
import unicodedata
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from openai import OpenAI
from infoflow_pipeline.config import WorkflowConfig, load_default_config


BASE_DIR = Path(".")
PROMPT_FILE = BASE_DIR / "prompts" / "06_relationship_findings_prompt.txt"
CSV_PATH = BASE_DIR / "relevant.csv"
JSON_PATH = BASE_DIR / "file_review.json"
SOURCE_DIRS = [
    BASE_DIR / "serp_downloads",
]

OUT_PATH = BASE_DIR / "summarized_findings.json"
REPORT_PATH = BASE_DIR / "summarized_findings_report.json"
REVIEW_CSV_PATH = BASE_DIR / "summarized_findings_dedupped_review.csv"

MODEL = os.getenv("SUMMARY_MODEL", "gpt-5")
MAX_ATTEMPTS = max(1, int(os.getenv("SUMMARY_MAX_ATTEMPTS", "2")))


def configure(config: WorkflowConfig) -> None:
    global BASE_DIR, PROMPT_FILE, CSV_PATH, JSON_PATH, SOURCE_DIRS
    global OUT_PATH, REPORT_PATH, REVIEW_CSV_PATH, MODEL, MAX_ATTEMPTS

    BASE_DIR = config.base_dir
    PROMPT_FILE = config.prompts.relationship_summary
    CSV_PATH = config.paths.relevant_csv
    JSON_PATH = config.paths.file_review_json
    SOURCE_DIRS = list(config.paths.download_lookup_dirs)
    OUT_PATH = config.paths.findings_json
    REPORT_PATH = config.paths.findings_report_json
    REVIEW_CSV_PATH = config.paths.findings_review_csv
    MODEL = config.models.relationship_summary
    MAX_ATTEMPTS = max(1, config.analysis.summary_max_attempts)

VALID_CLASSIFICATIONS = {
    "same_as_target_company",
    "direct_counterparty",
    "probable_counterparty",
    "context_reference",
    "geographic_jurisdiction_only",
    "unclear",
}
VALID_ENTITY_TYPES = {"company", "person", "state", "unknown"}
VALID_SUPPORT_LEVELS = {
    "explicit",
    "strongly_implied",
    "weakly_implied",
    "not_supported",
}

RESULT_TEMPLATE = {
    "entity_name_raw": "",
    "entity_type_guess": "",
    "classification": "",
    "business_connection_narrative": "",
    "evidence_analysis": {
        "source_text_snippets": [],
        "support_level_from_source_text": "",
        "support_level_reason": "",
    },
    "source_narrative": {
        "file_id": "",
        "target_company": "",
        "source_url": "",
        "source_type": "",
        "source_origin_summary": "",
        "date_last_updated": "",
        "relevant_country": "",
    },
    "overall_assessment": {
        "source_weight": "",
        "source_weight_reason": "",
        "source_text_summary": "",
    },
    "negative_or_cautionary_points": [],
}

MEDIUM_CONFIDENCE_THRESHOLD = 0.86

LEGAL_SUFFIX_SEQUENCES = [
    ("co", "ltd"),
    ("company", "limited"),
    ("pvt", "ltd"),
    ("pte", "ltd"),
    ("private", "limited"),
    ("s", "a"),
    ("a", "s"),
    ("b", "v"),
    ("n", "v"),
    ("l", "l", "c"),
    ("l", "l", "p"),
]

LEGAL_SUFFIX_TOKENS = {
    "ag",
    "ao",
    "as",
    "ab",
    "bv",
    "corp",
    "corporation",
    "gmbh",
    "inc",
    "incorporated",
    "jsc",
    "kg",
    "limited",
    "llc",
    "llp",
    "lp",
    "ltd",
    "nv",
    "ooo",
    "oy",
    "oyj",
    "pao",
    "pjsc",
    "plc",
    "pte",
    "pvt",
    "sarl",
    "sa",
    "sas",
    "spa",
    "srl",
    "sro",
    "zao",
}

PUNCTUATION_TO_SPACE_PATTERN = re.compile(r"[\u2010-\u2015\-_/\\.,;:(){}\[\]|+]+")
APOSTROPHE_PATTERN = re.compile(r"[\"'`\u00b4\u2019]")
NON_ALNUM_SPACE_PATTERN = re.compile(r"[^0-9a-z ]+")
LOWERCASE_STOPWORDS = {"a", "an", "and", "for", "in", "of", "on", "the", "to"}


@dataclass
class CanonicalCandidate:
    key: str
    count: int = 0
    raw_names: Counter = field(default_factory=Counter)


@dataclass
class CanonicalCluster:
    representative_key: str
    member_keys: set[str] = field(default_factory=set)
    total_count: int = 0


def load_prompt_template() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def load_relevant_rows() -> list[dict]:
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    return [row for row in rows if any((value or "").strip() for value in row.values())]


def load_file_items() -> dict[str, dict]:
    with open(JSON_PATH, "r", encoding="utf-8") as handle:
        items = json.load(handle)
    return {item["file_id"]: item for item in items}


def normalize_text(value) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = re.sub(r"\s+", " ", value).strip()
    return value.casefold()


def split_csv_snippets(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split("|||") if part.strip()]


def entity_match_score(entity: dict, row: dict) -> tuple[int, str]:
    row_name_raw = (row.get("entity_name_raw") or "").strip()
    entity_name_raw = (entity.get("entity_name_raw") or "").strip()
    row_name_norm = normalize_text(row_name_raw)
    entity_name_norm = normalize_text(entity_name_raw)

    score = 0
    reason = "matched_none"

    if row_name_raw and row_name_raw == entity_name_raw:
        score += 100
        reason = "matched_exact_raw"
    elif row_name_norm and row_name_norm == entity_name_norm:
        score += 80
        reason = "matched_normalized_name"

    row_claim = normalize_text(row.get("claim_substance"))
    entity_claim = normalize_text(entity.get("claim_substance"))
    if row_claim and entity_claim and row_claim == entity_claim:
        score += 20
        if reason == "matched_none":
            reason = "matched_claim_substance"

    row_snippets = {normalize_text(part) for part in split_csv_snippets(row.get("evidence_snippets", ""))}
    entity_snippets = {normalize_text(part) for part in entity.get("evidence_snippets", []) if part}
    if row_snippets and entity_snippets and row_snippets.intersection(entity_snippets):
        score += 10
        if reason == "matched_none":
            reason = "matched_evidence_snippets"

    return score, reason


def match_entity(row: dict, file_item: dict) -> tuple[dict | None, str]:
    entities = file_item.get("entities") or []
    if not entities:
        return None, "matched_none"

    best_entity = None
    best_score = 0
    best_reason = "matched_none"

    for entity in entities:
        score, reason = entity_match_score(entity, row)
        if score > best_score:
            best_entity = entity
            best_score = score
            best_reason = reason

    if best_score == 0:
        return None, "matched_none"
    return best_entity, best_reason


def build_row_specific_extract(row: dict, file_item: dict, matched_entity: dict | None, match_status: str) -> dict:
    extract = {
        "file_id": file_item["file_id"],
        "source_url": file_item.get("source_url", ""),
        "target_company": file_item.get("target_company", ""),
        "date_last_updated": file_item.get("date_last_updated", "unknown"),
        "relevant_country": file_item.get("relevant_country", "unknown"),
        "presort": file_item.get("presort", {}),
        "csv_row": {
            "file_id": (row.get("file_id") or "").strip(),
            "date_last_updated": (row.get("date_last_updated") or "").strip(),
            "relevant_country": (row.get("relevant_country") or "").strip(),
            "target_company": (row.get("target_company") or "").strip(),
            "entity_name_raw": (row.get("entity_name_raw") or "").strip(),
            "entity_type_guess": (row.get("entity_type_guess") or "").strip(),
            "claim_substance": (row.get("claim_substance") or "").strip(),
            "evidence_snippets": split_csv_snippets(row.get("evidence_snippets", "")),
            "confidence": (row.get("confidence") or "").strip(),
            "postprocess_flag": (row.get("postprocess_flag") or "").strip(),
            "postprocess_rule": (row.get("postprocess_rule") or "").strip(),
            "postprocess_detail": (row.get("postprocess_detail") or "").strip(),
        },
        "matched_entity": matched_entity,
        "match_status": match_status,
        "entities_present_in_file": len(file_item.get("entities") or []),
        "file_summary": file_item.get("file_summary", ""),
        "parser_notes": file_item.get("parser_notes", []),
    }
    if matched_entity is None:
        extract["entity_names_present_in_file"] = [
            (entity.get("entity_name_raw") or "").strip() for entity in file_item.get("entities", [])
        ]
    return extract


def resolve_source_text(file_id: str) -> tuple[Path, str]:
    for source_dir in SOURCE_DIRS:
        txt_path = source_dir / f"{file_id}.txt"
        if txt_path.is_file():
            return txt_path, txt_path.read_text(encoding="utf-8", errors="replace")
    raise FileNotFoundError(f"No source text file found for file_id {file_id}")


def build_prompt(prompt_template: str, entity_under_review: str, json_extract: dict, source_text: str, reminder: str = "") -> str:
    parts = [
        prompt_template.strip(),
        "INPUT A - entity_under_review",
        entity_under_review,
        "INPUT B - structured JSON extract",
        json.dumps(json_extract, ensure_ascii=False, indent=2),
        "INPUT C - original source file text",
        source_text,
    ]
    if reminder:
        parts.extend(
            [
                "VALIDATION REMINDER",
                reminder,
            ]
        )
    return "\n\n".join(parts)


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


def parse_json_object(text: str) -> dict:
    cleaned = strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model output is not a JSON object.")
    return parsed


def ensure_list_of_strings(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_entity_type(value: str, fallback: str) -> str:
    candidate = normalize_text(value).replace(" ", "_")
    if candidate in VALID_ENTITY_TYPES:
        return candidate
    fallback_norm = normalize_text(fallback).replace(" ", "_")
    if fallback_norm in VALID_ENTITY_TYPES:
        return fallback_norm
    return "unknown"


def normalize_result(
    parsed: dict,
    entity_under_review: str,
    row: dict,
    file_item: dict,
    matched_entity: dict | None,
) -> dict:
    result = deepcopy(RESULT_TEMPLATE)
    result["entity_name_raw"] = entity_under_review
    result["entity_type_guess"] = normalize_entity_type(
        parsed.get("entity_type_guess", ""),
        (matched_entity or {}).get("entity_type_guess", "") or row.get("entity_type_guess", ""),
    )
    result["classification"] = str(parsed.get("classification", "") or "").strip()
    result["business_connection_narrative"] = str(parsed.get("business_connection_narrative", "") or "").strip()

    evidence = parsed.get("evidence_analysis", {}) if isinstance(parsed.get("evidence_analysis"), dict) else {}
    result["evidence_analysis"]["source_text_snippets"] = ensure_list_of_strings(evidence.get("source_text_snippets"))
    result["evidence_analysis"]["support_level_from_source_text"] = str(
        evidence.get("support_level_from_source_text", "") or ""
    ).strip()
    result["evidence_analysis"]["support_level_reason"] = str(evidence.get("support_level_reason", "") or "").strip()

    source_narrative = (
        parsed.get("source_narrative", {}) if isinstance(parsed.get("source_narrative"), dict) else {}
    )
    result["source_narrative"]["file_id"] = file_item["file_id"]
    result["source_narrative"]["target_company"] = file_item.get("target_company", "")
    result["source_narrative"]["source_url"] = file_item.get("source_url", "")
    result["source_narrative"]["source_type"] = str(source_narrative.get("source_type", "") or "").strip()
    result["source_narrative"]["source_origin_summary"] = str(
        source_narrative.get("source_origin_summary", "") or ""
    ).strip()
    result["source_narrative"]["date_last_updated"] = file_item.get("date_last_updated", "unknown")
    result["source_narrative"]["relevant_country"] = file_item.get("relevant_country", "unknown")

    overall = parsed.get("overall_assessment", {}) if isinstance(parsed.get("overall_assessment"), dict) else {}
    result["overall_assessment"]["source_weight"] = str(overall.get("source_weight", "") or "").strip()
    result["overall_assessment"]["source_weight_reason"] = str(
        overall.get("source_weight_reason", "") or ""
    ).strip()
    result["overall_assessment"]["source_text_summary"] = str(
        overall.get("source_text_summary", "") or ""
    ).strip()

    result["negative_or_cautionary_points"] = ensure_list_of_strings(parsed.get("negative_or_cautionary_points"))
    return result


def validate_result(result: dict) -> None:
    if result["classification"] not in VALID_CLASSIFICATIONS:
        raise ValueError(
            "classification must be one of: "
            + ", ".join(sorted(VALID_CLASSIFICATIONS))
        )
    if result["entity_type_guess"] not in VALID_ENTITY_TYPES:
        raise ValueError(
            "entity_type_guess must be one of: "
            + ", ".join(sorted(VALID_ENTITY_TYPES))
        )
    if result["evidence_analysis"]["support_level_from_source_text"] not in VALID_SUPPORT_LEVELS:
        raise ValueError(
            "support_level_from_source_text must be one of: "
            + ", ".join(sorted(VALID_SUPPORT_LEVELS))
        )
    if not result["business_connection_narrative"]:
        raise ValueError("business_connection_narrative must not be empty.")
    if not result["source_narrative"]["source_type"]:
        raise ValueError("source_narrative.source_type must not be empty.")
    if not result["source_narrative"]["source_origin_summary"]:
        raise ValueError("source_narrative.source_origin_summary must not be empty.")
    if not result["overall_assessment"]["source_weight"]:
        raise ValueError("overall_assessment.source_weight must not be empty.")
    if not result["overall_assessment"]["source_weight_reason"]:
        raise ValueError("overall_assessment.source_weight_reason must not be empty.")
    if not result["overall_assessment"]["source_text_summary"]:
        raise ValueError("overall_assessment.source_text_summary must not be empty.")


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def collapse_suffixes(tokens: list[str]) -> list[str]:
    cleaned = list(tokens)

    changed = True
    while cleaned and changed:
        changed = False

        for suffix_sequence in LEGAL_SUFFIX_SEQUENCES:
            if len(cleaned) >= len(suffix_sequence) and tuple(cleaned[-len(suffix_sequence) :]) == suffix_sequence:
                cleaned = cleaned[: -len(suffix_sequence)]
                changed = True
                break
        if changed:
            continue

        if cleaned and cleaned[-1] in LEGAL_SUFFIX_TOKENS:
            cleaned.pop()
            changed = True

    return cleaned


def normalize_entity_name(raw_name: str) -> str:
    text = unicodedata.normalize("NFKC", str(raw_name or ""))
    text = strip_accents(text)
    text = text.casefold().strip()
    text = text.replace("&", " and ")
    text = APOSTROPHE_PATTERN.sub("", text)
    text = PUNCTUATION_TO_SPACE_PATTERN.sub(" ", text)
    text = NON_ALNUM_SPACE_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    tokens = collapse_suffixes(text.split())
    return " ".join(tokens).strip()


def display_name_from_key(key: str) -> str:
    if not key:
        return ""
    return " ".join(token.capitalize() for token in key.split())


def tokenize_raw_name(raw_name: str) -> tuple[list[str], list[str]]:
    text = unicodedata.normalize("NFKC", str(raw_name or "")).strip()
    if not text:
        return [], []

    working = re.sub(r"\s+", " ", text)
    working = working.replace("&", " and ")
    working = APOSTROPHE_PATTERN.sub("", working)
    working = PUNCTUATION_TO_SPACE_PATTERN.sub(" ", working)
    working = re.sub(r"\s+", " ", working).strip()

    raw_tokens = []
    normalized_tokens = []

    for raw_token in working.split():
        normalized_token = strip_accents(unicodedata.normalize("NFKC", raw_token)).casefold()
        normalized_token = NON_ALNUM_SPACE_PATTERN.sub("", normalized_token)
        if not normalized_token:
            continue
        raw_tokens.append(raw_token)
        normalized_tokens.append(normalized_token)

    cleaned_normalized_tokens = collapse_suffixes(normalized_tokens)
    keep_count = len(cleaned_normalized_tokens)
    return raw_tokens[:keep_count], cleaned_normalized_tokens


def clean_display_name(raw_name: str) -> str:
    raw_tokens, _ = tokenize_raw_name(raw_name)
    pretty_tokens = []

    for index, token in enumerate(raw_tokens):
        if token.isupper() and any(character.isalpha() for character in token):
            pretty_tokens.append(token)
            continue

        if token.islower():
            if 0 < index < len(raw_tokens) - 1 and token in LOWERCASE_STOPWORDS:
                pretty_tokens.append(token)
            else:
                pretty_tokens.append(token.capitalize())
            continue

        pretty_tokens.append(token)

    return " ".join(pretty_tokens).strip()


def choose_preferred_display_name(candidate: CanonicalCandidate) -> str:
    ranked_raw_names = sorted(
        candidate.raw_names,
        key=lambda raw_name: (
            -candidate.raw_names[raw_name],
            len(clean_display_name(raw_name)) or 10**9,
            clean_display_name(raw_name).casefold(),
            raw_name.casefold(),
        ),
    )

    for raw_name in ranked_raw_names:
        cleaned = clean_display_name(raw_name)
        if cleaned:
            return cleaned

    return display_name_from_key(candidate.key)


def similarity_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    left_tokens = left.split()
    right_tokens = right.split()
    left_set = set(left_tokens)
    right_set = set(right_tokens)

    sequence_ratio = SequenceMatcher(None, left, right).ratio()
    token_sort_ratio = SequenceMatcher(None, " ".join(sorted(left_tokens)), " ".join(sorted(right_tokens))).ratio()

    if left_set or right_set:
        intersection = len(left_set & right_set)
        union = len(left_set | right_set)
        jaccard_ratio = intersection / union if union else 0.0
        containment_ratio = intersection / min(len(left_set), len(right_set)) if min(len(left_set), len(right_set)) else 0.0
    else:
        jaccard_ratio = 0.0
        containment_ratio = 0.0

    return max(sequence_ratio, token_sort_ratio, (jaccard_ratio + containment_ratio) / 2)


def choose_better_representative(current: str, challenger: str, candidates: dict[str, CanonicalCandidate]) -> str:
    current_candidate = candidates[current]
    challenger_candidate = candidates[challenger]

    current_rank = (-current_candidate.count, len(current), current)
    challenger_rank = (-challenger_candidate.count, len(challenger), challenger)
    return challenger if challenger_rank < current_rank else current


def build_candidates(items: list[dict]) -> dict[str, CanonicalCandidate]:
    candidates: dict[str, CanonicalCandidate] = {}

    for item in items:
        raw_name = str(item.get("entity_name_raw", "") or "").strip()
        normalized_key = normalize_entity_name(raw_name)

        if normalized_key not in candidates:
            candidates[normalized_key] = CanonicalCandidate(key=normalized_key)

        candidates[normalized_key].count += 1
        candidates[normalized_key].raw_names[raw_name] += 1

    return candidates


def build_clusters(candidates: dict[str, CanonicalCandidate], threshold: float) -> list[CanonicalCluster]:
    sorted_keys = sorted(
        candidates,
        key=lambda key: (-candidates[key].count, len(key), key),
    )

    clusters: list[CanonicalCluster] = []

    for key in sorted_keys:
        if not key:
            clusters.append(
                CanonicalCluster(
                    representative_key=key,
                    member_keys={key},
                    total_count=candidates[key].count,
                )
            )
            continue

        best_cluster = None
        best_score = 0.0

        for cluster in clusters:
            representative = cluster.representative_key
            score = similarity_score(key, representative)
            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster is not None and best_score >= threshold:
            best_cluster.member_keys.add(key)
            best_cluster.total_count += candidates[key].count
            best_cluster.representative_key = choose_better_representative(
                best_cluster.representative_key,
                key,
                candidates,
            )
            continue

        clusters.append(
            CanonicalCluster(
                representative_key=key,
                member_keys={key},
                total_count=candidates[key].count,
            )
        )

    return clusters


def build_key_to_canonical_map(
    clusters: list[CanonicalCluster],
    candidates: dict[str, CanonicalCandidate],
) -> dict[str, str]:
    key_to_canonical_name: dict[str, str] = {}

    for cluster in clusters:
        canonical_name = choose_preferred_display_name(candidates[cluster.representative_key])
        for key in cluster.member_keys:
            key_to_canonical_name[key] = canonical_name

    return key_to_canonical_name


def with_canonical_name_first(item: dict, canonical_name: str) -> dict:
    ordered_item = {"canonical_name": canonical_name}
    for key, value in item.items():
        if key == "canonical_name":
            continue
        ordered_item[key] = value
    return ordered_item


def assign_canonical_names(items: list[dict], threshold: float = MEDIUM_CONFIDENCE_THRESHOLD) -> list[dict]:
    if not items:
        return []

    candidates = build_candidates(items)
    clusters = build_clusters(candidates, threshold=threshold)
    key_to_canonical_name = build_key_to_canonical_map(clusters, candidates)

    enriched_items = []
    for item in items:
        normalized_key = normalize_entity_name(str(item.get("entity_name_raw", "") or "").strip())
        canonical_name = key_to_canonical_name.get(normalized_key, display_name_from_key(normalized_key))
        enriched_items.append(with_canonical_name_first(item, canonical_name))

    return enriched_items


def sort_key(item: dict) -> tuple[str, str, str]:
    return (
        normalize_text(item.get("canonical_name", "")),
        normalize_text(item.get("entity_name_raw", "")),
        item.get("source_narrative", {}).get("file_id", ""),
    )


def write_json(path: Path, payload) -> None:
    json_text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )
    path.write_text(json_text + "\n", encoding="utf-8")


def write_review_csv(path: Path, items: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["canonical_name", "entity_name_raw"], delimiter=";")
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "canonical_name": str(item.get("canonical_name", "") or ""),
                    "entity_name_raw": str(item.get("entity_name_raw", "") or ""),
                }
            )


def summarize_row(
    client: OpenAI,
    prompt_template: str,
    row_number: int,
    row: dict,
    file_item: dict,
    source_text: str,
    matched_entity: dict | None,
    match_status: str,
) -> tuple[dict, str, int]:
    entity_under_review = (row.get("entity_name_raw") or "").strip() or file_item.get("target_company", "")
    json_extract = build_row_specific_extract(row, file_item, matched_entity, match_status)

    reminder = ""
    last_error = ""
    last_raw_output = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = build_prompt(prompt_template, entity_under_review, json_extract, source_text, reminder=reminder)
        raw_output = call_model(client, prompt)
        last_raw_output = raw_output
        try:
            parsed = parse_json_object(raw_output)
            normalized = normalize_result(parsed, entity_under_review, row, file_item, matched_entity)
            validate_result(normalized)
            return normalized, raw_output, attempt
        except Exception as exc:
            last_error = str(exc)
            reminder = (
                "Your previous answer failed validation. "
                f"Problem: {last_error}. "
                "Return exactly one valid JSON object that follows the required schema, "
                "with no markdown fences and no extra prose."
            )

    raise ValueError(
        f"Row {row_number} failed after {MAX_ATTEMPTS} attempt(s): {last_error}\n"
        f"Last raw model output:\n{last_raw_output}"
    )


def main(config: WorkflowConfig | None = None):
    if config is None:
        config = load_default_config()
    configure(config)

    api_key = config.providers.openai_api_key.resolve_optional()
    base_url = config.providers.openai_base_url
    client = OpenAI(api_key=api_key, base_url=base_url) if api_key else OpenAI(base_url=base_url)
    prompt_template = load_prompt_template()
    rows = load_relevant_rows()
    file_items = load_file_items()

    source_text_cache = {}
    combined_results = []
    report_rows = []
    failures = 0

    total = len(rows)
    for index, row in enumerate(rows, start=1):
        file_id = (row.get("file_id") or "").strip()
        entity_name = (row.get("entity_name_raw") or "").strip()
        print(f"[{index}/{total}] Summarising file_id={file_id} entity={entity_name or '<target fallback>'}")

        if file_id not in file_items:
            failures += 1
            error_message = f"file_id {file_id} was not found in file_review.json"
            print(f"  ERROR: {error_message}")
            report_rows.append(
                {
                    "row_number": index + 1,
                    "file_id": file_id,
                    "entity_name_raw": entity_name,
                    "status": "failed",
                    "error": error_message,
                }
            )
            write_json(REPORT_PATH, report_rows)
            continue

        file_item = file_items[file_id]
        matched_entity, match_status = match_entity(row, file_item)

        try:
            if file_id not in source_text_cache:
                source_path, source_text = resolve_source_text(file_id)
                source_text_cache[file_id] = (source_path, source_text)
            source_path, source_text = source_text_cache[file_id]

            result, raw_output, attempts_used = summarize_row(
                client=client,
                prompt_template=prompt_template,
                row_number=index + 1,
                row=row,
                file_item=file_item,
                source_text=source_text,
                matched_entity=matched_entity,
                match_status=match_status,
            )

            combined_results.append(result)
            report_rows.append(
                {
                    "row_number": index + 1,
                    "file_id": file_id,
                    "entity_name_raw": result["entity_name_raw"],
                    "status": "ok",
                    "attempts_used": attempts_used,
                    "match_status": match_status,
                    "matched_entity_name": (matched_entity or {}).get("entity_name_raw", ""),
                    "postprocess_flag": (row.get("postprocess_flag") or "").strip(),
                    "postprocess_rule": (row.get("postprocess_rule") or "").strip(),
                    "source_text_path": str(source_path),
                    "raw_output_preview": raw_output[:500],
                }
            )
            print(f"  OK ({match_status}, attempts={attempts_used})")
        except Exception as exc:
            failures += 1
            error_message = str(exc)
            print(f"  ERROR: {error_message}")
            report_rows.append(
                {
                    "row_number": index + 1,
                    "file_id": file_id,
                    "entity_name_raw": entity_name,
                    "status": "failed",
                    "match_status": match_status,
                    "matched_entity_name": (matched_entity or {}).get("entity_name_raw", ""),
                    "error": error_message,
                }
            )

        write_json(REPORT_PATH, report_rows)

    final_results = assign_canonical_names(combined_results)
    final_results.sort(key=sort_key)
    write_json(OUT_PATH, final_results)
    write_review_csv(REVIEW_CSV_PATH, final_results)
    print(f"\nWrote {len(final_results)} JSON findings to {OUT_PATH}")
    print(f"Review CSV written to {REVIEW_CSV_PATH}")
    print(f"Run report written to {REPORT_PATH}")

    if failures:
        raise SystemExit(f"{failures} row(s) failed before inclusion in {OUT_PATH}. See {REPORT_PATH}.")


if __name__ == "__main__":
    main()
