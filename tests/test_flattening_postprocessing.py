from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from infoflow_pipeline import flatten_findings
from infoflow_pipeline.config import (
    AnalysisSettings,
    FetchSettings,
    ModelsConfig,
    PathsConfig,
    PromptsConfig,
    ProvidersConfig,
    SearchConfig,
    SecretRef,
    TargetProfileConfig,
    WorkflowConfig,
    WorkflowSettings,
)


CYRILLIC_ALIAS = "\u0411\u0438\u043e\u043a\u0430\u0434"
CYRILLIC_UPPER_ALIAS = "\u0411\u0418\u041e\u041a\u0410\u0414"
CYRILLIC_SENTENCE = (
    "\u041f\u0430\u0440\u0442\u043d\u0435\u0440\u043e\u043c "
    "\u0432\u044b\u0441\u0442\u0443\u043f\u0430\u0435\u0442 "
    f"{CYRILLIC_UPPER_ALIAS}, "
    "\u0443\u043a\u0430\u0437\u0430\u043d\u043d\u044b\u0439 "
    "\u0432 \u0438\u0441\u0445\u043e\u0434\u043d\u043e\u043c "
    "\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0435."
)


def make_config(tmp_path: Path, *, aliases: list[str] | None = None) -> WorkflowConfig:
    aliases = aliases or ["Biokad", CYRILLIC_ALIAS]

    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    placeholder_prompt = prompt_dir / "placeholder.txt"
    placeholder_prompt.write_text("placeholder", encoding="utf-8")

    return WorkflowConfig(
        config_path=tmp_path / "workflow.config.json",
        base_dir=tmp_path,
        target_profile=TargetProfileConfig(
            target_name="Biocad",
            aliases=aliases,
        ),
        search=SearchConfig(),
        paths=PathsConfig(
            query_seed_file=tmp_path / "seed.txt",
            master_query_csv=tmp_path / "queries.csv",
            master_url_csv=tmp_path / "urls.csv",
            query_url_map_csv=tmp_path / "query_url_map.csv",
            downloads_dir=downloads_dir,
            download_lookup_dirs=[downloads_dir],
            file_review_json=tmp_path / "04_file_level_extractions.json",
            relevant_csv=tmp_path / "05_relevant_entity_rows.csv",
            other_csv=tmp_path / "05_nonrelevant_or_unclassified_rows.csv",
            findings_json=tmp_path / "06_relationship_findings.json",
            findings_report_json=tmp_path / "06_relationship_run_report.json",
            findings_review_csv=tmp_path / "06_canonical_name_review.csv",
            background_file=tmp_path / "background.txt",
            background_output_text=tmp_path / "07_updated_background_report.txt",
            background_output_csv=tmp_path / "07_background_change_log.csv",
            background_output_diff=tmp_path / "07_background_diff.html",
            background_artifacts_dir=tmp_path / "07_background_update_artifacts",
        ),
        prompts=PromptsConfig(
            query_generation=placeholder_prompt,
            file_analysis=placeholder_prompt,
            relationship_summary=placeholder_prompt,
            background_update=placeholder_prompt,
        ),
        models=ModelsConfig(),
        providers=ProvidersConfig(
            dataforseo_login=SecretRef(),
            dataforseo_password=SecretRef(),
            cloudflare_api_token=SecretRef(),
            cloudflare_account_id=SecretRef(),
            openai_api_key=SecretRef(),
        ),
        fetch=FetchSettings(),
        analysis=AnalysisSettings(),
        workflow=WorkflowSettings(),
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def write_file_review(path: Path, items: list[dict]) -> None:
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def test_flatten_main_expands_relevant_entities_into_row_level_csv(tmp_path: Path):
    config = make_config(tmp_path)
    write_file_review(
        config.paths.file_review_json,
        [
            {
                "file_id": "doc-relevant",
                "source_url": "https://example.test/relevant",
                "target_company": "Biocad",
                "date_last_updated": "2026-04-15",
                "relevant_country": "RU",
                "presort": {"category": "RELEVANT", "reason": "explicit partner mention"},
                "entities": [
                    {
                        "entity_name_raw": "PharmEvo",
                        "entity_type_guess": "company",
                        "claim_substance": "Distribution partner",
                        "evidence_snippets": ["Signed distribution agreement", "Regional rollout"],
                        "confidence": "88",
                    },
                    {
                        "entity_name_raw": "Julphar",
                        "entity_type_guess": "company",
                        "claim_substance": "License and supply agreement",
                        "evidence_snippets": ["License and supply agreement"],
                        "confidence": "91",
                    },
                ],
                "file_summary": "Two named counterparties found.",
                "parser_notes": [],
            }
        ],
    )

    flatten_findings.main(config)

    relevant_rows = read_csv_rows(config.paths.relevant_csv)
    other_rows = read_csv_rows(config.paths.other_csv)

    assert len(relevant_rows) == 2
    assert [row["entity_name_raw"] for row in relevant_rows] == ["PharmEvo", "Julphar"]
    assert relevant_rows[0]["evidence_snippets"] == "Signed distribution agreement ||| Regional rollout"
    assert relevant_rows[1]["claim_substance"] == "License and supply agreement"
    assert other_rows == []


def test_flatten_main_routes_nonrelevant_categories_to_other_csv(tmp_path: Path):
    config = make_config(tmp_path)
    write_file_review(
        config.paths.file_review_json,
        [
            {
                "file_id": "doc-list",
                "source_url": "https://example.test/list",
                "target_company": "Biocad",
                "date_last_updated": "2026-04-15",
                "relevant_country": "RU",
                "presort": {"category": "LIST", "reason": "directory-style mention only"},
                "entities": [],
                "file_summary": "Just a list of names.",
                "parser_notes": [],
            }
        ],
    )

    flatten_findings.main(config)

    relevant_rows = read_csv_rows(config.paths.relevant_csv)
    other_rows = read_csv_rows(config.paths.other_csv)

    assert relevant_rows == []
    assert other_rows == [
        {
            "file_id": "doc-list",
            "presort_category": "LIST",
            "presort_reason": "directory-style mention only",
            "file_summary": "Just a list of names.",
            "postprocess_flag": "",
            "postprocess_rule": "",
            "postprocess_detail": "",
        }
    ]


def test_flatten_main_promotes_not_in_text_rows_when_configured_target_term_is_found(tmp_path: Path):
    config = make_config(tmp_path)
    (config.paths.downloads_dir / "doc-rescue.txt").write_text(CYRILLIC_SENTENCE, encoding="utf-8")
    write_file_review(
        config.paths.file_review_json,
        [
            {
                "file_id": "doc-rescue",
                "source_url": "https://example.test/rescue",
                "target_company": "Biocad",
                "date_last_updated": "2026-04-15",
                "relevant_country": "RU",
                "presort": {"category": "NOT_IN_TEXT", "reason": "SMALL"},
                "entities": [
                    {
                        "entity_name_raw": "PharmEvo",
                        "entity_type_guess": "company",
                        "claim_substance": "Potential partner mention",
                        "evidence_snippets": ["PharmEvo appears near the target mention."],
                        "confidence": "51",
                    }
                ],
                "file_summary": "OCR fallback case.",
                "parser_notes": [],
            }
        ],
    )

    flatten_findings.main(config)

    relevant_rows = read_csv_rows(config.paths.relevant_csv)
    other_rows = read_csv_rows(config.paths.other_csv)

    assert len(relevant_rows) == 1
    assert relevant_rows[0]["file_id"] == "doc-rescue"
    assert relevant_rows[0]["postprocess_flag"] == "true"
    assert relevant_rows[0]["postprocess_rule"] == "target_term_match"
    assert CYRILLIC_ALIAS in relevant_rows[0]["postprocess_detail"]
    assert other_rows == []


def test_flatten_main_keeps_unrescued_not_in_text_rows_in_other_csv(tmp_path: Path):
    config = make_config(tmp_path)
    (config.paths.downloads_dir / "doc-other.txt").write_text(
        "This file contains no configured target term.",
        encoding="utf-8",
    )
    write_file_review(
        config.paths.file_review_json,
        [
            {
                "file_id": "doc-other",
                "source_url": "https://example.test/other",
                "target_company": "Biocad",
                "date_last_updated": "2026-04-15",
                "relevant_country": "RU",
                "presort": {"category": "NOT_IN_TEXT", "reason": "SMALL"},
                "entities": [],
                "file_summary": "No target mention recovered.",
                "parser_notes": [],
            }
        ],
    )

    flatten_findings.main(config)

    relevant_rows = read_csv_rows(config.paths.relevant_csv)
    other_rows = read_csv_rows(config.paths.other_csv)

    assert relevant_rows == []
    assert other_rows == [
        {
            "file_id": "doc-other",
            "presort_category": "NOT_IN_TEXT",
            "presort_reason": "SMALL",
            "file_summary": "No target mention recovered.",
            "postprocess_flag": "",
            "postprocess_rule": "",
            "postprocess_detail": "",
        }
    ]
