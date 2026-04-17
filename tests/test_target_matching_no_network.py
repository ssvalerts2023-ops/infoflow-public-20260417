from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from infoflow_pipeline import extract_file_findings
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


class BombResponses:
    def create(self, *args, **kwargs):
        raise AssertionError("Unexpected network call in target matching test")


class BombClient:
    responses = BombResponses()

def make_config(tmp_path: Path, *, aliases: list[str] | None = None, max_full_file_bytes: int = 1_000) -> WorkflowConfig:
    aliases = aliases or ["Biokad", CYRILLIC_ALIAS]

    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    file_analysis_prompt = prompt_dir / "04_file_level_extractions_prompt.txt"
    relationship_prompt = prompt_dir / "relationship_prompt.txt"
    background_prompt = prompt_dir / "background_prompt.txt"
    query_prompt = prompt_dir / "query_prompt.txt"
    for prompt_path in [file_analysis_prompt, relationship_prompt, background_prompt, query_prompt]:
        prompt_path.write_text("Prompt for {TARGET_NAME}\n{FILE_TEXT}", encoding="utf-8")

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
            query_generation=query_prompt,
            file_analysis=file_analysis_prompt,
            relationship_summary=relationship_prompt,
            background_update=background_prompt,
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
        analysis=AnalysisSettings(max_full_file_bytes=max_full_file_bytes, chunk_radius=20),
        workflow=WorkflowSettings(),
    )


def sample_analysis_result(file_id: str) -> dict:
    return {
        "file_id": file_id,
        "source_url": "https://example.test/doc",
        "target_company": "Biocad",
        "date_last_updated": "2026-04-15",
        "relevant_country": "RU",
        "presort": {"category": "RELEVANT", "reason": "explicit partner mention"},
        "entities": [
            {
                "entity_name_raw": "PharmEvo",
                "entity_type_guess": "company",
                "claim_substance": "Partnership with Biocad",
                "evidence_snippets": ["PharmEvo works with Biocad."],
                "confidence": "88",
            }
        ],
        "file_summary": "Found a target-linked business relationship.",
        "parser_notes": [],
    }


def test_compile_target_pattern_matches_name_and_aliases_case_insensitively():
    pattern = extract_file_findings._compile_target_pattern("Biocad", ["Biokad", CYRILLIC_ALIAS])

    assert pattern.search("BIOCAD signed a supply agreement.")
    assert pattern.search("A Biokad affiliate was referenced.")
    assert pattern.search(CYRILLIC_SENTENCE)


def test_analyze_file_direct_match_skips_low_cost_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = make_config(tmp_path)
    extract_file_findings.configure(config)

    analyze_calls: list[str] = []

    def fake_analyze_prompt(client, prompt: str):
        analyze_calls.append(prompt)
        return sample_analysis_result("doc-direct"), None

    def fail_low_cost(*args, **kwargs):
        raise AssertionError("low_cost_name_review should not run when the target already matches directly")

    monkeypatch.setattr(extract_file_findings, "analyze_prompt", fake_analyze_prompt)
    monkeypatch.setattr(extract_file_findings, "low_cost_name_review", fail_low_cost)

    text = "This filing says BIOCAD signed a distribution agreement."
    result = extract_file_findings.analyze_file(
        BombClient(),
        "Analyze {TARGET_NAME}\n{FILE_TEXT}",
        "doc-direct",
        "https://example.test/direct",
        text,
        len(text.encode("utf-8")),
    )

    assert result["presort"]["category"] == "RELEVANT"
    assert len(analyze_calls) == 1
    assert text in analyze_calls[0]


def test_analyze_file_small_no_direct_match_uses_low_cost_review_then_stops(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = make_config(tmp_path)
    extract_file_findings.configure(config)

    low_cost_calls: list[str] = []

    def fake_low_cost(client, file_id: str, source_url: str, file_text: str) -> bool:
        low_cost_calls.append(file_text)
        return False

    def fail_analyze_prompt(*args, **kwargs):
        raise AssertionError("Full analysis should not run when low-cost review rejects the file")

    monkeypatch.setattr(extract_file_findings, "low_cost_name_review", fake_low_cost)
    monkeypatch.setattr(extract_file_findings, "analyze_prompt", fail_analyze_prompt)

    text = "This file talks about a different company entirely."
    result = extract_file_findings.analyze_file(
        BombClient(),
        "Analyze {TARGET_NAME}\n{FILE_TEXT}",
        "doc-small-no-hit",
        "https://example.test/small-no-hit",
        text,
        len(text.encode("utf-8")),
    )

    assert result["presort"] == {"category": "NOT_IN_TEXT", "reason": "SMALL"}
    assert low_cost_calls == [text]


def test_analyze_file_small_no_direct_match_can_proceed_after_low_cost_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = make_config(tmp_path)
    extract_file_findings.configure(config)

    low_cost_calls: list[str] = []
    analyze_calls: list[str] = []

    def fake_low_cost(client, file_id: str, source_url: str, file_text: str) -> bool:
        low_cost_calls.append(file_id)
        return True

    def fake_analyze_prompt(client, prompt: str):
        analyze_calls.append(prompt)
        return sample_analysis_result("doc-low-cost-yes"), None

    monkeypatch.setattr(extract_file_findings, "low_cost_name_review", fake_low_cost)
    monkeypatch.setattr(extract_file_findings, "analyze_prompt", fake_analyze_prompt)

    text = "The OCR is messy, but maybe the target is referenced indirectly."
    result = extract_file_findings.analyze_file(
        BombClient(),
        "Analyze {TARGET_NAME}\n{FILE_TEXT}",
        "doc-low-cost-yes",
        "https://example.test/low-cost-yes",
        text,
        len(text.encode("utf-8")),
    )

    assert result["presort"]["category"] == "RELEVANT"
    assert low_cost_calls == ["doc-low-cost-yes"]
    assert len(analyze_calls) == 1


def test_analyze_file_large_no_direct_match_returns_big_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config = make_config(tmp_path, max_full_file_bytes=20)
    extract_file_findings.configure(config)

    def fail_low_cost(*args, **kwargs):
        raise AssertionError("low_cost_name_review should not run for oversized files without direct matches")

    def fail_analyze_prompt(*args, **kwargs):
        raise AssertionError("Full analysis should not run for oversized files without direct matches")

    monkeypatch.setattr(extract_file_findings, "low_cost_name_review", fail_low_cost)
    monkeypatch.setattr(extract_file_findings, "analyze_prompt", fail_analyze_prompt)

    text = "This oversized file mentions nothing relevant at all."
    result = extract_file_findings.analyze_file(
        BombClient(),
        "Analyze {TARGET_NAME}\n{FILE_TEXT}",
        "doc-big-no-hit",
        "https://example.test/big-no-hit",
        text,
        999,
    )

    assert result["presort"] == {"category": "NOT_IN_TEXT", "reason": "BIG"}
