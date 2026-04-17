from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "workflow.config.json"


@dataclass(frozen=True)
class SecretRef:
    value: str | None = None
    env: str | None = None

    @classmethod
    def from_raw(cls, raw: Any) -> "SecretRef":
        if raw is None:
            return cls()
        if isinstance(raw, str):
            return cls(value=raw)
        if not isinstance(raw, dict):
            raise TypeError(f"Secret config must be a string or object, got {type(raw).__name__}")
        return cls(
            value=(raw.get("value") or None),
            env=(raw.get("env") or None),
        )

    def resolve(self, label: str) -> str:
        if self.value:
            return self.value
        if self.env:
            resolved = (os.getenv(self.env) or "").strip()
            if resolved:
                return resolved
            raise ValueError(f"Missing required environment variable for {label}: {self.env}")
        raise ValueError(f"Missing configured credential for {label}")

    def resolve_optional(self) -> str | None:
        if self.value:
            return self.value
        if self.env:
            resolved = (os.getenv(self.env) or "").strip()
            return resolved or None
        return None


@dataclass(frozen=True)
class TargetProfileConfig:
    target_name: str
    aliases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchConfig:
    api_base_url: str = "https://api.dataforseo.com"
    api_path: str = "/v3/serp/google/organic/live/regular"
    api_retries: int = 3
    default_depth: int = 50


@dataclass(frozen=True)
class PathsConfig:
    query_seed_file: Path
    master_query_csv: Path
    master_url_csv: Path
    query_url_map_csv: Path
    downloads_dir: Path
    download_lookup_dirs: list[Path]
    file_review_json: Path
    relevant_csv: Path
    other_csv: Path
    findings_json: Path
    findings_report_json: Path
    findings_review_csv: Path
    background_file: Path
    background_output_text: Path
    background_output_csv: Path
    background_output_diff: Path
    background_artifacts_dir: Path


@dataclass(frozen=True)
class PromptsConfig:
    query_generation: Path
    file_analysis: Path
    relationship_summary: Path
    background_update: Path


@dataclass(frozen=True)
class ModelsConfig:
    query_generation: str = "gpt-5"
    file_analysis: str = "gpt-5-mini"
    relationship_summary: str = "gpt-5"
    background_update: str = "gpt-5"


@dataclass(frozen=True)
class ProvidersConfig:
    dataforseo_login: SecretRef
    dataforseo_password: SecretRef
    cloudflare_api_token: SecretRef
    cloudflare_account_id: SecretRef
    openai_api_key: SecretRef = field(default_factory=SecretRef)
    openai_base_url: str = "https://api.openai.com/v1"


@dataclass(frozen=True)
class FetchSettings:
    req_timeout: tuple[float, float] = (10.0, 50.0)
    req_retries: int = 2
    trafi_retries: int = 1
    backoff: float = 0.8
    cf_retries: int = 4
    cf_backoff_base: float = 2.0
    cf_backoff_cap: float = 20.0
    cloudflare_api_base_url: str = "https://api.cloudflare.com/client/v4"
    cloudflare_nav_timeout_ms: int = 60_000
    cloudflare_action_timeout_ms: int = 120_000
    cloudflare_extra_wait_ms: int = 4_000
    denylist_domains: list[str] = field(default_factory=list)
    per_domain_min_interval_sec: float = 1.0
    jitter_range_sec: tuple[float, float] = (0.05, 0.25)


@dataclass(frozen=True)
class AnalysisSettings:
    query_generation_max_attempts: int = 2
    max_full_file_bytes: int = 70 * 1024
    chunk_radius: int = 5_000
    summary_max_attempts: int = 2
    background_update_max_attempts: int = 2
    background_block_size: int = 10


@dataclass(frozen=True)
class WorkflowSettings:
    enable_background_update: bool = True


@dataclass(frozen=True)
class WorkflowConfig:
    config_path: Path
    base_dir: Path
    target_profile: TargetProfileConfig
    search: SearchConfig
    paths: PathsConfig
    prompts: PromptsConfig
    models: ModelsConfig
    providers: ProvidersConfig
    fetch: FetchSettings
    analysis: AnalysisSettings
    workflow: WorkflowSettings

    @property
    def target(self) -> TargetProfileConfig:
        return self.target_profile


def _resolve_path(base_dir: Path, raw: str, *, fallback: str | None = None) -> Path:
    value = raw or fallback
    if not value:
        raise ValueError("Missing required path value in workflow config")
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_path_list(base_dir: Path, raw: Any, *, fallback: list[str] | None = None) -> list[Path]:
    values = raw if raw is not None else fallback or []
    if not isinstance(values, list):
        raise TypeError("Expected a list of paths")
    return [_resolve_path(base_dir, str(value)) for value in values]


def _resolve_secret(raw: Any) -> SecretRef:
    return SecretRef.from_raw(raw)


def _get_section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{key}' must be an object")
    return value


def _parse_structured_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyYAML is required to load target_profile YAML files. "
                "Install it or keep target_profile inline in workflow.config.json."
            ) from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported structured config format: {path}")

    if not isinstance(payload, dict):
        raise ValueError(f"Structured config file must contain an object at the root: {path}")
    return payload


def _load_target_profile(raw: Any, base_dir: Path) -> TargetProfileConfig:
    if isinstance(raw, str):
        raw = _parse_structured_file(_resolve_path(base_dir, raw))
    elif not isinstance(raw, dict):
        raise TypeError("Config target_profile must be an object or a path string")

    target_profile = TargetProfileConfig(
        target_name=str(raw.get("target_name") or "").strip(),
        aliases=[str(value) for value in raw.get("aliases", [])],
    )
    if not target_profile.target_name:
        raise ValueError("Config target_profile.target_name is required")
    return target_profile


def load_workflow_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> WorkflowConfig:
    path = Path(config_path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Workflow config root must be a JSON object")

    base_dir = path.parent
    target_profile_raw = data.get("target_profile", data.get("target"))
    search_section = _get_section(data, "search")
    path_section = _get_section(data, "paths")
    prompt_section = _get_section(data, "prompts")
    model_section = _get_section(data, "models")
    provider_section = _get_section(data, "providers")
    fetch_section = _get_section(data, "fetch")
    analysis_section = _get_section(data, "analysis")
    workflow_section = _get_section(data, "workflow")

    target_profile = _load_target_profile(target_profile_raw, base_dir)

    search = SearchConfig(
        api_base_url=str(search_section.get("api_base_url") or SearchConfig.api_base_url),
        api_path=str(search_section.get("api_path") or SearchConfig.api_path),
        api_retries=int(search_section.get("api_retries", SearchConfig.api_retries)),
        default_depth=int(search_section.get("default_depth", SearchConfig.default_depth)),
    )

    downloads_dir = _resolve_path(base_dir, path_section.get("downloads_dir", "03_source_documents"))
    paths = PathsConfig(
        query_seed_file=_resolve_path(
            base_dir,
            path_section.get("query_seed_file", path_section.get("background_file", "sample_data/00_seed_background.txt")),
        ),
        master_query_csv=_resolve_path(base_dir, path_section.get("master_query_csv", "01_search_queries.csv")),
        master_url_csv=_resolve_path(base_dir, path_section.get("master_url_csv", "02_discovered_urls.csv")),
        query_url_map_csv=_resolve_path(
            base_dir,
            path_section.get("query_url_map_csv", "02_query_url_provenance.csv"),
        ),
        downloads_dir=downloads_dir,
        download_lookup_dirs=_resolve_path_list(
            base_dir,
            path_section.get("download_lookup_dirs"),
            fallback=[str(downloads_dir)],
        ),
        file_review_json=_resolve_path(
            base_dir,
            path_section.get("file_review_json", "04_file_level_extractions.json"),
        ),
        relevant_csv=_resolve_path(
            base_dir,
            path_section.get("relevant_csv", "05_relevant_entity_rows.csv"),
        ),
        other_csv=_resolve_path(
            base_dir,
            path_section.get("other_csv", "05_nonrelevant_or_unclassified_rows.csv"),
        ),
        findings_json=_resolve_path(
            base_dir,
            path_section.get("findings_json", "06_relationship_findings.json"),
        ),
        findings_report_json=_resolve_path(
            base_dir,
            path_section.get("findings_report_json", "06_relationship_run_report.json"),
        ),
        findings_review_csv=_resolve_path(
            base_dir,
            path_section.get("findings_review_csv", "06_canonical_name_review.csv"),
        ),
        background_file=_resolve_path(
            base_dir,
            path_section.get("background_file", "sample_data/00_seed_background.txt"),
        ),
        background_output_text=_resolve_path(
            base_dir,
            path_section.get("background_output_text", "07_updated_background_report.txt"),
        ),
        background_output_csv=_resolve_path(
            base_dir,
            path_section.get("background_output_csv", "07_background_change_log.csv"),
        ),
        background_output_diff=_resolve_path(
            base_dir,
            path_section.get("background_output_diff", "07_background_diff.html"),
        ),
        background_artifacts_dir=_resolve_path(
            base_dir,
            path_section.get("background_artifacts_dir", "07_background_update_artifacts"),
        ),
    )

    prompts = PromptsConfig(
        query_generation=_resolve_path(
            base_dir,
            prompt_section.get("query_generation", "prompts/01_search_queries_generation_prompt.txt"),
        ),
        file_analysis=_resolve_path(base_dir, prompt_section.get("file_analysis", "prompts/04_file_level_extractions_prompt.txt")),
        relationship_summary=_resolve_path(
            base_dir,
            prompt_section.get("relationship_summary", "prompts/06_relationship_findings_prompt.txt"),
        ),
        background_update=_resolve_path(
            base_dir,
            prompt_section.get(
                "background_update",
                "prompts/07_updated_background_report_prompt.txt",
            ),
        ),
    )

    models = ModelsConfig(
        query_generation=str(model_section.get("query_generation", ModelsConfig.query_generation)),
        file_analysis=str(model_section.get("file_analysis", ModelsConfig.file_analysis)),
        relationship_summary=str(
            model_section.get("relationship_summary", ModelsConfig.relationship_summary)
        ),
        background_update=str(model_section.get("background_update", ModelsConfig.background_update)),
    )

    providers = ProvidersConfig(
        dataforseo_login=_resolve_secret(provider_section.get("dataforseo_login")),
        dataforseo_password=_resolve_secret(provider_section.get("dataforseo_password")),
        cloudflare_api_token=_resolve_secret(provider_section.get("cloudflare_api_token")),
        cloudflare_account_id=_resolve_secret(provider_section.get("cloudflare_account_id")),
        openai_api_key=_resolve_secret(provider_section.get("openai_api_key")),
        openai_base_url=str(provider_section.get("openai_base_url") or ProvidersConfig.openai_base_url),
    )

    fetch = FetchSettings(
        req_timeout=tuple(fetch_section.get("req_timeout", list(FetchSettings.req_timeout))),
        req_retries=int(fetch_section.get("req_retries", FetchSettings.req_retries)),
        trafi_retries=int(fetch_section.get("trafi_retries", FetchSettings.trafi_retries)),
        backoff=float(fetch_section.get("backoff", FetchSettings.backoff)),
        cf_retries=int(fetch_section.get("cf_retries", FetchSettings.cf_retries)),
        cf_backoff_base=float(fetch_section.get("cf_backoff_base", FetchSettings.cf_backoff_base)),
        cf_backoff_cap=float(fetch_section.get("cf_backoff_cap", FetchSettings.cf_backoff_cap)),
        cloudflare_api_base_url=str(
            fetch_section.get("cloudflare_api_base_url", FetchSettings.cloudflare_api_base_url)
        ),
        cloudflare_nav_timeout_ms=int(
            fetch_section.get("cloudflare_nav_timeout_ms", FetchSettings.cloudflare_nav_timeout_ms)
        ),
        cloudflare_action_timeout_ms=int(
            fetch_section.get(
                "cloudflare_action_timeout_ms",
                FetchSettings.cloudflare_action_timeout_ms,
            )
        ),
        cloudflare_extra_wait_ms=int(
            fetch_section.get("cloudflare_extra_wait_ms", FetchSettings.cloudflare_extra_wait_ms)
        ),
        denylist_domains=[str(value).lower() for value in fetch_section.get("denylist_domains", [])],
        per_domain_min_interval_sec=float(
            fetch_section.get(
                "per_domain_min_interval_sec",
                FetchSettings.per_domain_min_interval_sec,
            )
        ),
        jitter_range_sec=tuple(fetch_section.get("jitter_range_sec", list(FetchSettings.jitter_range_sec))),
    )

    analysis = AnalysisSettings(
        query_generation_max_attempts=int(
            analysis_section.get(
                "query_generation_max_attempts",
                AnalysisSettings.query_generation_max_attempts,
            )
        ),
        max_full_file_bytes=int(
            analysis_section.get("max_full_file_bytes", AnalysisSettings.max_full_file_bytes)
        ),
        chunk_radius=int(analysis_section.get("chunk_radius", AnalysisSettings.chunk_radius)),
        summary_max_attempts=int(
            analysis_section.get("summary_max_attempts", AnalysisSettings.summary_max_attempts)
        ),
        background_update_max_attempts=int(
            analysis_section.get(
                "background_update_max_attempts",
                AnalysisSettings.background_update_max_attempts,
            )
        ),
        background_block_size=int(
            analysis_section.get("background_block_size", AnalysisSettings.background_block_size)
        ),
    )

    workflow = WorkflowSettings(
        enable_background_update=bool(
            workflow_section.get("enable_background_update", WorkflowSettings.enable_background_update)
        )
    )

    return WorkflowConfig(
        config_path=path,
        base_dir=base_dir,
        target_profile=target_profile,
        search=search,
        paths=paths,
        prompts=prompts,
        models=models,
        providers=providers,
        fetch=fetch,
        analysis=analysis,
        workflow=workflow,
    )


def load_default_config() -> WorkflowConfig:
    return load_workflow_config(DEFAULT_CONFIG_PATH)
