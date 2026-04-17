from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import DEFAULT_CONFIG_PATH, WorkflowConfig, load_workflow_config


StageFunc = Callable[[WorkflowConfig | None], None]


@dataclass(frozen=True)
class StageDefinition:
    module_name: str
    description: str
    output_paths: Callable[[WorkflowConfig], list[Path]]


STAGE_DEFINITIONS: dict[str, StageDefinition] = {
    "generate-query-list": StageDefinition(
        module_name="infoflow_pipeline.generate_query_list",
        description="Generate the seed query CSV from the configured seed/background file.",
        output_paths=lambda cfg: [cfg.paths.master_query_csv],
    ),
    "collect-search-results": StageDefinition(
        module_name="infoflow_pipeline.collect_search_results",
        description="Collect SERP URLs from the generated query list.",
        output_paths=lambda cfg: [
            cfg.paths.master_query_csv,
            cfg.paths.master_url_csv,
            cfg.paths.query_url_map_csv,
        ],
    ),
    "download-documents": StageDefinition(
        module_name="infoflow_pipeline.download_documents",
        description="Download and normalize source documents for discovered URLs.",
        output_paths=lambda cfg: [
            cfg.paths.master_url_csv,
            cfg.paths.downloads_dir,
        ],
    ),
    "extract-file-findings": StageDefinition(
        module_name="infoflow_pipeline.extract_file_findings",
        description="Run file-level extraction over the downloaded source corpus.",
        output_paths=lambda cfg: [
            cfg.paths.file_review_json,
            cfg.paths.file_review_json.with_suffix(".checkpoint.json"),
        ],
    ),
    "flatten-findings": StageDefinition(
        module_name="infoflow_pipeline.flatten_findings",
        description="Flatten file-level findings into relevant and non-relevant CSVs.",
        output_paths=lambda cfg: [
            cfg.paths.relevant_csv,
            cfg.paths.other_csv,
        ],
    ),
    "summarize-relationships": StageDefinition(
        module_name="infoflow_pipeline.summarize_relationships",
        description="Run row-level relationship review and canonicalization.",
        output_paths=lambda cfg: [
            cfg.paths.findings_json,
            cfg.paths.findings_report_json,
            cfg.paths.findings_review_csv,
        ],
    ),
    "update-background-report": StageDefinition(
        module_name="infoflow_pipeline.update_background_report",
        description="Apply relationship findings to the rolling background report.",
        output_paths=lambda cfg: [
            cfg.paths.background_output_text,
            cfg.paths.background_output_csv,
            cfg.paths.background_output_diff,
            cfg.paths.background_artifacts_dir,
        ],
    ),
}

STAGE_ORDER = list(STAGE_DEFINITIONS)


def _load_stage_function(stage_name: str) -> StageFunc:
    try:
        module_name = STAGE_DEFINITIONS[stage_name].module_name
    except KeyError as exc:
        raise ValueError(f"Unknown stage: {stage_name}") from exc

    module = importlib.import_module(module_name)
    stage_fn = getattr(module, "main", None)
    if stage_fn is None:
        raise ValueError(f"Stage module has no main(): {module_name}")
    return stage_fn


def run_stage(stage_name: str, config: WorkflowConfig) -> None:
    stage_fn = _load_stage_function(stage_name)
    stage_fn(config)


def _print_stage_outputs(stage_name: str, config: WorkflowConfig) -> None:
    outputs = STAGE_DEFINITIONS[stage_name].output_paths(config)
    print("Expected outputs:")
    for path in outputs:
        print(f"  - {path}")


def _print_next_step_hint(stage_name: str, config: WorkflowConfig) -> None:
    index = STAGE_ORDER.index(stage_name)
    if index == len(STAGE_ORDER) - 1:
        print("This was the final stage.")
        return

    next_stage = STAGE_ORDER[index + 1]
    config_arg = f' --config "{config.config_path}"'
    print(f"Next stage: {next_stage}")
    print(
        "If you want to review or edit this stage's interim files, modify them in place, "
        f"then run: python main.py step {next_stage}{config_arg}"
    )


def print_stage_list(config: WorkflowConfig) -> None:
    print("Workflow stages:")
    for index, stage_name in enumerate(STAGE_ORDER, start=1):
        stage = STAGE_DEFINITIONS[stage_name]
        print(f"{index}. {stage_name}: {stage.description}")
        for output in stage.output_paths(config):
            print(f"   output -> {output}")


def run_workflow(
    config: WorkflowConfig,
    *,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> None:
    start_index = STAGE_ORDER.index(from_stage) if from_stage else 0
    end_index = STAGE_ORDER.index(to_stage) if to_stage else len(STAGE_ORDER) - 1
    selected_stages = STAGE_ORDER[start_index : end_index + 1]

    if not config.workflow.enable_background_update:
        selected_stages = [name for name in selected_stages if name != "update-background-report"]

    for stage_name in selected_stages:
        print(f"\n=== Running stage: {stage_name} ===")
        run_stage(stage_name, config)


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Infoflow workflow either end to end in auto mode or one stage at a time in step mode."
        )
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="auto",
        choices=["auto", "run", "step", "stages", *STAGE_ORDER],
        help="Choose auto mode, step mode, list stages, or run a specific stage directly.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        choices=STAGE_ORDER,
        help="Stage name to run in step mode.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to workflow config JSON.",
    )
    parser.add_argument("--from-stage", choices=STAGE_ORDER, help="First stage to run in auto mode.")
    parser.add_argument("--to-stage", choices=STAGE_ORDER, help="Last stage to run in auto mode.")
    args = parser.parse_args(argv)

    config = load_workflow_config(args.config)

    if args.mode in {"auto", "run"}:
        run_workflow(config, from_stage=args.from_stage, to_stage=args.to_stage)
    elif args.mode == "step":
        if not args.stage:
            parser.error("step mode requires a stage name. Example: python main.py step flatten-findings")
        run_stage(args.stage, config)
        print()
        _print_stage_outputs(args.stage, config)
        _print_next_step_hint(args.stage, config)
    elif args.mode == "stages":
        print_stage_list(config)
    else:
        run_stage(args.mode, config)
        print()
        _print_stage_outputs(args.mode, config)

    return 0
