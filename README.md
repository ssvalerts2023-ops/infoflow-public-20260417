# Infoflow

An AI-powered OSINT pipeline for discovering and analyzing business relationships around a target entity. Given a seed background file describing a company or individual, Infoflow generates search queries, collects search-engine results, downloads source documents, extracts entity mentions, canonicalizes findings, and integrates everything into a narrative intelligence report, which can be fed again into the workflow from step one to obtain further, more detailed info.

The architecture is inspired by [Karpathy's LLM Wiki publication](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), which describes a pattern where an LLM maintains a persistent, structured knowledge base rather than re-deriving answers from raw documents on each query. Every new source adds to and refines the wiki, so knowledge compounds. Infoflow adopts this pattern and extends it in one direction: automated, target-oriented OSINT, the product of which can be re-fed to the workflow and further refined.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Stages](#pipeline-stages)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Files](#data-files)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Overview

Infoflow runs a 7-stage pipeline:

```
Seed background â†’ Search queries â†’ SERP collection â†’ Document download
    â†’ Entity extraction â†’ Flattening â†’ Relationship summary â†’ Background update
```

### Human review

Each stage reads inputs produced by the previous stage and writes its own outputs to disk. 

All stages can be run automatically in sequence (consuming the product of the previous stage without human review) or can be individually started (with human review of the output/input files at the end of a particular stage); the runner can resume from any point. 

All human reviewable CSV files are delimited by ";" only to avoid interference by commas in the evidence snippet text.


**External services required:**

| Service | Purpose |
|---|---|
| OpenAI API | Query generation, entity extraction, summarization, report update |
| DataForSEO API | Search engine results (SERP) collection |
| Cloudflare Browser Rendering API | JavaScript-heavy page fetching (optional) |

---

## Pipeline Stages

### Stage 1 â€” `generate-query-list`

Reads the seed background file and calls an LLM to produce 100 targeted search queries.

- **Input:** `sample_data/00_seed_background.txt`
- **Output:** `01_search_queries.csv` (columns: `keyword`, `language_code`, `location_code`, `depth`)
- **Model:** configured under `models.query_generation` (default: `gpt-5`)

NOTE: the initial '00_seed_background.txt' file can be one of the following:
- an available OSINT narrative on the target
- a minimal seed info file, stating expressly target name and the aliases only
- the final output of Stage 7 run, in line with Karpathy's pattern of self-compounding knowledge file.

---

### Stage 2 â€” `collect-search-results`

Submits each query to the DataForSEO organic SERP API and collects unique result URLs.

- **Input:** `01_search_queries.csv`
- **Output:** `02_discovered_urls.csv`, `02_query_url_provenance.csv`
- **API:** DataForSEO (`dataforseo_login` / `dataforseo_password`)

---

### Stage 3 â€” `download-documents`

Downloads each URL. Tries plain HTTP, then Trafilatura extraction, then Cloudflare Browser Rendering. Applies per-domain rate limiting and retry logic.

- **Input:** `02_discovered_urls.csv`
- **Output:** `03_source_documents/<id>.txt`, download metadata columns written back to `02_discovered_urls.csv`

---

### Stage 4 - `extract-file-findings`

Sends each downloaded document to an LLM for entity extraction. Finds business relationships, transaction signals, and confidence scores. Writes a checkpoint file after each processed document.

- **Input:** `03_source_documents/`, `02_discovered_urls.csv`
- **Output:** `04_file_level_extractions.json`, `04_file_level_extractions.checkpoint.json`
- **Model:** configured under `models.file_analysis` (default: `gpt-5-mini`)

---

### Stage 5 - `flatten-findings`

Flattens the nested extraction JSON into two CSVs. Applies post-processing by re-checking `NOT_IN_TEXT` items against configured target terms in source text and promoting matching rows into the relevant output.

- **Input:** `04_file_level_extractions.json`, source text files from `paths.download_lookup_dirs`
- **Output:** `05_relevant_entity_rows.csv`, `05_nonrelevant_or_unclassified_rows.csv`

---

### Stage 6 - `summarize-relationships`

Reviews each relevant entity row with full source-text context. Classifies each relationship, assigns a canonical entity name, and deduplicates across sources.

- **Input:** `05_relevant_entity_rows.csv`, `04_file_level_extractions.json`, source text files from `paths.download_lookup_dirs`
- **Output:** `06_relationship_findings.json`, `06_relationship_run_report.json`, `06_canonical_name_review.csv`
- **Model:** configured under `models.relationship_summary` (default: `gpt-5`)
- **Classifications:** `same_as_target_company`, `direct_counterparty`, `probable_counterparty`, `context_reference`, `geographic_jurisdiction_only`, `unclear`

---

### Stage 7 - `update-background-report` *(optional)*

Integrates the canonicalized findings into the seed background narrative. Preserves all existing content. Emits a structured change log and an HTML diff. Disable with `workflow.enable_background_update: false`.

- **Input:** `06_relationship_findings.json`, `sample_data/00_seed_background.txt`
- **Output:** `07_updated_background_report.txt`, `07_background_change_log.csv`, `07_background_diff.html`, `07_background_update_artifacts/`
- **Model:** configured under `models.background_update` (default: `gpt-5`)

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd infoflow
pip install -r requirements.txt

# 2. Set credentials
export OPENAI_API_KEY=sk-...
export SEO_USER=your_dataforseo_login
export SEO_PASS=your_dataforseo_password

# 3. Describe your target
#    Edit sample_data/00_seed_background.txt and workflow.config.json
#    (set target_profile.target_name and target_profile.aliases)

# 4. Run the full pipeline
python main.py
```

NOTE: target name and aliases where relevant must be entered twice: in the seed file and separately in workflow.config.json. Name is not automatically duplicated to avoid potential mistakes and disambiguation issues.

---

## Installation

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

If you need Playwright for JavaScript-rendered pages:

```bash
playwright install chromium
```

**Dependencies:**

| Package | Purpose |
|---|---|
| `requests>=2.32` | HTTP document fetching |
| `beautifulsoup4>=4.12` | HTML parsing |
| `trafilatura>=2.0` | Web content extraction |
| `markdownify>=1.2` | HTML-to-text conversion |
| `pandas>=2.3` | CSV/dataframe processing |
| `openai>=2.9` | LLM API client |
| `PyYAML>=6.0` | Optional YAML config support |
| `pdfminer.six>=20250506` | PDF text extraction |
| `playwright>=1.55` | Browser automation |
| `playwright-stealth>=1.0` | Anti-bot evasion for Playwright |

---

## Configuration

All settings live in `workflow.config.json`. Every path can be absolute or relative to the config file.

### `target_profile`

```json
"target_profile": {
  "target_name": "Acme Corp",
  "aliases": ["Acme", "ÐÐšÐœÐ•"]
}
```

Can also reference an external YAML/JSON file:

```json
"target_profile": "targets/acme.yaml"
```

---

### `providers` â€” credentials

Credentials can be inline strings or references to environment variables:

```json
"providers": {
  "openai_api_key":        { "env": "OPENAI_API_KEY" },
  "openai_base_url":       "https://api.openai.com/v1",
  "dataforseo_login":      { "env": "SEO_USER" },
  "dataforseo_password":   { "env": "SEO_PASS" },
  "cloudflare_api_token":  { "env": "FLARE_API_KEY" },
  "cloudflare_account_id": { "env": "FLARE_CLIENT_ID" }
}
```

---

### `models` â€” LLM selection per stage

```json
"models": {
  "query_generation":    "gpt-5",
  "file_analysis":       "gpt-5-mini",
  "relationship_summary": "gpt-5",
  "background_update":   "gpt-5"
}
```

---

### `search` â€” DataForSEO options

| Key | Default | Description |
|---|---|---|
| `api_base_url` | `https://api.dataforseo.com` | DataForSEO base URL |
| `api_path` | `/v3/serp/google/organic/live/regular` | SERP endpoint |
| `api_retries` | `3` | Retry count on API error |
| `default_depth` | `50` | Results per query |

NOTE: DataForSEO also consumes language_code - hardcoded to "en" and location_code - hardcoded to "2826", United Kingdom. These can be manually amended, but hardcoded for simplicity.

---

### `fetch` â€” document download behavior

| Key | Default | Description |
|---|---|---|
| `req_timeout` | `[10.0, 50.0]` | `[connect_timeout, read_timeout]` in seconds |
| `req_retries` | `2` | HTTP retry count |
| `trafi_retries` | `1` | Trafilatura retry count |
| `backoff` | `0.8` | Exponential backoff multiplier |
| `cf_retries` | `4` | Cloudflare browser rendering retries |
| `cf_backoff_base` | `2.0` | Cloudflare backoff base seconds |
| `cf_backoff_cap` | `20.0` | Cloudflare max backoff seconds |
| `cloudflare_api_base_url` | `https://api.cloudflare.com/client/v4` | Cloudflare API |
| `cloudflare_nav_timeout_ms` | `60000` | Navigation timeout |
| `cloudflare_action_timeout_ms` | `120000` | Action timeout |
| `cloudflare_extra_wait_ms` | `4000` | Extra wait after page load |
| `denylist_domains` | `[]` | Domains to skip entirely |
| `per_domain_min_interval_sec` | `1.0` | Minimum seconds between requests to the same domain |
| `jitter_range_sec` | `[0.05, 0.25]` | Random jitter added to each request |

---

### `analysis` â€” LLM processing parameters

| Key | Default | Description |
|---|---|---|
| `query_generation_max_attempts` | `2` | Retries for query generation |
| `max_full_file_bytes` | `71680` | Max bytes per file sent to LLM (~70 KB) |
| `chunk_radius` | `5000` | Context window radius for chunked analysis |
| `summary_max_attempts` | `2` | Retries for relationship summarization |
| `background_update_max_attempts` | `2` | Retries for background update |
| `background_block_size` | `10` | Findings per background-update block |

---

### `workflow` â€” pipeline switches

| Key | Default | Description |
|---|---|---|
| `enable_background_update` | `true` | Set `false` to skip stage 7 |

---

### `paths` â€” output file locations

All paths can be overridden. Defaults:

| Key | Default |
|---|---|
| `query_seed_file` | `sample_data/00_seed_background.txt` |
| `master_query_csv` | `01_search_queries.csv` |
| `master_url_csv` | `02_discovered_urls.csv` |
| `query_url_map_csv` | `02_query_url_provenance.csv` |
| `downloads_dir` | `03_source_documents` |
| `download_lookup_dirs` | `[03_source_documents]` |
| `file_review_json` | `04_file_level_extractions.json` |
| `relevant_csv` | `05_relevant_entity_rows.csv` |
| `other_csv` | `05_nonrelevant_or_unclassified_rows.csv` |
| `findings_json` | `06_relationship_findings.json` |
| `findings_report_json` | `06_relationship_run_report.json` |
| `findings_review_csv` | `06_canonical_name_review.csv` |
| `background_file` | `sample_data/00_seed_background.txt` |
| `background_output_text` | `07_updated_background_report.txt` |
| `background_output_csv` | `07_background_change_log.csv` |
| `background_output_diff` | `07_background_diff.html` |
| `background_artifacts_dir` | `07_background_update_artifacts` |

---

## Usage

### Run the full pipeline

```bash
python main.py
# or equivalently:
python main.py auto
python main.py run
```

### Resume from a specific stage

```bash
python main.py auto --from-stage summarize-relationships
python main.py auto --from-stage download-documents --to-stage flatten-findings
```

### Run a single stage

```bash
python main.py step generate-query-list
python main.py step download-documents
# or invoke the stage name directly:
python main.py flatten-findings
python main.py update-background-report
```

### List all stages

```bash
python main.py stages
```

### Use an alternate config file

```bash
python main.py --config /path/to/other.config.json auto
```

---

## Data Files

### Inputs

| File | Description |
|---|---|
| `00_seed_background.txt` | Freeform narrative describing the target: name, aliases, locale hints, prior findings |
| `workflow.config.json` | Pipeline configuration |

### Outputs (in stage order)

| File | Description |
|---|---|
| `01_search_queries.csv` | Generated search queries with language/location parameters |
| `02_discovered_urls.csv` | Deduplicated URL ledger with download tracking metadata |
| `02_query_url_provenance.csv` | Query-to-URL mapping for traceability |
| `03_source_documents/<id>.txt` | Normalized text extracted from each downloaded page or PDF |
| `03_source_documents/<id>.html` | Saved rendered HTML artifact for non-PDF pages |
| `03_source_documents/<id>.pdf` | Saved PDF artifact when the fetched source is a PDF |
| `04_file_level_extractions.json` | Per-file entity extraction results with confidence scores and evidence snippets |
| `04_file_level_extractions.checkpoint.json` | Rolling checkpoint written during stage 4 processing |
| `05_relevant_entity_rows.csv` | Entities with transaction signals (semicolon-delimited) |
| `05_nonrelevant_or_unclassified_rows.csv` | Context mentions and list-directory entries |
| `06_relationship_findings.json` | Canonical, deduplicated relationship findings |
| `06_relationship_run_report.json` | Execution metadata and error log for stage 6 |
| `06_canonical_name_review.csv` | Entity canonicalization decisions and variant mapping |
| `07_updated_background_report.txt` | Enriched narrative (with `[section_id:]` and `[file_id:]` citations) |
| `07_background_change_log.csv` | Per-item amendment log with action type and section reference |
| `07_background_diff.html` | Side-by-side visual diff of the background before and after update |

### `06_relationship_findings.json` record schema

```json
{
  "canonical_name": "Acme Logistics",
  "entity_name_raw": "Acme Logistics GmbH",
  "entity_type_guess": "company",
  "classification": "direct_counterparty",
  "business_connection_narrative": "...",
  "evidence_analysis": {
    "source_text_snippets": ["..."],
    "support_level_from_source_text": "explicit",
    "support_level_reason": "..."
  },
  "source_narrative": {
    "file_id": "...",
    "target_company": "Acme Corp",
    "source_url": "https://example.com/source",
    "source_type": "company website",
    "source_origin_summary": "...",
    "date_last_updated": "2026-04-01",
    "relevant_country": "DE"
  },
  "overall_assessment": {
    "source_weight": "high",
    "source_weight_reason": "...",
    "source_text_summary": "..."
  },
  "negative_or_cautionary_points": []
}
```

---

## Testing

```bash
pytest tests/
```

| Test file | Coverage |
|---|---|
| `test_flattening_postprocessing.py` | Unicode/Cyrillic alias matching in the flatten stage |
| `test_target_matching_no_network.py` | Target term identification in source text (no network calls) |
| `test_background_update_block_grouping.py` | Block grouping by `canonical_name` in the background update stage |

---

## Project Structure

```text
infoflow/
|-- main.py                           # CLI entry point
|-- workflow.config.json              # Pipeline configuration
|-- requirements.txt
|
|-- sample_data/
|   `-- 00_seed_background.txt        # Example seed background
|
|-- prompts/
|   |-- 01_search_queries_generation_prompt.txt
|   |-- 04_file_level_extractions_prompt.txt
|   |-- 06_relationship_findings_prompt.txt
|   `-- 07_updated_background_report_prompt.txt
|
|-- src/
|   |-- infoflow_pipeline/
|   |   |-- config.py                 # WorkflowConfig dataclass, SecretRef, config loader
|   |   |-- runner.py                 # Stage registry, CLI dispatcher, workflow runner
|   |   |-- generate_query_list.py    # Stage 1
|   |   |-- collect_search_results.py # Stage 2
|   |   |-- download_documents.py     # Stage 3
|   |   |-- extract_file_findings.py  # Stage 4
|   |   |-- flatten_findings.py       # Stage 5
|   |   |-- summarize_relationships.py # Stage 6
|   |   `-- update_background_report.py # Stage 7
|   `-- clientdcf/
|       `-- client.py                 # DataForSEO REST client
|
|-- tests/
|   |-- test_flattening_postprocessing.py
|   |-- test_target_matching_no_network.py
|   `-- test_background_update_block_grouping.py
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `SEO_USER` | Yes | DataForSEO login |
| `SEO_PASS` | Yes | DataForSEO password |
| `FLARE_API_KEY` | No | Cloudflare Browser Rendering API token |
| `FLARE_CLIENT_ID` | No | Cloudflare account ID |
