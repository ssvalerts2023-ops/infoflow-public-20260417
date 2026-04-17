# Infoflow and Karpathy's LLM Wiki Pattern

## The core idea

[Karpathy's LLM Wiki publication](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) describes a pattern where an LLM maintains a persistent, structured knowledge base rather than re-deriving answers from raw documents on each query. Every new source adds to and refines the wiki, so knowledge compounds. 

Infoflow adopts this pattern and extends it in one direction: automated, target-oriented research of worldwide trading partners of a particular entity / individual.

Infoflow is an AI-powered OSINT pipeline for discovering and analyzing business relationships around a target entity. Given a seed background file describing a company or individual, Infoflow generates search queries, collects search-engine results, downloads source documents, extracts entity mentions, canonicalizes findings, and integrates everything into a narrative intelligence report, which can be seeded again into the workflow from step one to obtain further, more detailed info.


## Three-layer architecture

Karpathy's publication defines three layers, and Infoflow maps onto them directly:

- **Raw sources (immutable):** `03_source_documents/` holds downloaded web pages that the pipeline reads but never edits.
- **Wiki (LLM-maintained):** `07_updated_background_report` is the rolling synthesis â€” rewritten and diffed on each run.
- **Schema (operating instructions):** `workflow.config.json` plus the `prompts/` directory encode model choices and stage-by-stage instructions, turning the LLM into a disciplined analyst rather than an ad-hoc chatbot.

## The ingest flow

The publication's canonical ingest is: read a source, extract key information, update entity pages, note contradictions. Infoflow's six-stage pipeline implements this step by step:

1. Download documents.
2. Extract entity-level findings (`04_file_level_extractions`).
3. Classify and flatten those findings for possible human review (`05_relevant_entity_rows`).
4. Canonicalize relationships (`06_relationship_findings`).
5. Integrate everything into the final background report and emit a change diff (`07_updated_background_report.diff`).

The diff step is the direct embodiment of the article's emphasis on keeping synthesis current and making changes explicit.

## Provenance and logging

The article recommends an append-only log and a navigable index. Infoflow provides equivalent transparency through a CSV provenance chain: `02_query_url_provenance.csv` links every URL back to the search query that produced it, and the numbered outputs (`01_` through `07_`) form a readable audit trail of each run.

## The main extension: automated sourcing

This is where Infoflow most clearly develops the publication's ideas. The article assumes a human curates which sources to add; the LLM then processes them. Infoflow automates sourcing end-to-end:

1. Generate search queries from the existing background document.
2. Query a SERP API.
3. Download the resulting pages â€” all before human review.

The human's role shifts upstream: set the target profile and review the final diff, rather than hand-pick documents.

## What stays the same

The underlying logic is unchanged from the publication: knowledge should accumulate rather than be re-derived; the human directs and curates; the LLM handles the bookkeeping humans tend to abandon. Infoflow applies this to a specific use case â€” competitive intelligence on a named entity â€” and operationalizes the sourcing side that the publication deliberately left abstract.
