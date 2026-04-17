from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
import trafilatura

from infoflow_pipeline.config import WorkflowConfig, load_default_config



# =============================
# CONFIG
# =============================

CSV_PATH = Path("master_url_list.csv")
OUTPUT_DIR = Path("serp_downloads")

# CSV columns we manage / add
ID_COL = "download_id"
FILE_COL = "download_file"
FETCHED_AT_COL = "fetched_at"
STATUS_COL = "fetch_status"

# Failure hygiene: track more detail about failures + HTTP metadata
ERROR_COL = "fetch_error"
LAST_ERROR_AT_COL = "last_error_at"
ATTEMPTS_COL = "attempt_count"
HTTP_STATUS_COL = "http_status"
CONTENT_TYPE_COL = "content_type"
FINAL_URL_COL = "final_url"
MECH_COL = "mechanism"
RESULT_STAGE_COL = "result_stage"
CLOUDFLARE_STATUS_COL = "cloudflare_http_status"
CLOUDFLARE_STAGE_COL = "cloudflare_stage"
CLOUDFLARE_ERROR_COL = "cloudflare_error"

# NEW: separate failure stage bucket (requests / trafilatura / playwright / extract / save / etc.)
FAILURE_STAGE_COL = "failure_stage"

# Cloudflare Browser Rendering defaults for HTML pages
CLOUDFLARE_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
CLOUDFLARE_RETRY_JITTER_SEC = (0.0, 1.0)

# Rate limiting + denylist
DENYLIST_DOMAINS = {
    # "example.com",
}
PER_DOMAIN_MIN_INTERVAL_SEC = 1.0
JITTER_RANGE_SEC = (0.05, 0.25)
CSV_DELIMITER = ";"


# =============================
# FETCH + SAVE
# =============================

@dataclass(frozen=True)
class FetchConfig:
    req_timeout: Tuple[float, float] = (10.0, 50.0)
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
    cloudflare_api_token: str = ""
    cloudflare_account_id: str = ""


# In-memory state for per-domain rate limiting
_DOMAIN_LAST_HIT: Dict[str, float] = {}


def configure(config: WorkflowConfig) -> None:
    global CSV_PATH, OUTPUT_DIR
    global DENYLIST_DOMAINS, PER_DOMAIN_MIN_INTERVAL_SEC, JITTER_RANGE_SEC
    global _DOMAIN_LAST_HIT

    CSV_PATH = config.paths.master_url_csv
    OUTPUT_DIR = config.paths.downloads_dir
    DENYLIST_DOMAINS = set(config.fetch.denylist_domains)
    PER_DOMAIN_MIN_INTERVAL_SEC = config.fetch.per_domain_min_interval_sec
    JITTER_RANGE_SEC = config.fetch.jitter_range_sec
    _DOMAIN_LAST_HIT = {}


class StageError(Exception):
    """
    Exception carrying a failure stage label so the caller can persist
    the reason bucket in CSV (e.g., requests_fetch, text_extract, file_save).
    """
    def __init__(self, stage: str, message: str, details: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.stage = stage
        self.details = details or {}


class CloudflareRenderError(RuntimeError):
    """Structured Cloudflare rendering failure with stage + status details."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        status_code: Optional[int] = None,
        body_excerpt: str = "",
    ):
        super().__init__(message)
        self.stage = stage
        self.status_code = status_code
        self.body_excerpt = body_excerpt


def _cloudflare_details_from_exception(exc: Exception) -> Dict[str, str]:
    """Normalize Cloudflare exceptions into CSV-safe diagnostics."""
    if isinstance(exc, CloudflareRenderError):
        message = str(exc)
        if exc.body_excerpt:
            message = f"{message} | body={exc.body_excerpt}"
        return {
            CLOUDFLARE_STAGE_COL: exc.stage,
            CLOUDFLARE_STATUS_COL: str(exc.status_code or ""),
            CLOUDFLARE_ERROR_COL: message,
        }

    return {
        CLOUDFLARE_STAGE_COL: "unknown",
        CLOUDFLARE_STATUS_COL: "",
        CLOUDFLARE_ERROR_COL: f"{type(exc).__name__}: {exc}",
    }


def _normalize_url(url: str) -> str:
    """Trim whitespace and normalize trivial formatting."""
    return (url or "").strip()


def _make_stable_id(url: str, length: int = 16) -> str:
    """Stable unique identifier derived from URL content (SHA-256 truncated)."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:length]


def _extract_main_text(html: str) -> str:
    """
    Extract AI-processable text from HTML.
    Primary path uses trafilatura on fully rendered HTML; fallback is a DOM text scrape.
    """
    extracted = (
        trafilatura.extract(
            html,
            include_tables=True,
            include_links=False,
            include_comments=False,
        )
        or ""
    ).strip()
    if extracted:
        return extracted

    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception:
        return ""


def _domain_of(url: str) -> str:
    """Parse domain/netloc for denylist + per-domain throttling."""
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _rate_limit_per_domain(domain: str) -> None:
    """
    Rate limiting: ensure we do not hit the same domain too frequently.
    Adds jitter so request timing looks less bot-like.
    """
    if not domain:
        time.sleep(random.uniform(*JITTER_RANGE_SEC))
        return

    last = _DOMAIN_LAST_HIT.get(domain, 0.0)
    now = time.time()
    elapsed = now - last
    wait = max(0.0, PER_DOMAIN_MIN_INTERVAL_SEC - elapsed)

    jitter = random.uniform(*JITTER_RANGE_SEC)
    total_sleep = wait + jitter
    if total_sleep > 0:
        time.sleep(total_sleep)

    _DOMAIN_LAST_HIT[domain] = time.time()


def _looks_like_pdf_bytes(content: bytes) -> bool:
    """
    Robust PDF detection even when headers lie:
    PDFs start with the magic bytes: b'%PDF-'
    """
    return content[:5] == b"%PDF-"


def _looks_like_pdf_url(url: str) -> bool:
    """Cheap heuristic: URL ends with .pdf (ignoring querystring fragments)."""
    try:
        path = urlparse(url).path.lower()
        return path.endswith(".pdf")
    except Exception:
        return False


def _content_type_is_pdf(content_type: str) -> bool:
    """Header heuristic: Content-Type indicates a PDF."""
    return "application/pdf" in (content_type or "").lower()


def _looks_like_binary_payload(content: bytes) -> bool:
    """
    Detect obviously non-text payloads so we do not persist compressed/binary
    bytes as fake HTML when the requests fallback is used.
    """
    sample = content[:4096]
    if not sample:
        return False
    if b"\x00" in sample:
        return True

    printable = sum(1 for b in sample if b in (9, 10, 13) or 32 <= b <= 126)
    return (printable / len(sample)) < 0.75


def _looks_like_html_text(text: str) -> bool:
    """Cheap sanity check that decoded text resembles HTML markup."""
    sample = (text or "").lstrip().lower()[:4096]
    if not sample:
        return False
    html_markers = (
        "<!doctype html",
        "<html",
        "<head",
        "<body",
        "<meta",
        "<title",
        "<script",
        "<article",
        "<div",
    )
    return any(marker in sample for marker in html_markers)


def _hex_prefix(content: bytes, length: int = 8) -> str:
    """Compact byte signature for error messages."""
    return " ".join(f"{b:02X}" for b in content[:length])


def _fetch_requests(url: str, cfg: FetchConfig) -> Tuple[bytes, int, str, str, str]:
    """
    Primary retrieval mechanism:
      - requests.get(url) with retries + backoff
      - returns: (content_bytes, http_status, content_type, final_url, last_modified_header)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; serp-fetch/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
    }

    domain = _domain_of(url)
    last_exc: Optional[Exception] = None

    with requests.Session() as s:
        for attempt in range(cfg.req_retries + 1):
            try:
                _rate_limit_per_domain(domain)

                r = s.get(url, headers=headers, timeout=cfg.req_timeout, allow_redirects=True)
                status = int(r.status_code)
                ctype = (r.headers.get("Content-Type") or "").strip()
                final_url = (r.url or "").strip()
                last_mod = (r.headers.get("Last-Modified") or "").strip()

                r.raise_for_status()
                return r.content, status, ctype, final_url, last_mod

            except Exception as e:
                last_exc = e
                if attempt < cfg.req_retries:
                    time.sleep(cfg.backoff * (2 ** attempt) + random.uniform(*JITTER_RANGE_SEC))

    raise last_exc  # type: ignore[misc]


def _required_secret(label: str, value: str) -> str:
    value = (value or "").strip()
    if not value:
        raise CloudflareRenderError("config", f"Missing required credential: {label}")
    return value


def _retry_after_seconds(value: str) -> Optional[float]:
    """Parse Retry-After as seconds or an HTTP date."""
    raw = (value or "").strip()
    if not raw:
        return None

    try:
        return max(0.0, float(raw))
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(raw)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(tz=retry_at.tzinfo)).total_seconds())
    except Exception:
        return None


def _cloudflare_backoff_sleep(response: Optional[requests.Response], attempt: int, cfg: FetchConfig) -> None:
    """Back off only for retryable Cloudflare failures."""
    retry_after = None
    if response is not None:
        retry_after = _retry_after_seconds(response.headers.get("Retry-After", ""))

    if retry_after is None:
        retry_after = min(cfg.cf_backoff_cap, cfg.cf_backoff_base * (2 ** attempt))

    time.sleep(retry_after + random.uniform(*CLOUDFLARE_RETRY_JITTER_SEC))


def _artifacts_for_id(out_dir: Path, stable_id: str) -> Dict[str, Path]:
    return {
        "pdf": out_dir / f"{stable_id}.pdf",
        "txt": out_dir / f"{stable_id}.txt",
        "html": out_dir / f"{stable_id}.html",
    }


def _has_complete_artifact_set(artifacts: Dict[str, Path]) -> bool:
    has_txt = artifacts["txt"].exists()
    has_html = artifacts["html"].exists()
    has_pdf = artifacts["pdf"].exists()
    return has_txt and (has_html or has_pdf)


def _cloudflare_payload(url: str, cfg: FetchConfig) -> Dict[str, Any]:
    """General-purpose render profile for unknown JS-heavy sites."""
    return {
        "url": url,
        "actionTimeout": cfg.cloudflare_action_timeout_ms,
        "bestAttempt": True,
        "setJavaScriptEnabled": True,
        "userAgent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "viewport": {
            "width": 1440,
            "height": 2200,
            "deviceScaleFactor": 1,
            "isMobile": False,
            "hasTouch": False,
            "isLandscape": False,
        },
        "gotoOptions": {
            "timeout": cfg.cloudflare_nav_timeout_ms,
            "waitUntil": ["load", "networkidle2"],
        },
        "waitForTimeout": cfg.cloudflare_extra_wait_ms,
    }


def _fetch_cloudflare_rendered_html(url: str, cfg: FetchConfig) -> Tuple[str, int, str, str, str]:
    """
    Primary HTML fetch path for non-PDF pages.
    Uses Cloudflare Browser Rendering to return fully rendered HTML.
    """
    api_token = _required_secret("cloudflare_api_token", cfg.cloudflare_api_token)
    account_id = _required_secret("cloudflare_account_id", cfg.cloudflare_account_id)

    endpoint = f"{cfg.cloudflare_api_base_url}/accounts/{account_id}/browser-rendering/content"
    last_error: Optional[CloudflareRenderError] = None

    for attempt in range(cfg.cf_retries + 1):
        response: Optional[requests.Response] = None
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json",
                },
                params={"cacheTTL": 0},
                json=_cloudflare_payload(url, cfg),
                timeout=(20, 180),
            )
        except requests.RequestException as exc:
            last_error = CloudflareRenderError("request", f"{type(exc).__name__}: {exc}")
            if attempt < cfg.cf_retries:
                _cloudflare_backoff_sleep(None, attempt, cfg)
                continue
            raise last_error from exc

        try:
            payload = response.json()
        except ValueError as exc:
            last_error = CloudflareRenderError(
                "response_parse",
                "Cloudflare returned non-JSON response",
                status_code=response.status_code,
                body_excerpt=response.text[:500],
            )
            if response.status_code in CLOUDFLARE_RETRYABLE_STATUS_CODES and attempt < cfg.cf_retries:
                _cloudflare_backoff_sleep(response, attempt, cfg)
                continue
            raise last_error from exc

        if response.status_code >= 400 or not payload.get("success", False):
            errors = payload.get("errors") or []
            message = "; ".join(
                err.get("message", "Unknown Cloudflare error")
                for err in errors
                if isinstance(err, dict)
            ) or f"HTTP {response.status_code}"
            last_error = CloudflareRenderError(
                "api_error",
                message,
                status_code=response.status_code,
                body_excerpt=(json.dumps(payload)[:500] if payload else ""),
            )
            if response.status_code in CLOUDFLARE_RETRYABLE_STATUS_CODES and attempt < cfg.cf_retries:
                _cloudflare_backoff_sleep(response, attempt, cfg)
                continue
            raise last_error

        html = payload.get("result")
        if not isinstance(html, str) or not html.strip():
            raise CloudflareRenderError(
                "empty_result",
                "Cloudflare returned an empty rendered HTML result",
                status_code=response.status_code,
                body_excerpt=(json.dumps(payload)[:500] if payload else ""),
            )

        meta = payload.get("meta") or {}
        status = int(meta.get("status") or 200)
        return html, status, "text/html (cloudflare-rendered)", url, ""

    raise last_error or CloudflareRenderError("unknown", "Cloudflare render failed without details")


def _parse_timestamp_candidates_from_html(html: str) -> Dict[str, str]:
    """
    Ported from timestamp.txt:
      - meta tags (article:published_time, article:modified_time, date, publishdate, dc.date)
      - JSON-LD (datePublished/dateModified/dateCreated)
    Falls back gracefully if BeautifulSoup is unavailable.
    """
    dates: Dict[str, str] = {}
    if not html:
        return dates

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")

            for meta in soup.find_all("meta"):
                prop = meta.get("property", "") or meta.get("name", "")
                if prop in (
                    "article:published_time",
                    "article:modified_time",
                    "date",
                    "publishdate",
                    "dc.date",
                ):
                    content = meta.get("content")
                    if content:
                        dates[prop] = str(content)

            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    raw = script.string
                    if not raw:
                        continue
                    ld = json.loads(raw)

                    def _walk(obj):
                        if isinstance(obj, dict):
                            for k in ("datePublished", "dateModified", "dateCreated"):
                                if k in obj and obj[k]:
                                    dates[f"jsonld_{k}"] = str(obj[k])
                            for v in obj.values():
                                _walk(v)
                        elif isinstance(obj, list):
                            for it in obj:
                                _walk(it)

                    _walk(ld)
                except Exception:
                    pass
        except Exception:
            pass
        return dates

    # Minimal fallback: regex scrape
    try:
        meta_patterns = [
            r'(?:property|name)\s*=\s*"(article:published_time|article:modified_time|date|publishdate|dc\.date)"[^>]*content\s*=\s*"([^"]+)"',
            r'(?:content)\s*=\s*"([^"]+)"[^>]*(?:property|name)\s*=\s*"(article:published_time|article:modified_time|date|publishdate|dc\.date)"',
        ]
        for pat in meta_patterns:
            for m in re.finditer(pat, html, flags=re.IGNORECASE):
                key = m.group(1)
                val = m.group(2)
                if key and val:
                    dates[key] = val

        for m in re.finditer(r'"(datePublished|dateModified|dateCreated)"\s*:\s*"([^"]+)"', html):
            dates[f"jsonld_{m.group(1)}"] = m.group(2)
    except Exception:
        pass
    return dates


def _choose_best_timestamp(candidates: Dict[str, str]) -> str:
    """Choose the most useful timestamp (best-effort)."""
    priority = [
        "article:modified_time",
        "jsonld_dateModified",
        "article:published_time",
        "jsonld_datePublished",
        "jsonld_dateCreated",
        "date",
        "publishdate",
        "dc.date",
        "http_last_modified",
    ]
    for k in priority:
        v = (candidates.get(k) or "").strip()
        if v:
            return v

    for k in sorted(candidates.keys()):
        v = (candidates.get(k) or "").strip()
        if v:
            return v
    return ""


def _timestamp_sentence_for_txt(url: str, final_url: str, *, last_modified: str, html: str = "") -> str:
    """Build the first sentence inserted into every TXT output file."""
    candidates: Dict[str, str] = {}
    if last_modified:
        candidates["http_last_modified"] = last_modified
    candidates.update(_parse_timestamp_candidates_from_html(html))
    best = _choose_best_timestamp(candidates)

    chosen = best or "unknown"
    ref = final_url or url
    return f"Source timestamp (uploaded/created/updated): {chosen}. URL: {ref}"


def _extract_pdf_text_no_ocr(pdf_bytes: bytes) -> str:
    """Extract selectable text from a PDF without OCR (pdfminer.six if available)."""
    if not pdf_bytes:
        return ""
    try:
        from io import BytesIO
        try:
            from pdfminer.high_level import extract_text  # type: ignore
        except Exception:
            return ""
        return (extract_text(BytesIO(pdf_bytes)) or "").strip()
    except Exception:
        return "PDF miner does not work"


def fetch_and_save(url: str, out_dir: Path, stable_id: str, cfg: FetchConfig) -> Dict[str, str]:
    """
    Fetch a URL and save it:
      - If it is a PDF: save as <id>.pdf AND also save <id>.txt with extracted selectable text (no OCR)
      - Else: save fully rendered HTML as <id>.html AND AI-ready text as <id>.txt

    HTML sequence:
      - requests for initial fetch + PDF detection
      - Cloudflare Browser Rendering as the primary non-PDF HTML source
      - raw requests HTML only as a fallback if Cloudflare rendering fails

    In all cases, the first line of the TXT output contains an estimated source timestamp.

    Raises StageError with a specific stage label on failure.
    """
    # ---- Attempt 1: requests (bytes) ----
    try:
        content, status, ctype, final_url, last_mod = _fetch_requests(url, cfg)
    except Exception as e:
        requests_error = f"{type(e).__name__}: {e}"
        cloudflare_diag = {
            CLOUDFLARE_STAGE_COL: "",
            CLOUDFLARE_STATUS_COL: "",
            CLOUDFLARE_ERROR_COL: "",
        }
        try:
            best_html, status, ctype, _, _ = _fetch_cloudflare_rendered_html(url, cfg)
            text = _extract_main_text(best_html)
            if not text.strip():
                raise StageError(
                    "text_extract",
                    "Cloudflare rendered HTML succeeded after requests failed, but no AI text was extracted."
                    + f" requests error: {requests_error}",
                    details=cloudflare_diag,
                )

            html_path = out_dir / f"{stable_id}.html"
            txt_path = out_dir / f"{stable_id}.txt"
            ts_sentence = _timestamp_sentence_for_txt(url, url, last_modified="", html=best_html)
            html_path.write_text(best_html, encoding="utf-8")
            txt_path.write_text(f"{ts_sentence}\n\n{text}".strip() + "\n", encoding="utf-8")
            return {
                STATUS_COL: "downloaded_html_and_text",
                FILE_COL: str(txt_path),
                HTTP_STATUS_COL: str(status),
                CONTENT_TYPE_COL: ctype,
                FINAL_URL_COL: url,
                MECH_COL: "cloudflare_direct",
                RESULT_STAGE_COL: "cloudflare_direct_after_requests_failure",
                CLOUDFLARE_STAGE_COL: "",
                CLOUDFLARE_STATUS_COL: "",
                CLOUDFLARE_ERROR_COL: "",
            }
        except StageError:
            raise
        except Exception as cf_exc:
            cloudflare_diag = _cloudflare_details_from_exception(cf_exc)
            raise StageError("requests_fetch", requests_error, details=cloudflare_diag) from e

    # ---- Decide whether it's a PDF (robust detection) ----
    try:
        is_pdf = (
            _content_type_is_pdf(ctype)
            or _looks_like_pdf_url(final_url or url)
            or _looks_like_pdf_bytes(content)
        )
    except Exception as e:
        raise StageError("pdf_detect", f"{type(e).__name__}: {e}") from e

    if is_pdf:
        # Save binary PDF as-is
        pdf_path = out_dir / f"{stable_id}.pdf"
        try:
            pdf_path.write_bytes(content)
        except Exception as e:
            raise StageError("file_save_pdf", f"{type(e).__name__}: {e}") from e

        # Also save a TXT sibling with selectable text (no OCR)
        try:
            ts_sentence = _timestamp_sentence_for_txt(url, final_url, last_modified=last_mod, html="")
            pdf_text = _extract_pdf_text_no_ocr(content)
            txt_path = out_dir / f"{stable_id}.txt"
            txt_path.write_text(f"{ts_sentence}\n\n{pdf_text}".strip() + "\n", encoding="utf-8")
        except Exception as e:
            raise StageError("file_save_pdf_txt", f"{type(e).__name__}: {e}") from e

        return {
            STATUS_COL: "downloaded_pdf_and_text",
            FILE_COL: str(txt_path),
            HTTP_STATUS_COL: str(status),
            CONTENT_TYPE_COL: ctype,
            FINAL_URL_COL: final_url,
            MECH_COL: "requests",
            RESULT_STAGE_COL: "pdf_from_requests",
        }

    # ---- Otherwise treat as HTML-like ----
    try:
        raw_html = content.decode("utf-8", errors="ignore")
    except Exception as e:
        raise StageError("text_extract_requests", f"{type(e).__name__}: {e}") from e

    raw_html_is_valid = (
        not _looks_like_binary_payload(content)
        and _looks_like_html_text(raw_html)
    )

    best_html = raw_html
    mechanism = "requests_html_fallback"
    cloudflare_diag = {
        CLOUDFLARE_STAGE_COL: "",
        CLOUDFLARE_STATUS_COL: "",
        CLOUDFLARE_ERROR_COL: "",
    }

    # ---- Primary HTML fetch: Cloudflare Browser Rendering ----
    try:
        rendered_html, rendered_status, rendered_ctype, _, _ = _fetch_cloudflare_rendered_html(final_url or url, cfg)
        best_html = rendered_html
        status = rendered_status
        ctype = rendered_ctype
        mechanism = "cloudflare"
    except Exception as e:
        cloudflare_diag = _cloudflare_details_from_exception(e)

    if mechanism == "requests_html_fallback" and not raw_html_is_valid:
        cf_message = cloudflare_diag.get(CLOUDFLARE_ERROR_COL, "")
        raise StageError(
            "requests_fallback_invalid_html",
            "requests fallback returned non-HTML/binary content; refusing to save unreadable HTML/text."
            + f" content_type={ctype or 'unknown'}"
            + (
                f" byte_prefix={_hex_prefix(content)}"
                if content
                else ""
            )
            + (
                f" Cloudflare error: {cf_message}"
                if cf_message
                else ""
            ),
            details=cloudflare_diag,
        )

    # ---- Extract AI-processable text from the chosen HTML ----
    try:
        text = _extract_main_text(best_html)
        if not text.strip() and best_html != raw_html and raw_html_is_valid:
            fallback_text = _extract_main_text(raw_html)
            if fallback_text.strip():
                text = fallback_text
                best_html = raw_html
                mechanism = "requests_html_fallback"
    except Exception as e:
        raise StageError("text_extract", f"{type(e).__name__}: {e}") from e

    if not text.strip():
        raise StageError(
            "text_extract",
            "No extractable AI text found from rendered or fallback HTML."
            + (
                f" Cloudflare error: {cloudflare_diag.get(CLOUDFLARE_ERROR_COL, '')}"
                if cloudflare_diag.get(CLOUDFLARE_ERROR_COL, "")
                else ""
            ),
            details=cloudflare_diag,
        )

    html_path = out_dir / f"{stable_id}.html"
    txt_path = out_dir / f"{stable_id}.txt"
    try:
        ts_sentence = _timestamp_sentence_for_txt(url, final_url, last_modified=last_mod, html=best_html)
        html_path.write_text(best_html, encoding="utf-8")
        txt_path.write_text(f"{ts_sentence}\n\n{text}".strip() + "\n", encoding="utf-8")
    except Exception as e:
        raise StageError("file_save_html_or_txt", f"{type(e).__name__}: {e}") from e

    return {
        STATUS_COL: "downloaded_html_and_text",
        FILE_COL: str(txt_path),
        HTTP_STATUS_COL: str(status),
        CONTENT_TYPE_COL: ctype,
        FINAL_URL_COL: final_url,
        MECH_COL: mechanism,
        RESULT_STAGE_COL: (
            "html_from_cloudflare" if mechanism == "cloudflare" else "html_from_requests_fallback"
        ),
        CLOUDFLARE_STAGE_COL: cloudflare_diag.get(CLOUDFLARE_STAGE_COL, ""),
        CLOUDFLARE_STATUS_COL: cloudflare_diag.get(CLOUDFLARE_STATUS_COL, ""),
        CLOUDFLARE_ERROR_COL: cloudflare_diag.get(CLOUDFLARE_ERROR_COL, ""),
    }

    # Legacy markdown path removed.


# =============================
# CSV IO + UPDATE LOGIC
# =============================

def _detect_url_column(fieldnames: List[str]) -> str:
    """Detect which CSV column contains URLs."""
    if "url" in fieldnames:
        return "url"
    if "URL" in fieldnames:
        return "URL"

    lowered = {name.lower(): name for name in fieldnames}
    if "url" in lowered:
        return lowered["url"]

    for name in fieldnames:
        if "url" in name.lower():
            return name

    raise KeyError(
        f"Could not detect URL column in CSV headers: {fieldnames}. "
        "Rename the URL column to 'url' (recommended) or include 'url' in its name."
    )


def _ensure_columns(fieldnames: List[str], required: Iterable[str]) -> List[str]:
    """Ensure output CSV includes required columns; preserve existing order and append missing ones."""
    out = list(fieldnames)
    for c in required:
        if c not in out:
            out.append(c)
    return out


def _count_nonempty_urls(rows: List[Dict[str, str]], url_col: str) -> int:
    """Count rows that actually contain a URL, for progress display denominator."""
    return sum(1 for row in rows if _normalize_url(row.get(url_col, "")))


def update_csv_and_download(
    csv_path: Path,
    out_dir: Path,
    *,
    cfg: FetchConfig = FetchConfig(),
    dry_run: bool = False,
) -> None:
    """
    Main routine:
      - Load CSV
      - For each URL row, decide skip vs download
      - Save PDF as <id>.pdf plus <id>.txt
      - Save rendered HTML as <id>.html plus AI text as <id>.txt
      - Write updated CSV back (with backup)
      - Show progress on screen during execution
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Read input CSV ----
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=CSV_DELIMITER)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row (fieldnames missing).")

        fieldnames = list(reader.fieldnames)
        url_col = _detect_url_column(fieldnames)

        required_cols = [
            ID_COL, FILE_COL, FETCHED_AT_COL, STATUS_COL,
            ERROR_COL, LAST_ERROR_AT_COL, ATTEMPTS_COL,
            HTTP_STATUS_COL, CONTENT_TYPE_COL, FINAL_URL_COL, MECH_COL, RESULT_STAGE_COL,
            CLOUDFLARE_STATUS_COL, CLOUDFLARE_STAGE_COL, CLOUDFLARE_ERROR_COL,
            FAILURE_STAGE_COL,  # NEW
        ]
        out_fieldnames = _ensure_columns(fieldnames, required_cols)

        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})

    # ---- Backup original CSV before overwriting ----
    backup_path = csv_path.with_suffix(f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    if not dry_run:
        backup_path.write_text(csv_path.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")

    total = skipped = downloaded = failed = denied = 0

    # Count actual URL rows so progress is meaningful (e.g. 4 out of 65)
    progress_total = _count_nonempty_urls(rows, url_col)
    progress_i = 0

    for row in rows:
        url = _normalize_url(row.get(url_col, ""))
        if not url:
            continue

        total += 1
        progress_i += 1

        # ---- Progress display (a) ----
        print(f"[{progress_i}/{progress_total}] Processing: {url}", flush=True)

        # Default: clear failure stage unless a failure occurs
        row[FAILURE_STAGE_COL] = ""
        row[RESULT_STAGE_COL] = ""
        row[CLOUDFLARE_STATUS_COL] = ""
        row[CLOUDFLARE_STAGE_COL] = ""
        row[CLOUDFLARE_ERROR_COL] = ""

        # ---- Denylist check ----
        domain = _domain_of(url)
        if domain and domain in DENYLIST_DOMAINS:
            row[STATUS_COL] = "denied_domain"
            row[ERROR_COL] = f"Domain denylisted: {domain}"
            row[LAST_ERROR_AT_COL] = datetime.now().isoformat(timespec="seconds")
            row[FAILURE_STAGE_COL] = "precheck_denylist"
            row[RESULT_STAGE_COL] = ""
            row[CLOUDFLARE_STATUS_COL] = ""
            row[CLOUDFLARE_STAGE_COL] = ""
            row[CLOUDFLARE_ERROR_COL] = ""
            denied += 1
            print(f"    -> denied_domain (stage={row[FAILURE_STAGE_COL]})", flush=True)
            continue

        # ---- Skip logic: if we already have an ID and the saved file exists, do nothing ----
        existing_id = (row.get(ID_COL, "") or "").strip()
        if existing_id:
            artifacts = _artifacts_for_id(out_dir, existing_id)
            if _has_complete_artifact_set(artifacts):
                row[FILE_COL] = str(artifacts["txt"])
                row[STATUS_COL] = "skipped_exists"
                row[ERROR_COL] = ""
                row[LAST_ERROR_AT_COL] = ""
                row[FAILURE_STAGE_COL] = ""
                row[RESULT_STAGE_COL] = "skip_existing_file"
                row[CLOUDFLARE_STATUS_COL] = ""
                row[CLOUDFLARE_STAGE_COL] = ""
                row[CLOUDFLARE_ERROR_COL] = ""
                skipped += 1
                print(f"    -> skipped_exists", flush=True)
                continue
            if any(path.exists() for path in artifacts.values()):
                print("    -> incomplete_artifacts_found; re-fetching", flush=True)

        # ---- Compute stable ID from URL (used for all saved artifacts) ----
        stable_id = existing_id or _make_stable_id(url)

        # If ID missing but file exists, attach and skip
        artifacts = _artifacts_for_id(out_dir, stable_id)
        if not existing_id and _has_complete_artifact_set(artifacts):
            row[ID_COL] = stable_id
            row[FILE_COL] = str(artifacts["txt"])
            row[STATUS_COL] = "skipped_file_present"
            row[ERROR_COL] = ""
            row[LAST_ERROR_AT_COL] = ""
            row[FAILURE_STAGE_COL] = ""
            row[RESULT_STAGE_COL] = "skip_attach_existing_file"
            row[CLOUDFLARE_STATUS_COL] = ""
            row[CLOUDFLARE_STAGE_COL] = ""
            row[CLOUDFLARE_ERROR_COL] = ""
            skipped += 1
            print(f"    -> skipped_file_present", flush=True)
            continue
        if not existing_id and any(path.exists() for path in artifacts.values()):
            print("    -> incomplete_artifacts_found; re-fetching", flush=True)

        # ---- Attempt tracking ----
        try:
            prev_attempts = int((row.get(ATTEMPTS_COL, "") or "").strip() or "0")
        except ValueError:
            prev_attempts = 0
        row[ATTEMPTS_COL] = str(prev_attempts + 1)

        # ---- Download ----
        try:
            if dry_run:
                row[ID_COL] = stable_id
                row[FILE_COL] = ""
                row[FETCHED_AT_COL] = datetime.now().isoformat(timespec="seconds")
                row[STATUS_COL] = "dry_run"
                row[ERROR_COL] = ""
                row[LAST_ERROR_AT_COL] = ""
                row[FAILURE_STAGE_COL] = ""
                row[RESULT_STAGE_COL] = "dry_run"
                row[CLOUDFLARE_STATUS_COL] = ""
                row[CLOUDFLARE_STAGE_COL] = ""
                row[CLOUDFLARE_ERROR_COL] = ""
                downloaded += 1
                print("    -> dry_run", flush=True)
                continue

            meta = fetch_and_save(url, out_dir, stable_id, cfg)

            # Update CSV fields based on what was saved
            row[ID_COL] = stable_id
            row[FILE_COL] = meta.get(FILE_COL, "")
            row[FETCHED_AT_COL] = datetime.now().isoformat(timespec="seconds")
            row[STATUS_COL] = meta.get(STATUS_COL, "downloaded")
            row[ERROR_COL] = ""
            row[LAST_ERROR_AT_COL] = ""
            row[HTTP_STATUS_COL] = meta.get(HTTP_STATUS_COL, "")
            row[CONTENT_TYPE_COL] = meta.get(CONTENT_TYPE_COL, "")
            row[FINAL_URL_COL] = meta.get(FINAL_URL_COL, "")
            row[MECH_COL] = meta.get(MECH_COL, "")
            row[RESULT_STAGE_COL] = meta.get(RESULT_STAGE_COL, "")
            row[CLOUDFLARE_STATUS_COL] = meta.get(CLOUDFLARE_STATUS_COL, "")
            row[CLOUDFLARE_STAGE_COL] = meta.get(CLOUDFLARE_STAGE_COL, "")
            row[CLOUDFLARE_ERROR_COL] = meta.get(CLOUDFLARE_ERROR_COL, "")
            row[FAILURE_STAGE_COL] = ""

            downloaded += 1
            print(f"    -> {row[STATUS_COL]} ({row.get(MECH_COL, '')})", flush=True)

        except StageError as e:
            row[STATUS_COL] = "failed"
            row[ERROR_COL] = str(e)
            row[LAST_ERROR_AT_COL] = datetime.now().isoformat(timespec="seconds")
            row[FAILURE_STAGE_COL] = e.stage
            row[RESULT_STAGE_COL] = ""
            row[CLOUDFLARE_STATUS_COL] = e.details.get(CLOUDFLARE_STATUS_COL, "")
            row[CLOUDFLARE_STAGE_COL] = e.details.get(CLOUDFLARE_STAGE_COL, "")
            row[CLOUDFLARE_ERROR_COL] = e.details.get(CLOUDFLARE_ERROR_COL, "")
            failed += 1
            print(f"    -> failed (stage={e.stage}) {e}", flush=True)

        except Exception as e:
            # Safety net: unexpected failures still get a stage bucket
            row[STATUS_COL] = "failed"
            row[ERROR_COL] = f"{type(e).__name__}: {e}"
            row[LAST_ERROR_AT_COL] = datetime.now().isoformat(timespec="seconds")
            row[FAILURE_STAGE_COL] = "unknown"
            row[RESULT_STAGE_COL] = ""
            row[CLOUDFLARE_STATUS_COL] = ""
            row[CLOUDFLARE_STAGE_COL] = ""
            row[CLOUDFLARE_ERROR_COL] = ""
            failed += 1
            print(f"    -> failed (stage=unknown) {type(e).__name__}: {e}", flush=True)

    # ---- Write updated CSV ----
    if not dry_run:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=out_fieldnames,
                delimiter=CSV_DELIMITER,
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in out_fieldnames})

    print(
        "Done.\n"
        f"  total_urls={total}\n"
        f"  skipped={skipped}\n"
        f"  denied={denied}\n"
        f"  downloaded={downloaded}\n"
        f"  failed={failed}\n"
        f"CSV:    {csv_path}\n"
        f"Output: {out_dir}\n"
        f"Backup: {backup_path if not dry_run else ' (dry_run: none)'}"
    )


def main(config: WorkflowConfig | None = None) -> None:
    if config is None:
        config = load_default_config()
    configure(config)
    update_csv_and_download(
        csv_path=CSV_PATH,
        out_dir=OUTPUT_DIR,
        cfg=FetchConfig(
            req_timeout=config.fetch.req_timeout,
            req_retries=config.fetch.req_retries,
            trafi_retries=config.fetch.trafi_retries,
            backoff=config.fetch.backoff,
            cf_retries=config.fetch.cf_retries,
            cf_backoff_base=config.fetch.cf_backoff_base,
            cf_backoff_cap=config.fetch.cf_backoff_cap,
            cloudflare_api_base_url=config.fetch.cloudflare_api_base_url,
            cloudflare_nav_timeout_ms=config.fetch.cloudflare_nav_timeout_ms,
            cloudflare_action_timeout_ms=config.fetch.cloudflare_action_timeout_ms,
            cloudflare_extra_wait_ms=config.fetch.cloudflare_extra_wait_ms,
            cloudflare_api_token=config.providers.cloudflare_api_token.resolve("Cloudflare API token"),
            cloudflare_account_id=config.providers.cloudflare_account_id.resolve("Cloudflare account ID"),
        ),
        dry_run=False,
    )


if __name__ == "__main__":
    main()
