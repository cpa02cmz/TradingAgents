"""IDX disclosure + fundamentals fetcher (Keterbukaan Informasi).

Additive IDX equivalent of the US SEC EDGAR stack, active only when
``idx_mode`` is on. ``IDXDisclosureClient.fetch_announcements`` reads the
idx.co.id disclosure endpoints; ``parse_fundamentals_pdf`` is the entry
point for PDF/XLS financial-statement parsing.

Every public entry point degrades gracefully (empty list / unavailable
marker, never raises). Lazy imports only: no PDF/XLS/OCR dependency is
imported at module load, so missing optional deps never break import.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"

# IDX disclosure endpoints tried in order (``{q}`` = URL-quoted symbol).
_DISCLOSURE_URLS = (
    "https://www.idx.co.id/en-us/news-and-events/disclosure?q={q}",
    "https://www.idx.co.id/api/disclosure?symbol={q}",
)

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _base_symbol(ticker: str) -> str:
    """Strip an exchange suffix (``BBCA.JK`` -> ``BBCA``) for IDX lookup."""
    return (ticker or "").strip().upper().split(".")[0]


class IDXDisclosureClient:
    """Fetch IDX announcements (Keterbukaan Informasi). Never raises."""

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def fetch_announcements(
        self,
        symbol: str,
        limit: int = 10,
        timeout: float | None = None,
    ) -> list[dict]:
        """Return ``[{date, title, category, link}]`` for ``symbol``.

        Hits the idx.co.id disclosure endpoints. Any failure logs a warning
        and returns an empty list — never raises.
        """
        import json
        import urllib.parse
        import urllib.request

        code = _base_symbol(symbol)
        if not code:
            logger.warning("IDX disclosure fetch skipped: empty symbol")
            return []
        wait = self.timeout if timeout is None else timeout
        query = urllib.parse.quote(code)
        for template in _DISCLOSURE_URLS:
            url = template.format(q=query)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": _UA})
                with urllib.request.urlopen(request, timeout=wait) as response:
                    raw = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    logger.warning("IDX disclosure response exceeded cap: %s", url)
                    continue
                try:
                    payload = json.loads(raw)
                except ValueError:
                    # HTML portal page (JS-driven): nothing machine-readable
                    # here; the next endpoint may still answer.
                    logger.warning("IDX disclosure endpoint returned non-JSON: %s", url)
                    continue
                items = payload if isinstance(payload, list) else payload.get("data", [])
                announcements = []
                for entry in items[:limit]:
                    if not isinstance(entry, dict):
                        continue
                    announcements.append(
                        {
                            "date": str(entry.get("date") or entry.get("datetime") or "?"),
                            "title": str(entry.get("title") or entry.get("subject") or ""),
                            "category": str(entry.get("category") or entry.get("type") or ""),
                            "link": str(entry.get("link") or entry.get("url") or ""),
                        }
                    )
                return announcements
            except Exception as exc:  # noqa: BLE001 — graceful degradation by contract
                logger.warning(
                    "IDX disclosure fetch failed for %s via %s: %s", code, url, exc
                )
                continue
        return []


def fetch_idx_announcements(symbol: str, limit: int = 10) -> str:
    """Announcements for ``symbol`` as a prompt-ready block. Never raises."""
    try:
        items = IDXDisclosureClient().fetch_announcements(symbol, limit=limit)
    except Exception as exc:  # noqa: BLE001 — belt and braces: never raise
        logger.warning("IDX disclosure fetch failed for %s: %s", symbol, exc)
        items = []
    code = _base_symbol(symbol)
    if not items:
        return (
            f"<IDX disclosure unavailable: no Keterbukaan Informasi announcements "
            f"for {code} could be retrieved from idx.co.id; this is not an "
            f"absence of filings>"
        )
    lines = [f"Keterbukaan Informasi IDX untuk {code} — {len(items)} pengumuman:"]
    for item in items:
        lines.append(
            f"  [{item.get('date', '?')} · {item.get('category', '?')}] "
            f"{item.get('title', '')}"
            + (f" ({item['link']})" if item.get("link") else "")
        )
    return "\n".join(lines)


def parse_fundamentals_pdf(path: str, symbol: str = "") -> dict:
    """Parse an IDX financial-statement PDF/XLS file into a fundamentals dict.

    Entry point for quarterly/annual statement ingestion. Tries, in order:
    text-layer PDF extraction, spreadsheet parsing, then an OCR fallback.

    No heavy dependency is imported at module load — every reader is lazily
    imported here, and a missing one yields an ``unavailable`` dict rather
    than an ImportError. Never raises.
    """
    import os

    try:
        if not path or not os.path.exists(path):
            return {
                "status": "unavailable",
                "symbol": _base_symbol(symbol),
                "reason": f"file not found: {path}",
            }
        suffix = os.path.splitext(path)[1].lower()
        if suffix in (".xls", ".xlsx", ".csv"):
            return _parse_spreadsheet(path, symbol)
        return _parse_pdf(path, symbol)
    except Exception as exc:  # noqa: BLE001 — never raise out of a parser
        logger.warning("IDX fundamentals parse failed for %s: %s", path, exc)
        return {"status": "unavailable", "symbol": _base_symbol(symbol), "reason": str(exc)}


def _parse_pdf(path: str, symbol: str) -> dict:
    """Text-layer PDF parse; OCR fallback is a clearly marked TODO stub."""
    for module_name, reader in (
        ("pypdf", "pypdf"),
        ("pdfplumber", "pdfplumber"),
        ("pdfminer", "pdfminer.high_level"),
    ):
        try:
            __import__(module_name)
            break
        except ImportError:
            continue
    else:
        return {
            "status": "unavailable",
            "symbol": _base_symbol(symbol),
            "reason": "no PDF reader installed (pypdf/pdfplumber/pdfminer); "
            "install one to enable IDX PDF parsing",
        }
    try:
        if reader == "pypdf":
            from pypdf import PdfReader

            text = "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
        elif reader == "pdfplumber":
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        else:
            from pdfminer.high_level import extract_text

            text = extract_text(path) or ""
    except Exception as exc:  # noqa: BLE001 — a corrupt PDF is unavailable, not fatal
        logger.warning("IDX PDF text extraction failed for %s: %s", path, exc)
        text = ""
    if not text.strip():
        # TODO(ocr-fallback): run an OCR pass (e.g. tesseract via ocrmypdf or
        # pytesseract) on scanned IDX filings that have no text layer, then
        # re-enter _parse_pdf on the OCR'd output. Requires optional system +
        # Python OCR deps; must stay lazy and must never raise when absent.
        return {
            "status": "unavailable",
            "symbol": _base_symbol(symbol),
            "reason": "PDF has no extractable text layer; OCR fallback not yet "
            "implemented (TODO)",
        }
    return {
        "status": "ok",
        "symbol": _base_symbol(symbol),
        "source": path,
        "text_excerpt": text[:4000],
        "note": "raw text excerpt; quarterly-lag and sparse-data tolerant "
        "downstream prompting applies",
    }


def _parse_spreadsheet(path: str, symbol: str) -> dict:
    """Spreadsheet (XLS/XLSX/CSV) parse with lazy optional imports."""
    import os

    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        try:
            import csv

            with open(path, encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))[:50]
            return {
                "status": "ok",
                "symbol": _base_symbol(symbol),
                "source": path,
                "rows": rows,
            }
        except Exception as exc:  # noqa: BLE001 — never raise out of a parser
            logger.warning("IDX CSV parse failed for %s: %s", path, exc)
            return {"status": "unavailable", "symbol": _base_symbol(symbol), "reason": str(exc)}
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return {
            "status": "unavailable",
            "symbol": _base_symbol(symbol),
            "reason": "openpyxl not installed; install it to enable IDX XLS parsing",
        }
    try:
        from openpyxl import load_workbook

        workbook = load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.active
        rows = [[cell.value for cell in row[:12]] for row in sheet.iter_rows(max_row=50)]
        return {
            "status": "ok",
            "symbol": _base_symbol(symbol),
            "source": path,
            "sheet": sheet.title,
            "rows": rows,
        }
    except Exception as exc:  # noqa: BLE001 — never raise out of a parser
        logger.warning("IDX XLS parse failed for %s: %s", path, exc)
        return {"status": "unavailable", "symbol": _base_symbol(symbol), "reason": str(exc)}
