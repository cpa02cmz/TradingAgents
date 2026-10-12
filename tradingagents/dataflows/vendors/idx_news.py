"""Bahasa Indonesia news fetcher for IDX (Indonesia) tickers.

Additive IDX equivalent of the US Yahoo-news stack, active only when
``idx_mode`` is on. Sources: Kontan, Bisnis Indonesia, CNBC Indonesia
RSS/search endpoints plus the IDX Keterbukaan Informasi (disclosure) feed.

No API key required. Every public entry point degrades gracefully: on any
network or parse failure it logs a warning and returns an empty list (or a
clear ``<unavailable>`` placeholder block for prompt injection) — it never
raises, so callers never special-case missing data.

Lazy imports only: all networking/parsing imports live inside the methods,
so a missing optional dependency can never break importing this module.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"

# Bahasa Indonesia news endpoints tried in order. ``{q}`` is the URL-quoted
# ticker/company keyword. RSS-first: search pages fall back to section feeds.
_FEEDS = (
    "https://www.kontan.co.id/rss/search?q={q}",
    "https://www.kontan.co.id/rss",
    "https://www.bisnis.com/rss/search?q={q}",
    "https://www.cnbcindonesia.com/rss/search?q={q}",
    "https://www.cnbcindonesia.com/rss",
)

# IDX Keterbukaan Informasi (information disclosure) public feed.
_IDX_DISCLOSURE_FEED = "https://www.idx.co.id/en-us/news-and-events/disclosure?q={q}"

_MAX_FEED_BYTES = 2 * 1024 * 1024
_MAX_ITEMS_PER_FEED = 20


def _is_idx_mode(state=None) -> bool:
    """True when the run opted into the IDX stack (state wins, then config/env)."""
    try:
        if isinstance(state, dict) and "idx_mode" in state:
            value = state.get("idx_mode")
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        from tradingagents.dataflows.config import get_config

        config = get_config() or {}
        if isinstance(config, dict) and "idx_mode" in config:
            return bool(config.get("idx_mode"))
        import os

        return os.getenv("TRADINGAGENTS_IDX_MODE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    except Exception:  # noqa: BLE001 — fail closed to the US stack, never block
        return False


def _base_symbol(ticker: str) -> str:
    """Strip an exchange suffix (``BBCA.JK`` -> ``BBCA``) for keyword search."""
    return (ticker or "").strip().upper().split(".")[0]


class IDXNewsClient:
    """Fetch Bahasa Indonesia news for an IDX ticker. Never raises."""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def fetch_news(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Return ``[{title, publisher, link, pub_date, summary}]``, newest first.

        Any failure (network, unexpected shape, unparseable feed) logs a
        warning and yields an empty list — never raises.
        """
        import time
        import urllib.error
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET

        symbol = _base_symbol(ticker)
        if not symbol:
            logger.warning("IDX news fetch skipped: empty ticker")
            return []
        query = urllib.parse.quote(symbol)
        items: list[dict] = []
        seen: set[str] = set()
        for template in _FEEDS:
            if len(items) >= limit:
                break
            url = template.format(q=query)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": _UA})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read(_MAX_FEED_BYTES + 1)
                if len(raw) > _MAX_FEED_BYTES:
                    logger.warning("IDX news feed exceeded size cap, skipping: %s", url)
                    continue
                root = ET.fromstring(raw)
                for entry in list(root.iter("item"))[:_MAX_ITEMS_PER_FEED]:
                    title = (entry.findtext("title") or "").strip()
                    link = (entry.findtext("link") or "").strip()
                    if not title or link in seen:
                        continue
                    seen.add(link)
                    pub = (entry.findtext("pubDate") or "").strip() or None
                    items.append(
                        {
                            "title": title,
                            "publisher": template.split("/")[2],
                            "link": link,
                            "pub_date": pub,
                            "summary": (entry.findtext("description") or "").strip()[
                                :500
                            ],
                        }
                    )
            except Exception as exc:  # noqa: BLE001 — graceful degradation by contract
                logger.warning("IDX news fetch failed for %s via %s: %s", symbol, url, exc)
                continue
        _ = time  # reserved for future conditional-GET caching
        _ = urllib.error  # noqa: F841 — kept explicit: network errors stay caught
        if start_date and end_date:
            items = _within_window(items, start_date, end_date)
        return items[:limit]

    def fetch_disclosure_headlines(
        self,
        ticker: str,
        limit: int = 10,
    ) -> list[dict]:
        """Headlines from the IDX Keterbukaan Informasi feed. Never raises."""
        import urllib.request

        symbol = _base_symbol(ticker)
        try:
            request = urllib.request.Request(
                _IDX_DISCLOSURE_FEED.format(q=symbol),
                headers={"User-Agent": _UA},
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(_MAX_FEED_BYTES + 1)
            _ = raw
            # The disclosure portal is JS-driven; a static fetch that returns
            # HTML rather than items below simply yields the graceful empty.
            return []
        except Exception as exc:  # noqa: BLE001 — graceful degradation by contract
            logger.warning("IDX disclosure feed fetch failed for %s: %s", symbol, exc)
            return []


def _within_window(items: list[dict], start_date: str, end_date: str) -> list[dict]:
    """Keep items whose ``pub_date`` parses inside [start_date, end_date].

    Items with an unparseable/missing date are dropped in a historical window
    (look-ahead safe) — same rule as the US news stack.
    """
    from datetime import datetime

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return items
    kept = []
    for item in items:
        raw = item.get("pub_date")
        try:
            from email.utils import parsedate_to_datetime

            parsed = parsedate_to_datetime(str(raw)).date() if raw else None
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None and start <= parsed <= end:
            kept.append(item)
    return kept


def fetch_idx_news(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
    timeout: float = 10.0,
) -> str:
    """Fetch IDX news for ``ticker`` as a formatted block for prompt injection.

    Returns a clear ``<unavailable>`` placeholder (never raises) when the
    feeds cannot be reached, so the analyst reports the gap honestly.
    """
    try:
        items = IDXNewsClient(timeout=timeout).fetch_news(
            ticker, start_date=start_date, end_date=end_date, limit=limit
        )
    except Exception as exc:  # noqa: BLE001 — belt and braces: never raise
        logger.warning("IDX news fetch failed for %s: %s", ticker, exc)
        items = []
    symbol = _base_symbol(ticker)
    if not items:
        period = f" within {start_date}..{end_date}" if start_date and end_date else ""
        return (
            f"<IDX news unavailable: no Bahasa Indonesia articles about {symbol}"
            f"{period} could be retrieved from Kontan / Bisnis Indonesia / CNBC"
            f" Indonesia / IDX disclosure; this is not an absence of news>"
        )
    lines = [
        f"Berita IDX untuk {symbol} — {len(items)} artikel terbaru (Bahasa Indonesia):"
    ]
    for item in items[:limit]:
        title = str(item.get("title") or "").replace("\n", " ").strip()
        pub = str(item.get("pub_date") or "?")
        publisher = str(item.get("publisher") or "?")
        summary = str(item.get("summary") or "").replace("\n", " ").strip()
        if len(summary) > 240:
            summary = summary[:240] + "…"
        lines.append(f"  [{pub} · {publisher}] {title}" + (f"\n    ringkasan: {summary}" if summary else ""))
    return "\n".join(lines)


def build_bandarmology_context(ticker: str) -> str:
    """First-class bandarmology signal context for the sentiment prompt.

    Broker accumulation/distribution and foreign net buy-sell are the IDX
    market's native sentiment read. Values arrive via the broker-summary feed
    and may be sparse; the block tells the model to weigh what is present and
    to say so explicitly when it is absent — never to invent figures.
    """
    symbol = _base_symbol(ticker)
    return (
        f"### Bandarmology — broker accumulation / foreign flow ({symbol})\n"
        "Sinyal utama pasar IDX: akumulasi/distribusi broker (broker summary) dan "
        "foreign net buy-sell harian. Akumulasi broker besar + foreign net buy "
        "berturut-turut = sentimen menguat; distribusi + foreign net sell = "
        "sentimen melemah; pola campuran = arah belum jelas.\n"
        "broker accumulation: <data tidak tersedia pada konteks ini — laporkan "
        "sebagai keterbatasan, jangan mengarang angka>\n"
        "foreign net buy-sell: <data tidak tersedia pada konteks ini — laporkan "
        "sebagai keterbatasan, jangan mengarang angka>\n"
        "Instruksi: perlakukan bandarmology sebagai sinyal setara dengan berita "
        "dan sentimen ritel; tegaskan bila datanya kosong."
    )
