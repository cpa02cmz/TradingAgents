"""IDX (Indonesia Stock Exchange) data adapter.

Additive IDX layer: US vendors keep working unchanged. Everything here
degrades gracefully — network failures return an empty result plus a
warning and never raise, so the router can fall through to the next
vendor. No execution/order code. No secrets in this file.

Primary endpoints live on idx.co.id:
- stock summary: ``GetStockSummary``
- company profile: ``GetCompanyProfiles``

Third-party imports (``requests``, ``yfinance``) stay inside functions so
a missing optional dependency never breaks ``import``.
"""

from __future__ import annotations

import logging
from typing import Any

from tradingagents.dataflows.symbols import normalize_idx_symbol

logger = logging.getLogger(__name__)

IDX_BASE_URL = "https://www.idx.co.id"
IDX_STOCK_SUMMARY_ENDPOINT = "/api/GetStockSummary"
IDX_COMPANY_PROFILE_ENDPOINT = "/api/GetCompanyProfiles"

_REQUEST_TIMEOUT = 15.0


def _native_code(symbol: str) -> str:
    """Symbol as idx.co.id expects it: bare code without the ``.JK`` suffix."""
    return normalize_idx_symbol(symbol, for_idx_native=True)


def get_idx_summary(symbol: str) -> dict:
    """Fetch the IDX stock summary for ``symbol``.

    Returns the parsed payload as a dict, or ``{}`` (with a warning, never
    an exception) when the network fails, the payload is unparsable, or the
    optional ``requests`` dependency is missing.
    """
    code = _native_code(symbol)
    try:
        import requests  # lazy: optional dep must never break import

        resp = requests.get(
            IDX_BASE_URL + IDX_STOCK_SUMMARY_ENDPOINT,
            params={"code": code},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # graceful degradation: empty + warning, never crash
        logger.warning("IDX GetStockSummary unavailable for %r: %s", symbol, exc)
        return {}
    if isinstance(data, dict):
        return data
    return {"data": data}


def get_idx_profile(symbol: str) -> dict:
    """Fetch the IDX company profile for ``symbol``.

    Same graceful-degradation contract as :func:`get_idx_summary`.
    """
    code = _native_code(symbol)
    try:
        import requests  # lazy: optional dep must never break import

        resp = requests.get(
            IDX_BASE_URL + IDX_COMPANY_PROFILE_ENDPOINT,
            params={"code": code},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # graceful degradation: empty + warning, never crash
        logger.warning("IDX GetCompanyProfiles unavailable for %r: %s", symbol, exc)
        return {}
    if isinstance(data, dict):
        return data
    return {"data": data}


def get_yfinance_jk_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Primary IDX price source: yfinance with ``.JK`` normalization.

    ``BBCA`` is queried as ``BBCA.JK``; already-suffixed or non-IDX symbols
    pass through via :func:`normalize_idx_symbol` rules. Delegates lazily to
    the Yahoo vendor so this module imports without ``yfinance`` installed.
    """
    from tradingagents.dataflows.vendors.yahoo.market import (  # lazy
        get_YFin_data_online,
    )

    canonical = normalize_idx_symbol(symbol)
    return get_YFin_data_online(canonical, start_date, end_date)


def get_stock_data(symbol: str, start_date: str, end_date: str) -> Any:
    """Router ``idx`` fallback: render the IDX summary as text.

    Raises :class:`NoMarketDataError` when the summary is empty so the
    router can continue down the fallback chain; network failures themselves
    never crash here because :func:`get_idx_summary` already degrades to ``{}``.
    """
    from tradingagents.dataflows.errors import NoMarketDataError  # lazy, cheap

    canonical = normalize_idx_symbol(symbol)
    native = _native_code(symbol)
    summary = get_idx_summary(symbol)
    if not summary:
        raise NoMarketDataError(symbol, canonical, "IDX GetStockSummary returned no data")
    return f"# IDX stock summary for {native} ({canonical}) {start_date} to {end_date}\n{summary}"
