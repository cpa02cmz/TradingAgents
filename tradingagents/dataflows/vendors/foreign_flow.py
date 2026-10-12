"""Foreign-flow data for IDX symbols (foreign net buy/sell, broker summary).

Additive enrichment layer: disabled unless configured. No execution/order
code. No secrets in this file — credentials come from the environment.

Third-party imports (``requests``) stay inside functions so a missing
optional dependency never breaks ``import``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Env hook for the IndexAlpha-compatible REST source. Empty (the default)
# means disabled — fetch() degrades to an empty result with a warning.
FOREIGN_FLOW_BASE_URL_ENV = "INDEX_ALPHA_BASE_URL"
FOREIGN_FLOW_API_KEY_ENV = "INDEX_ALPHA_API_KEY"

_REQUEST_TIMEOUT = 15.0


class ForeignFlowClient:
    """Interface: ``fetch(symbol)`` -> ``{foreign_net, broker_summary}``."""

    def fetch(self, symbol: str) -> dict:
        """Return foreign-flow data for ``symbol`` (base: disabled sentinel)."""
        return {"foreign_net": None, "broker_summary": "disabled: no foreign-flow source configured"}


class IndexAlphaForeignFlowClient(ForeignFlowClient):
    """IndexAlpha-compatible REST hook.

    Reads ``base_url``/``api_key`` from the environment
    (``INDEX_ALPHA_BASE_URL`` / ``INDEX_ALPHA_API_KEY``); empty (default) is
    disabled. Never raises on network failure — returns an empty payload
    with a warning so callers can proceed without it.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url if base_url is not None else os.getenv(FOREIGN_FLOW_BASE_URL_ENV, "")
        self.api_key = api_key if api_key is not None else os.getenv(FOREIGN_FLOW_API_KEY_ENV, "")

    @property
    def enabled(self) -> bool:
        """Whether the hook is configured (both base_url and api_key set)."""
        return bool(self.base_url and self.api_key)

    def fetch(self, symbol: str) -> dict:
        """Fetch foreign-flow data; empty + warning when disabled or failing."""
        if not self.enabled:
            logger.warning("Foreign-flow (IndexAlpha) disabled: no base_url/api_key configured")
            return {"foreign_net": None, "broker_summary": "disabled"}
        try:
            import requests  # lazy: optional dep must never break import

            resp = requests.get(
                self.base_url.rstrip("/") + "/foreign-flow",
                params={"symbol": symbol},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Foreign-flow request failed for %r: %s", symbol, exc)
            return {"foreign_net": None, "broker_summary": ""}
        if isinstance(data, dict):
            return {
                "foreign_net": data.get("foreign_net"),
                "broker_summary": data.get("broker_summary", ""),
            }
        return {"foreign_net": None, "broker_summary": ""}


class StockbitScrapeClient(ForeignFlowClient):
    """UNOFFICIAL stub: Stockbit scraping is not an official API.

    Disabled by default (``enabled=False``). Even when constructed with
    ``enabled=True`` this stub performs no scraping and returns an empty
    payload — it exists only to mark the integration point. Do not enable
    scraping against a site's terms of service.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def fetch(self, symbol: str) -> dict:
        """Always an empty payload; warns that the unofficial stub is disabled."""
        logger.warning(
            "Foreign-flow (Stockbit scrape, UNOFFICIAL) disabled: refusing to scrape %r",
            symbol,
        )
        return {"foreign_net": None, "broker_summary": "disabled: unofficial Stockbit stub"}


def get_foreign_flow(symbol: str) -> dict:
    """Convenience: IndexAlpha hook when configured, else the disabled sentinel."""
    return IndexAlphaForeignFlowClient().fetch(symbol)
