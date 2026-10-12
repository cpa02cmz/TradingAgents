"""IDX sentiment/fundamentals stack: graceful degradation, routing, bandarmology."""
from __future__ import annotations

import copy
import urllib.error

import pytest

import tradingagents.default_config as default_config
import tradingagents.dataflows.config as config_module
from tradingagents.dataflows.config import set_config


def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture(autouse=True)
def _clean_config(monkeypatch):
    _reset_config()
    monkeypatch.delenv("TRADINGAGENTS_IDX_MODE", raising=False)
    yield
    _reset_config()
    monkeypatch.delenv("TRADINGAGENTS_IDX_MODE", raising=False)


@pytest.mark.unit
def test_idx_news_degrades_gracefully_offline(monkeypatch):
    """Every endpoint down -> empty list + unavailable block, never raises."""
    import urllib.request

    import tradingagents.dataflows.vendors.idx_news as idx_news

    def _boom(*a, **k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert idx_news.IDXNewsClient(timeout=0.1).fetch_news("BBCA.JK") == []
    block = idx_news.fetch_idx_news("BBCA.JK", "2026-10-01", "2026-10-08")
    assert "unavailable" in block and "not an absence" in block


@pytest.mark.unit
def test_idx_disclosure_degrades_gracefully_offline(monkeypatch):
    """Disclosure + PDF entry point never raise on failure."""
    import urllib.request

    import tradingagents.dataflows.vendors.idx_disclosure as idxd

    def _boom(*a, **k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert idxd.IDXDisclosureClient(timeout=0.1).fetch_announcements("BBCA.JK") == []
    out = idxd.parse_fundamentals_pdf("/nonexistent/laporan.pdf", symbol="BBCA.JK")
    assert out["status"] == "unavailable"


@pytest.mark.unit
def test_idx_mode_routes_sentiment_to_idx_stack(monkeypatch):
    """idx_mode on -> IDX news in the prompt, US forums disabled-with-reason."""
    from unittest.mock import MagicMock

    import tradingagents.agents.analysts.sentiment_analyst as sentiment
    from tradingagents.agents.schemas import SentimentBand, SentimentReport

    set_config({"idx_mode": True})
    monkeypatch.setattr(
        "tradingagents.dataflows.vendors.idx_news.IDXNewsClient.fetch_news",
        lambda self, *a, **k: [{"title": "BBCA laba naik", "publisher": "kontan.co.id",
                                        "link": "https://x", "pub_date": "2026-10-01",
                                        "summary": "laba bersih menguat"}],
    )

    captured = {}
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt)
        or SentimentReport(overall_band=SentimentBand.BULLISH, overall_score=7.0,
                           confidence="medium", narrative="n")
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    sentiment.create_sentiment_analyst(llm)({
        "company_of_interest": "BBCA.JK", "trade_date": "2026-10-08",
        "asset_type": "stock", "messages": [],
    })

    def _text(prompt):
        if isinstance(prompt, str):
            return prompt
        parts = []
        for m in prompt:
            if isinstance(m, dict):
                parts.append(str(m.get("content", "")))
            else:
                parts.append(str(getattr(m, "content", m)))
        return "\n".join(parts)

    text = _text(captured["prompt"])
    assert "BBCA laba naik" in text  # routed to idx_news
    assert "disabled in IDX mode" in text  # US forums gated
    assert "wallstreetbets" not in text


@pytest.mark.unit
def test_bandarmology_context_is_first_class_in_idx_prompt():
    """Bandarmology block carries broker accumulation + foreign net buy-sell."""
    import tradingagents.agents.analysts.sentiment_analyst as sentiment
    from tradingagents.dataflows.vendors.idx_news import build_bandarmology_context

    block = build_bandarmology_context("BBCA.JK")
    assert "broker accumulation" in block
    assert "foreign net" in block

    msg = sentiment._build_system_message(
        ticker="BBCA.JK", start_date="2026-10-01", end_date="2026-10-08",
        news_block="n", stocktwits_block="s", reddit_block="r",
        subreddits=(), bandarmology_block=block, idx_mode=True,
    )
    assert "broker accumulation" in msg and "foreign net buy-sell" in msg
    assert "Bahasa Indonesia" in msg or "code-mix" in msg


@pytest.mark.unit
def test_us_paths_untouched_when_idx_mode_off(monkeypatch):
    """Flag off -> sentiment still searches subreddits; EDGAR still serves."""
    import tradingagents.agents.analysts.sentiment_analyst as sentiment
    import tradingagents.dataflows.vendors.sec_edgar as edgar

    assert sentiment._is_idx_mode({}) is False
    assert edgar._is_idx_mode() is False
    msg = sentiment._build_system_message(
        ticker="NVDA", start_date="2026-10-01", end_date="2026-10-08",
        news_block="n", stocktwits_block="s", reddit_block="r",
    )
    assert "r/wallstreetbets" in msg
    assert "bandarmology" not in msg.lower()
