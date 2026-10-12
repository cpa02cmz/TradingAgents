"""IDX (Indonesia) localization: prompts, signals-only gate, universe mapping."""

from __future__ import annotations

import pytest

from tradingagents.agents.context import (
    _idx_prompt_note,
    guard_live_execution,
)
from tradingagents.dataflows.config import run_config


def _base_debate_state(**overrides):
    state = {
        "company_of_interest": "BBCA.JK",
        "trade_date": "2026-08-14",
        "asset_type": "stock",
        "instrument_context": "",
        "portfolio_context": "",
        "past_context": "",
        "market_report": "Price 10250 IDR, RSI 61.",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_plan": "P",
        "trader_investment_plan": "T",
        "investment_debate_state": {
            "bull_history": "", "bear_history": "", "history": "",
            "current_response": "", "count": 0,
        },
        "risk_debate_state": {
            "history": "", "latest_speaker": "", "count": 0,
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "current_aggressive_response": "",
            "current_conservative_response": "", "current_neutral_response": "",
        },
    }
    state.update(overrides)
    return state


@pytest.mark.unit
def test_idx_note_present_when_on_absent_when_off():
    with run_config({"idx_mode": True}):
        note = _idx_prompt_note()
    assert "IHSG" in note
    assert "IDR" in note
    assert "lot" in note.lower()
    assert "ARA" in note and "ARB" in note
    assert "BUMN" in note
    assert "BI-Rate" in note
    assert "IDR/USD" in note
    assert "coal" in note.lower() and "CPO" in note

    with run_config({"idx_mode": False}):
        assert _idx_prompt_note() == ""


@pytest.mark.unit
def test_idx_note_reaches_researcher_and_risk_prompts():
    from langchain_core.messages import AIMessage

    from tradingagents.agents.researchers import bull_researcher
    from tradingagents.agents.risk_mgmt import aggressive_debator

    seen: list[str] = []

    class _LLM:
        def invoke(self, prompt, *a, **k):
            seen.append(prompt if isinstance(prompt, str) else str(prompt))
            return AIMessage("argument")

    with run_config({"idx_mode": True}):
        bull_researcher.create_bull_researcher(_LLM())(_base_debate_state())
        bull_prompt = " ".join(seen)
        assert "IHSG" in bull_prompt
        assert "IDR" in bull_prompt

        seen.clear()
        aggressive_debator.create_aggressive_debator(_LLM())(_base_debate_state())
        assert "IHSG" in " ".join(seen)

    with run_config({"idx_mode": False}):
        seen.clear()
        bull_researcher.create_bull_researcher(_LLM())(_base_debate_state())
        off_prompt = " ".join(seen)
        assert "IHSG" not in off_prompt
        assert "^JKSE" not in off_prompt


@pytest.mark.unit
def test_signals_only_guard_refuses_live_execution_but_paper_works(tmp_path):
    with run_config({"signals_only": True}):
        with pytest.raises(RuntimeError) as excinfo:
            guard_live_execution("place live order for BBCA.JK")
    message = str(excinfo.value)
    assert "OJK" in message
    assert "licensed broker" in message.lower()

    with run_config({"signals_only": False}):
        assert guard_live_execution("place live order") is None

    # Paper portfolio accounting stays fully working under signals-only.
    from tradingagents.portfolio import load_portfolio

    book = tmp_path / "book.json"
    book.write_text('{"cash": 1000000, "holdings": {"BBCA.JK": 10}}', encoding="utf-8")
    with run_config({"signals_only": True}):
        assert load_portfolio(str(book)) is not None


@pytest.mark.unit
def test_idx_universe_maps_to_jk_symbols_and_benchmark():
    from cli.prompts import (
        get_benchmark_display,
        get_idx_universe_symbols,
        list_idx_universes,
    )

    assert set(list_idx_universes()) == {"LQ45", "IDX30", "IHSG-picks"}
    for universe in list_idx_universes():
        symbols = get_idx_universe_symbols(universe)
        assert len(symbols) >= 5
        assert all(s.endswith(".JK") for s in symbols)
    assert "BBCA.JK" in get_idx_universe_symbols("LQ45")

    assert get_benchmark_display(True) == "^JKSE"
    assert get_benchmark_display(False) == "SPY"


@pytest.mark.unit
def test_idx_defaults_wire_through_default_config(monkeypatch):
    from tradingagents.default_config import build_default_config

    monkeypatch.delenv("TRADINGAGENTS_IDX_MODE", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_SIGNALS_ONLY", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_BENCHMARK_TICKER", raising=False)
    us = build_default_config()
    assert us["idx_mode"] is False
    assert us["signals_only"] is False
    assert us["benchmark_map"][".JK"] == "^JKSE"

    monkeypatch.setenv("TRADINGAGENTS_IDX_MODE", "true")
    idx = build_default_config()
    assert idx["signals_only"] is True
    assert idx["benchmark_ticker"] == "^JKSE"
