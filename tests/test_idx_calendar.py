"""IDX calendar + microstructure: sessions, holidays, ARA/ARB clamps."""
from __future__ import annotations

from datetime import datetime

import pytest

from tradingagents.dataflows import idx_calendar


@pytest.mark.unit
def test_sesi_times_are_wib():
    sessions = idx_calendar.sessions_for_day("2025-04-08")  # Tue, trading day
    assert len(sessions) == 2
    (s1, e1), (s2, e2) = sessions
    assert (s1.hour, s1.minute) == (9, 0) and (e1.hour, e1.minute) == (11, 30)
    assert (s2.hour, s2.minute) == (13, 30) and (e2.hour, e2.minute) == (14, 50)
    assert str(s1.tzinfo) == "Asia/Jakarta"
    assert idx_calendar.session_state(datetime(2025, 4, 8, 10, 0)) == "sesi-1"
    assert idx_calendar.session_state(datetime(2025, 4, 8, 12, 0)) == "break"
    nxt = idx_calendar.next_session(datetime(2025, 4, 8, 12, 0))
    assert nxt is not None and (nxt[0].hour, nxt[0].minute) == (13, 30)


@pytest.mark.unit
def test_lebaran_week_is_holiday():
    assert not idx_calendar.is_trading_day("2025-04-02")  # cuti bersama Lebaran
    assert not idx_calendar.is_trading_day("2026-03-20")  # Idul Fitri 2026
    assert not idx_calendar.is_trading_day("2025-04-05")  # Saturday
    assert idx_calendar.is_trading_day("2025-04-08")  # Tue after Lebaran week
    assert idx_calendar.sessions_for_day("2025-04-02") == []


@pytest.mark.unit
def test_ara_arb_clamp_with_auto_reject():
    low, high = idx_calendar.price_band(100)  # 35% / 15%
    assert high == pytest.approx(135, abs=1) and low == pytest.approx(85, abs=1)
    assert idx_calendar.ara_pct(100) == 0.35
    assert idx_calendar.ara_pct(1000) == 0.25
    assert idx_calendar.ara_pct(6000) == 0.20
    assert idx_calendar.arb_pct(1000) == 0.15
    assert idx_calendar.is_within_band(1000, 1200)
    assert not idx_calendar.is_within_band(1000, 1300)  # > +25%
    assert not idx_calendar.is_within_band(1000, 800)  # < -15%
    # auto-reject semantics: outside band rejected, never executed
    assert idx_calendar.validate_order(price=1300, prev_close=1000, quantity=100) is not None
    assert "band" in idx_calendar.validate_order(price=1300, prev_close=1000, quantity=100)
    assert idx_calendar.validate_order(price=1100, prev_close=1000, quantity=100) is None


@pytest.mark.unit
def test_tick_and_lot_rules():
    assert idx_calendar.tick_size(150) == 1
    assert idx_calendar.tick_size(300) == 2
    assert idx_calendar.tick_size(1000) == 5
    assert idx_calendar.tick_size(3000) == 10
    assert idx_calendar.tick_size(8000) == 25
    assert idx_calendar.round_down_to_lot(250) == 200
    assert idx_calendar.is_round_lot(200) and not idx_calendar.is_round_lot(150)
    assert not idx_calendar.is_round_lot(100.5)
    # fractional / off-tick / naked short all rejected
    assert idx_calendar.validate_order(price=1000, prev_close=1000, quantity=150) is not None
    assert idx_calendar.validate_order(price=1001, prev_close=1000, quantity=100) is not None
    assert idx_calendar.validate_order(price=1000, prev_close=1000, quantity=-100, held_qty=0) is not None
    assert idx_calendar.validate_order(price=1000, prev_close=1000, quantity=-100, held_qty=100) is None


@pytest.mark.unit
def test_date_window_routes_through_idx():
    from tradingagents.dataflows import date_window

    assert date_window.is_trading_day("2025-04-02", idx_mode=True) is False
    assert date_window.is_trading_day("2025-04-08", idx_mode=True) is True
    assert date_window.is_trading_day("2025-04-05") is False  # US path: Saturday
    assert date_window.next_trading_day("2025-04-02", idx_mode=True) == "2025-04-08"
    assert date_window.next_session("2025-04-08", idx_mode=True) is not None


@pytest.mark.unit
def test_idx_mode_defaults_off_and_helpers():
    from tradingagents import default_config
    from tradingagents.backtest import size_to_lots
    from tradingagents.portfolio import format_money

    assert default_config.DEFAULT_CONFIG["idx_mode"] is False
    assert default_config.DEFAULT_CONFIG["idx_lot_size"] == 100
    assert size_to_lots(1_000_000, 1000) == 1000  # 1000 shares = 10 lots
    assert size_to_lots(50_000, 1000) == 0  # cannot afford one lot
    assert format_money(1500, "IDR", idx_mode=True) == "Rp1.500"
