"""IDX (Indonesia Stock Exchange) trading calendar and market microstructure.

Additive module behind the ``idx_mode`` flag; US paths never import it unless
that flag is on (callers use lazy imports).

Sessions (Asia/Jakarta, WIB = UTC+7):
- Pre-open (no matching): from 08:45
- Sesi I: 09:00-11:30, Sesi II: 13:30-14:50, Monday-Friday.

Holidays: static 2025-2026 list below, including cuti bersama and the
Lebaran weeks. NOTE: this list MUST be refreshed yearly from the official IDX
trading-calendar announcements (IDX Pengumuman Libur Bursa) plus the joint
ministerial SKB on national holidays / cuti bersama — dates move every year,
especially Nyepi / Idul Fitri / Idul Adha / Waisak / Imlek.

Microstructure (pure math, no execution / order code):
- Lot size 100 shares; no fractional shares.
- Tick sizes (Rp): <200:Rp1, 200-<500:Rp2, 500-<2000:Rp5,
  2000-<5000:Rp10, >=5000:Rp25.
- Auto Rejection limits: ARA +35% for prev Rp50-200, +25% for >200-5000,
  +20% for >5000; ARB flat -15%. Orders outside [floor, ceiling] are
  rejected, never executed.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

try:
    WIB = ZoneInfo("Asia/Jakarta")
except Exception:  # minimal tzdata installs: fixed UTC+7 fallback, same wall-clock
    from datetime import timezone

    WIB = timezone(timedelta(hours=7))  # type: ignore[assignment]

# --- sessions ---------------------------------------------------------------
PREOPEN_START = time(8, 45)
SESI_I_START = time(9, 0)
SESI_I_END = time(11, 30)
SESI_II_START = time(13, 30)
SESI_II_END = time(14, 50)

LOT_SIZE = 100

# (exclusive upper bound, tick). Last row is open-ended.
TICK_TABLE: tuple[tuple[float, int], ...] = (
    (200, 1),
    (500, 2),
    (2000, 5),
    (5000, 10),
    (float("inf"), 25),
)

ARB_PCT = 0.15


def _d(y: int, m: int, d: int) -> date:
    return date(y, m, d)


# Static holiday list. See module docstring: refresh yearly from IDX.
IDX_HOLIDAYS: frozenset[date] = frozenset(
    [
        # --- 2025: national holidays + cuti bersama (SKB 3 Menteri) ---
        _d(2025, 1, 1),  # Tahun Baru
        _d(2025, 1, 27),  # Isra Mikraj
        _d(2025, 1, 28),  # cuti bersama Imlek
        _d(2025, 1, 29),  # Imlek 2576
        _d(2025, 3, 28),  # cuti bersama Nyepi
        _d(2025, 3, 29),  # Nyepi (Saturday; listed for completeness)
        _d(2025, 3, 31),  # cuti bersama / Lebaran week
        _d(2025, 4, 1),  # Idul Fitri 1446 H
        _d(2025, 4, 2),  # cuti bersama Lebaran
        _d(2025, 4, 3),  # cuti bersama Lebaran
        _d(2025, 4, 4),  # cuti bersama Lebaran
        _d(2025, 4, 7),  # cuti bersama Lebaran
        _d(2025, 4, 18),  # Jumat Agung
        _d(2025, 5, 1),  # Hari Buruh
        _d(2025, 5, 12),  # Waisak
        _d(2025, 5, 13),  # cuti bersama Waisak
        _d(2025, 5, 29),  # Kenaikan Yesus Kristus
        _d(2025, 5, 30),  # cuti bersama Kenaikan
        _d(2025, 6, 6),  # Idul Adha
        _d(2025, 6, 27),  # Tahun Baru Islam 1447 H
        _d(2025, 8, 18),  # cuti bersama Kemerdekaan (17th is Sunday)
        _d(2025, 9, 5),  # Maulid Nabi
        _d(2025, 12, 25),  # Natal
        _d(2025, 12, 26),  # cuti bersama Natal
        # --- 2026 (provisional: lunar dates shift; refresh from IDX) ---
        _d(2026, 1, 1),  # Tahun Baru
        _d(2026, 1, 16),  # Isra Mikraj (approx)
        _d(2026, 2, 17),  # Imlek 2577
        # Lebaran week 2026: Nyepi ~Mar 19, Idul Fitri ~Mar 20-21
        _d(2026, 3, 18),  # cuti bersama Lebaran
        _d(2026, 3, 19),  # Nyepi / cuti Lebaran
        _d(2026, 3, 20),  # Idul Fitri
        _d(2026, 3, 23),  # cuti bersama Lebaran
        _d(2026, 3, 24),  # cuti bersama Lebaran
        _d(2026, 4, 3),  # Jumat Agung
        _d(2026, 5, 1),  # Hari Buruh
        _d(2026, 5, 14),  # Kenaikan Yesus Kristus
        _d(2026, 5, 15),  # cuti bersama Kenaikan
        _d(2026, 5, 27),  # Idul Adha (approx)
        _d(2026, 5, 31),  # Waisak (approx; Sunday listed for completeness)
        _d(2026, 6, 1),  # Hari Pancasila
        _d(2026, 6, 16),  # Tahun Baru Islam (approx)
        _d(2026, 8, 17),  # Kemerdekaan
        _d(2026, 8, 26),  # Maulid Nabi (approx)
        _d(2026, 12, 25),  # Natal
    ]
)


# --- calendar ---------------------------------------------------------------
def _coerce_date(d: date | datetime | str) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def _coerce_wib(dt: date | datetime | str) -> datetime:
    """Naive values are assumed to be WIB wall-clock; aware ones are converted."""
    if isinstance(dt, datetime):
        return dt.replace(tzinfo=WIB) if dt.tzinfo is None else dt.astimezone(WIB)
    if isinstance(dt, date):
        return datetime(dt.year, dt.month, dt.day, tzinfo=WIB)
    parsed = datetime.strptime(str(dt), "%Y-%m-%d")
    return parsed.replace(tzinfo=WIB)


def is_trading_day(d: date | datetime | str) -> bool:
    """A Monday-Friday that is not an IDX holiday."""
    day = _coerce_date(d)
    return day.weekday() < 5 and day not in IDX_HOLIDAYS


def _day_start(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=WIB)


def sessions_for_day(d: date | datetime | str) -> list[tuple[datetime, datetime]]:
    """Sesi I + Sesi II as WIB-aware (start, end) pairs, or [] on holidays."""
    day = _coerce_date(d)
    if not is_trading_day(day):
        return []
    base = _day_start(day)
    return [
        (
            base.replace(hour=SESI_I_START.hour, minute=SESI_I_START.minute),
            base.replace(hour=SESI_I_END.hour, minute=SESI_I_END.minute),
        ),
        (
            base.replace(hour=SESI_II_START.hour, minute=SESI_II_START.minute),
            base.replace(hour=SESI_II_END.hour, minute=SESI_II_END.minute),
        ),
    ]


def session_state(dt: datetime | date | str) -> str:
    """Where ``dt`` falls: pre-open / sesi-1 / break / sesi-2 / closed."""
    moment = _coerce_wib(dt)
    if not is_trading_day(moment.date()):
        return "closed"
    t = moment.timetz().replace(tzinfo=None)
    if t < PREOPEN_START:
        return "closed"
    if t < SESI_I_START:
        return "pre-open"
    if t < SESI_I_END:
        return "sesi-1"
    if t < SESI_II_START:
        return "break"
    if t < SESI_II_END:
        return "sesi-2"
    return "closed"


def next_session(
    dt: datetime | date | str,
) -> tuple[datetime, datetime] | None:
    """Next (start, end) session at or after ``dt`` (current one if inside).

    Scans forward at most 14 days; returns None only when nothing is found,
    which cannot happen with a weekly Mon-Fri calendar.
    """
    moment = _coerce_wib(dt)
    for offset in range(14):
        day = (moment + timedelta(days=offset)).date()
        for start, end in sessions_for_day(day):
            if offset == 0 and moment > end:
                continue
            if offset == 0 and start <= moment <= end:
                return (start, end)
            if offset == 0 and moment < start:
                return (start, end)
            if offset > 0:
                return (start, end)
    return None


# --- microstructure ---------------------------------------------------------
def tick_size(price: float) -> int:
    """Minimum price increment for ``price`` (Rp)."""
    p = float(price)
    for upper, tick in TICK_TABLE:
        if p < upper:
            return tick
    return 25


def round_price_to_tick(price: float, mode: str = "nearest") -> int:
    """Round ``price`` to the tick grid; mode nearest|up|down."""
    tick = tick_size(price)
    p = float(price)
    if mode == "up":
        import math

        return int(math.ceil(p / tick) * tick)
    if mode == "down":
        import math

        return int(math.floor(p / tick) * tick)
    return int(round(p / tick) * tick)


def is_valid_tick(price: float) -> bool:
    """Whether ``price`` sits on the tick grid (integer Rp assumed)."""
    tick = tick_size(price)
    return abs(float(price) / tick - round(float(price) / tick)) < 1e-9


def ara_pct(prev_close: float) -> float:
    """ARA upside cap: 35% for Rp50-200, 25% for >200-5000, 20% above."""
    prev = float(prev_close)
    if prev <= 200:
        return 0.35
    if prev <= 5000:
        return 0.25
    return 0.20


def arb_pct(prev_close: float = 0.0) -> float:
    """ARB downside cap: flat 15% regardless of price."""
    return ARB_PCT


def price_band(prev_close: float) -> tuple[int, int]:
    """(floor, ceiling) for ``prev_close``, tick-aligned.

    Ceiling is rounded down and floor rounded up to the tick of the raw
    band edge, so the band itself is always executable.
    """
    prev = float(prev_close)
    raw_high = prev * (1.0 + ara_pct(prev))
    raw_low = prev * (1.0 - arb_pct(prev))
    return (
        round_price_to_tick(raw_low, "up"),
        round_price_to_tick(raw_high, "down"),
    )


def is_within_band(prev_close: float, price: float) -> bool:
    """Whether ``price`` is inside the ARA/ARB band (inclusive)."""
    low, high = price_band(prev_close)
    return low - 1e-9 <= float(price) <= high + 1e-9


def round_down_to_lot(shares: float, lot_size: int = LOT_SIZE) -> int:
    """Whole lots affordable within ``shares`` (floor); never fractional."""
    import math

    return int(math.floor(float(shares) / lot_size) * lot_size)


def is_round_lot(quantity: float, lot_size: int = LOT_SIZE) -> bool:
    """Integer shares in multiples of ``lot_size`` (no fractional shares)."""
    q = float(quantity)
    return q.is_integer() and int(q) % lot_size == 0


def validate_order(
    *,
    price: float,
    prev_close: float,
    quantity: float,
    held_qty: float = 0.0,
    lot_size: int = LOT_SIZE,
) -> str | None:
    """Auto-reject check for an IDX order; None means pass.

    ``quantity`` > 0 is a buy, < 0 a sell. Rejects (never executes):
    zero quantity, fractional/non-lot size, off-tick price, price outside
    the ARA/ARB band, and naked short (sell beyond ``held_qty``).
    """
    q = float(quantity)
    if q == 0:
        return "rejected: quantity is zero"
    if not is_round_lot(q, lot_size):
        return f"rejected: quantity {quantity} is not a multiple of lot {lot_size}"
    if not is_valid_tick(price):
        return f"rejected: price {price} is off the Rp{tick_size(price)} tick"
    if not is_within_band(prev_close, price):
        low, high = price_band(prev_close)
        return f"rejected: price {price} outside ARA/ARB band [{low}, {high}]"
    if q < 0 and float(held_qty) + q < -1e-9:
        return "rejected: naked short (sell exceeds holding; no short selling in idx_mode)"
    return None


def format_idr(amount: float) -> str:
    """Rp formatting with dot thousands separators and no decimals."""
    return "Rp" + f"{float(amount):,.0f}".replace(",", ".")
