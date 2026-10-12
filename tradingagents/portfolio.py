"""The caller's book, as the decision agents see it.

Optional input to a run: what is held, at what average price, and how much cash
is free. Without it the agents cannot tell adding to a full position from
opening a new one. Three states are distinct and must stay so: a position, a
flat book, and no context at all, since treating "not provided" as "flat" would
invent a fact about the caller's account.

Broker-neutral by construction: quantities are generic units and the currency is
whatever label the caller passes, so nothing here implies a venue or an
execution path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class Position(BaseModel):
    ticker: str = Field(description="Instrument symbol, e.g. AAPL")
    quantity: float = Field(description="Signed units held; negative is short")
    average_price: float | None = Field(default=None, description="Average entry price per unit")


class PortfolioContext(BaseModel):
    cash: float | None = Field(default=None, description="Free cash available")
    currency: str | None = Field(default=None, description="Currency label for cash and prices")
    positions: list[Position] = Field(default_factory=list)

    def position_in(self, ticker: str) -> Position | None:
        return next((p for p in self.positions if p.ticker.upper() == ticker.strip().upper()), None)

    def render(self, ticker: str) -> str:
        """The portfolio block for the decision agents, led by the analyzed instrument."""
        symbol = ticker.strip().upper()
        held = self.position_in(symbol)
        if held is None:
            lines = [f"- No current position in {symbol}"]
        else:
            price = f", average price {held.average_price:,.2f}" if held.average_price is not None else ""
            lines = [f"- Current position in {symbol}: {held.quantity:,.4g} units{price}"]
        if self.cash is not None:
            lines.append(f"- Cash available: {self.cash:,.2f}{' ' + self.currency if self.currency else ''}")
        others = [p for p in self.positions if p is not held]
        if others:
            lines.append("- Other positions: " + ", ".join(f"{p.ticker.upper()} {p.quantity:,.4g}" for p in others))
        return "Portfolio at the analysis date:\n" + "\n".join(lines)

    def fingerprint(self) -> str:
        """Stable digest of the book, so a changed one cannot resume a stale run."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


def load_portfolio(path: str | Path) -> PortfolioContext:
    """Read a portfolio JSON file, failing before the run rather than mid-graph."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return PortfolioContext.model_validate(data)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"portfolio file {path} is not usable: {exc}") from exc


# --- IDX helpers (additive; US behavior unchanged when idx_mode is off) --------
IDX_LOT_SIZE = 100


def _idx_enabled(idx_mode: bool = False, config: dict | None = None) -> bool:
    if config is not None:
        try:
            if bool(config.get("idx_mode")) or bool(idx_mode):
                return True
        except AttributeError:
            return bool(idx_mode)
        if idx_mode:
            return True
    elif idx_mode:
        return True
    try:
        from tradingagents.dataflows.config import get_config

        return bool(get_config().get("idx_mode", False))
    except Exception:
        return False


def format_money(amount: float, currency: str | None = None,
                 idx_mode: bool = False, config: dict | None = None) -> str:
    """Money label; IDR in idx_mode renders as Rp with no decimals."""
    cur = (currency or "").strip()
    if _idx_enabled(idx_mode, config) or cur.upper() == "IDR":
        from tradingagents.dataflows import idx_calendar  # lazy

        label = idx_calendar.format_idr(amount)
        return label if not cur or cur.upper() == "IDR" else f"{label} {cur}"
    return f"{float(amount):,.2f}" + (f" {cur}" if cur else "")


def round_down_to_lot(shares: float, lot_size: int = IDX_LOT_SIZE) -> int:
    """Floor ``shares`` to whole lots (lazy IDX import keeps US path untouched)."""
    from tradingagents.dataflows import idx_calendar  # lazy

    return idx_calendar.round_down_to_lot(shares, lot_size)


def is_round_lot(quantity: float, lot_size: int = IDX_LOT_SIZE) -> bool:
    from tradingagents.dataflows import idx_calendar  # lazy

    return idx_calendar.is_round_lot(quantity, lot_size)


def validate_idx_order(*, price: float, prev_close: float, quantity: float,
                       held_qty: float = 0.0,
                       lot_size: int = IDX_LOT_SIZE) -> str | None:
    """Auto-reject check for IDX orders; None passes, string rejects (never executes)."""
    from tradingagents.dataflows import idx_calendar  # lazy

    return idx_calendar.validate_order(price=price, prev_close=prev_close,
                                       quantity=quantity, held_qty=held_qty,
                                       lot_size=lot_size)
