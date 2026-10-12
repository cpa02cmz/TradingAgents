"""Data vendors: one module or package per source, serving the router's methods.

IDX additions (``idx``, ``foreign_flow``) are re-exported lazily via
``__getattr__`` so a missing optional dependency never breaks ``import``.
"""

from __future__ import annotations

_LAZY_VENDORS = {
    "idx": "tradingagents.dataflows.vendors.idx",
    "foreign_flow": "tradingagents.dataflows.vendors.foreign_flow",
}


def __getattr__(name: str):
    if name in _LAZY_VENDORS:
        import importlib

        return importlib.import_module(_LAZY_VENDORS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["foreign_flow", "idx"]
