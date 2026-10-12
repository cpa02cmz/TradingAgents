"""Tests for IDX (.JK) symbol normalization."""

import pytest

from tradingagents.dataflows.symbols import normalize_idx_symbol


@pytest.mark.unit
class TestNormalizeIdxSymbol:
    def test_bare_code_appends_jk(self):
        assert normalize_idx_symbol("BBCA") == "BBCA.JK"
        assert normalize_idx_symbol("GOTO") == "GOTO.JK"
        assert normalize_idx_symbol("TLKM") == "TLKM.JK"

    def test_already_suffixed_is_idempotent(self):
        assert normalize_idx_symbol("BBCA.JK") == "BBCA.JK"
        assert normalize_idx_symbol("bbca.jk") == "BBCA.JK"
        assert normalize_idx_symbol("  bbca.jk  ") == "BBCA.JK"

    def test_strip_suffix_for_idx_native_calls(self):
        assert normalize_idx_symbol("BBCA.JK", for_idx_native=True) == "BBCA"
        assert normalize_idx_symbol("BBCA", for_idx_native=True) == "BBCA"

    def test_accepts_four_letter_codes_case_insensitive(self):
        assert normalize_idx_symbol("bbri") == "BBRI.JK"
        assert normalize_idx_symbol("asii") == "ASII.JK"
        assert normalize_idx_symbol("bmri") == "BMRI.JK"

    def test_non_idx_symbols_pass_through(self):
        # Other exchange suffixes are never rewritten by the IDX normalizer.
        assert normalize_idx_symbol("600519.SS") == "600519.SS"
        assert normalize_idx_symbol("0700.HK") == "0700.HK"
        assert normalize_idx_symbol("") == ""
