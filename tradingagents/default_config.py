import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_DEEP_THINK_PROVIDER":       "deep_think_provider",
    "TRADINGAGENTS_QUICK_THINK_PROVIDER":      "quick_think_provider",
    "TRADINGAGENTS_DEEP_THINK_BACKEND_URL":    "deep_think_backend_url",
    "TRADINGAGENTS_QUICK_THINK_BACKEND_URL":   "quick_think_backend_url",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_MAX_TOOL_ROUNDS":      "max_tool_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
    "TRADINGAGENTS_IDX_MODE":             "idx_mode",
    "TRADINGAGENTS_SIGNALS_ONLY":         "signals_only",
    "TRADINGAGENTS_TEMPERATURE":          "temperature",
    "TRADINGAGENTS_LLM_MAX_RETRIES":      "llm_max_retries",
    "TRADINGAGENTS_MAX_TOKENS":           "max_tokens",
    # Provider-specific reasoning/thinking knobs (None = each provider's own
    # default). Settable here for non-interactive runs; the CLI also offers an
    # interactive choice, which is skipped when the matching var is set.
    "TRADINGAGENTS_GOOGLE_THINKING_LEVEL":   "google_thinking_level",
    "TRADINGAGENTS_OPENAI_REASONING_EFFORT": "openai_reasoning_effort",
    "TRADINGAGENTS_ANTHROPIC_EFFORT":        "anthropic_effort",
}


_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value.

    Invalid values raise ``ValueError`` rather than silently falling back to a
    default — a misspelled boolean (e.g. ``treu``) or non-numeric int should fail
    loudly at startup, not quietly misconfigure an unattended run.
    """
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(
            f"expected a boolean ({'/'.join(_BOOL_TRUE + _BOOL_FALSE)}), got {value!r}"
        )
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        try:
            config[key] = _coerce(raw, config.get(key))
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_var}: {exc}") from exc
    return config


def build_default_config() -> dict:
    """The built-in defaults with the TRADINGAGENTS_* environment folded in.

    Read when the package is imported, as DEFAULT_CONFIG; call it again to see
    the     environment as it is now.
    """
    config = _apply_env_overrides({
        "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR") or os.path.join(_TRADINGAGENTS_HOME, "logs"),
        "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR") or os.path.join(_TRADINGAGENTS_HOME, "cache"),
        "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH") or os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md"),
        # Optional cap on the number of resolved memory log entries. When set,
        # the oldest resolved entries are pruned once this limit is exceeded.
        # Pending entries are never pruned. None disables rotation entirely.
        "memory_log_max_entries": None,
        # LLM settings
        "llm_provider": "openai",
        "deep_think_llm": "gpt-6-sol",
        "quick_think_llm": "gpt-6-luna",
        # When None, each provider's client falls back to its own default endpoint
        # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
        # The CLI overrides this per provider when the user picks one. Keeping a
        # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
        # being forwarded to Gemini, producing malformed request URLs).
        "backend_url": None,
        # A tier may name its own provider and endpoint (#1440): the quick tier serves
        # the analysts, researchers, debaters and trader, the deep tier the managers.
        # None means the tier uses llm_provider and backend_url.
        "quick_think_provider": None,
        "deep_think_provider": None,
        "quick_think_backend_url": None,
        "deep_think_backend_url": None,
        # Provider-specific thinking configuration
        "google_thinking_level": None,      # "high", "minimal", etc.
        "openai_reasoning_effort": None,    # "medium", "high", "low"
        "anthropic_effort": None,           # "high", "medium", "low"
        # Sampling temperature, forwarded to every provider when set. None leaves
        # each provider at its own default. Lower values reduce run-to-run
        # variation on models that honor it; reasoning models largely ignore it
        # and no setting makes LLM output bit-identical across runs (see README).
        "temperature": None,
        # SDK retry budget forwarded to every provider chat client. None leaves each
        # provider/SDK at its own default (usually 2). Raise it to ride out bursty
        # 429 throttling on rate-limited deployments instead of aborting a run (#1091).
        "llm_max_retries": None,
        # Cap on output tokens forwarded to every provider chat client. None leaves
        # each provider at its own default. Set it to bound a model that emits
        # unbounded reasoning/output and hangs or trips a gateway idle timeout
        # (e.g. some deepseek-v4-flash deployments, #1204).
        "max_tokens": None,
        # Checkpoint/resume: when True, LangGraph saves state after each node
        # so a crashed run can resume from the last successful step.
        "checkpoint_enabled": False,
        # Output language for analyst reports and final decision
        # Internal agent debate stays in English for reasoning quality
        "output_language": "English",
        # IDX (Indonesia) mode: when True, the sentiment/fundamentals stack
        # uses Bahasa Indonesia news (Kontan / Bisnis Indonesia / CNBC
        # Indonesia), IDX Keterbukaan Informasi disclosures, and bandarmology
        # flow signals; the US-only vendors (Reddit, StockTwits, SEC EDGAR)
        # degrade to disabled-with-reason. Additive: False keeps every US
        # path exactly as before.
        "idx_mode": False,
        # Debate and discussion settings
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "max_recur_limit": 100,
        # Rounds of tool calls an analyst may make before it is asked for its report.
        "max_tool_rounds": 20,
        # News / data fetching parameters
        # Increase for longer lookback strategies or to broaden macro coverage;
        # decrease to reduce token usage in agent prompts.
        "news_article_limit": 20,             # max articles per ticker (ticker-news)
        "global_news_article_limit": 10,      # max articles for global/macro news
        "global_news_lookback_days": 7,       # macro news lookback window
        # Search queries used by get_global_news for macro headlines. Extend or
        # replace to broaden geographic / sector coverage.
        "global_news_queries": [
            "Federal Reserve interest rates inflation",
            "S&P 500 earnings GDP economic outlook",
            "geopolitical risk trade war sanctions",
            "ECB Bank of England BOJ central bank policy",
            "oil commodities supply chain energy",
        ],
        # Data vendor configuration
        # Category-level configuration (default for all tools in category).
        # The configured value is the exact vendor chain — requests are NOT silently
        # routed to vendors you didn't choose. For ordered fallback, list several,
        # e.g. "yfinance,alpha_vantage". "default" uses all available vendors.
        "data_vendors": {
            "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance, yfinance_jk, idx
            "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
            # Statements come from SEC EDGAR as filed (US filers), then Yahoo; the
        # overview and insider tools, which SEC EDGAR does not serve, from Yahoo.
        "fundamental_data": "sec_edgar,yfinance",  # Options: sec_edgar, alpha_vantage, yfinance
            "news_data": "yfinance",             # Options: alpha_vantage, yfinance
            "macro_data": "fred",                # Options: fred (needs FRED_API_KEY)
            "prediction_markets": "polymarket",  # Options: polymarket (keyless)
        },
        # IDX (Indonesia) data layer — additive behind ``idx_mode`` (default off,
        # US vendors keep working unchanged). ``yfinance_jk`` is the primary IDX
        # price source (yfinance with .JK normalization); ``idx`` is the native
        # idx.co.id fallback. Enable with e.g.
        # {"data_vendors": {"core_stock_apis": "yfinance_jk,idx,yfinance"}}.
        # (``idx_mode`` master switch is defined once, above.)
        # IDX universe (subset of LQ45 shown; full list configurable here).
        "idx_universe": ["BBCA", "BBRI", "BMRI", "TLKM", "ASII", "GOTO"],
        # Tool-level configuration (takes precedence over category-level)
        "tool_vendors": {
            # Example: "get_stock_data": "alpha_vantage",  # Override category default
        },
        # Benchmark for alpha calculation in the reflection layer.
        # ``benchmark_ticker`` (when set) overrides the suffix map for all
        # tickers; leave it None to use ``benchmark_map`` for auto-detection
        # based on the ticker's exchange suffix. SPY remains the US default
        # so the reflection label keeps reading "Alpha vs SPY" for US tickers
        # while non-US tickers get their regional index automatically.
        # Trading days after the analysis date over which a decision's outcome is
        # measured, for reflection and for the backtest figures.
        "holding_period_days": 5,
        "benchmark_ticker": None,
        # IDX (Indonesia) mode: when True, calendar, sizing and risk math use
        # IDX microstructure (WIB sessions, 100-share lots, Rp tick grid,
        # ARA/ARB auto-rejection bands). Default off: US behavior unchanged.
        "idx_mode": False,
        "idx_currency": "IDR",
        "idx_lot_size": 100,
        "idx_timezone": "Asia/Jakarta",
        # Tick grid as [exclusive upper bound, tick in Rp]; None bound = open.
        "idx_tick_table": [[200, 1], [500, 2], [2000, 5], [5000, 10], [None, 25]],
        # ARA caps by previous close; ARB is a flat 15% below previous close.
        "idx_ara_pct_low": 0.35,   # prev Rp50-200
        "idx_ara_pct_mid": 0.25,   # prev >200-5000
        "idx_ara_pct_high": 0.20,  # prev >5000
        "idx_arb_pct": 0.15,
        # IDX (Indonesia) localization, additive behind ``idx_mode`` (default
        # False, so US behavior is unchanged). When True: prompts gain the IDX
        # context block (see agents.context._idx_prompt_note), the CLI offers
        # the LQ45/IDX30/IHSG universe (.JK symbols), and the benchmark
        # defaults to IHSG (^JKSE). ``signals_only`` defaults ON under idx_mode
        # (OJK compliance: research signals only, manual execution via a
        # licensed broker app); paper-portfolio accounting keeps working.
        "signals_only": False,
        "idx_benchmark_ticker": "^JKSE",
        "benchmark_map": {
            ".JK":  "^JKSE",       # Indonesia (IHSG / IDX Composite)
            ".NS":  "^NSEI",       # NSE India (Nifty 50)
            ".BO":  "^BSESN",      # BSE India (Sensex)
            ".T":   "^N225",       # Tokyo (Nikkei 225)
            ".TW":  "^TWII",       # Taiwan (TAIEX)
            ".TWO": "^TWII",       # Taipei OTC (TPEx has no Yahoo index; TAIEX)
            ".KS":  "^KS11",       # Korea (KOSPI)
            ".KQ":  "^KQ11",       # Korea (KOSDAQ)
            ".HK":  "^HSI",        # Hong Kong (Hang Seng)
            ".SI":  "^STI",        # Singapore (Straits Times)
            ".L":   "^FTSE",       # London (FTSE 100)
            ".DE":  "^GDAXI",      # Germany (DAX)
            ".PA":  "^FCHI",       # Paris (CAC 40)
            ".AS":  "^AEX",        # Amsterdam (AEX)
            ".SW":  "^SSMI",       # Switzerland (SMI)
            ".MI":  "FTSEMIB.MI",  # Milan (FTSE MIB)
            ".TO":  "^GSPTSE",     # Toronto (TSX Composite)
            ".AX":  "^AXJO",       # Australia (ASX 200)
            ".SS":  "000001.SS",   # Shanghai (SSE Composite)
            ".SZ":  "399001.SZ",   # Shenzhen (SZSE Component)
            ".SA":  "^BVSP",       # B3 Brazil (Ibovespa)
            ".JK":  "^JKSE",       # Indonesia (Jakarta Composite)
            "":     "SPY",         # default for US-listed tickers (no suffix)
        },

    })
    if config.get("idx_mode"):
        # signals_only defaults ON under idx_mode unless explicitly set
        # (explicit False is respected); US default stays OFF.
        if os.environ.get("TRADINGAGENTS_SIGNALS_ONLY") in (None, ""):
            config["signals_only"] = True
        # IDX benchmark defaults to IHSG unless explicitly overridden.
        if os.environ.get("TRADINGAGENTS_BENCHMARK_TICKER") in (None, ""):
            if not config.get("benchmark_ticker"):
                config["benchmark_ticker"] = config.get("idx_benchmark_ticker", "^JKSE")
    return config


DEFAULT_CONFIG = build_default_config()
