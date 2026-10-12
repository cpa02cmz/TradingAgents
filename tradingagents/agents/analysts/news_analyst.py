from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import (
    get_global_news,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_news,
    get_global_news,
    get_macro_indicators,
    get_prediction_markets,
)


def _is_idx_mode(state=None) -> bool:
    """True when the run opted into the IDX stack (state wins, then config/env)."""
    try:
        if isinstance(state, dict) and "idx_mode" in state:
            value = state.get("idx_mode")
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        from tradingagents.dataflows.config import get_config

        config = get_config() or {}
        if isinstance(config, dict) and "idx_mode" in config:
            return bool(config.get("idx_mode"))
        import os

        return os.getenv("TRADINGAGENTS_IDX_MODE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    except Exception:  # noqa: BLE001 — fail closed to the US stack
        return False


def _idx_news_context(ticker: str, start_date: str, end_date: str) -> str:
    """Pre-fetched IDX news block for IDX mode. Never raises."""
    try:
        from tradingagents.dataflows.vendors.idx_news import fetch_idx_news

        return fetch_idx_news(ticker, start_date, end_date)
    except Exception:  # noqa: BLE001 — graceful: the prompt notes the gap
        return (
            "<IDX news unavailable: IDX vendor failed; "
            "this is not an absence of news>"
        )


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)
        idx_mode = _is_idx_mode(state)

        if idx_mode:
            # IDX mode: route to the Bahasa Indonesia stack (Kontan / Bisnis
            # Indonesia / CNBC Indonesia + IDX Keterbukaan Informasi) instead
            # of the US Yahoo-news path. The block is pre-fetched so the
            # model reports on data it was given; the get_news(start_date,
            # end_date) tool remains for follow-up US-independent queries.
            ticker = state.get("company_of_interest", "")
            idx_block = _idx_news_context(ticker, current_date, current_date)
            system_message = (
                f"You are a news researcher for the Indonesia Stock Exchange (IDX), analyzing the past week for {ticker}. "
                "Use the available tools: get_news(start_date, end_date) for follow-up queries about the instrument under analysis, "
                "get_global_news(curr_date, look_back_days, limit) for broader macro news, get_macro_indicators(indicator, curr_date, look_back_days) for macro data, "
                "and get_prediction_markets(topic, limit) for market-implied probabilities. "
                "Prefer Bahasa Indonesia sources (Kontan, Bisnis Indonesia, CNBC Indonesia) and IDX Keterbukaan Informasi disclosures, "
                "already pre-fetched below — quote them first and report honestly when they are silent. "
                "Berita IDX yang sudah dikumpulkan:\n<start_of_idx_news>\n" + idx_block + "\n<end_of_idx_news>"
                + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
                + get_language_instruction()
            )
        else:
            system_message = (
            f"You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(start_date, end_date) for news about the {asset_label} under analysis, get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news, get_macro_indicators(indicator, curr_date, look_back_days) to ground macro commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve'), and get_prediction_markets(topic, limit) for live market-implied probabilities of forward-looking events (e.g. 'Fed rate cut', 'recession 2026', geopolitical or sector events). Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Report what your tools support; another agent decides the trade."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in TOOLS]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, TOOLS, state["messages"])

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
