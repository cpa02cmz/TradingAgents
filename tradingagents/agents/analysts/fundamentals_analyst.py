from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
)

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_insider_transactions,
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


def _idx_disclosure_context(ticker: str) -> str:
    """Pre-fetched IDX disclosure block for IDX mode. Never raises."""
    try:
        from tradingagents.dataflows.vendors.idx_disclosure import (
            fetch_idx_announcements,
        )

        return fetch_idx_announcements(ticker)
    except Exception:  # noqa: BLE001 — graceful: the prompt notes the gap
        return (
            "<IDX disclosure unavailable: IDX vendor failed; "
            "this is not an absence of filings>"
        )


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        if _is_idx_mode(state):
            # IDX mode: route to IDX Keterbukaan Informasi disclosures instead
            # of the US SEC EDGAR path. IDX filings lag the quarter end and
            # coverage is sparse, so the brief is quarterly-lag and
            # sparse-data tolerant: report only filed figures, never estimate.
            ticker = state.get("company_of_interest", "")
            idx_block = _idx_disclosure_context(ticker)
            system_message = (
                "You are a researcher analyzing fundamental information for an Indonesia Stock Exchange (IDX) listing. "
                "Write a comprehensive report from IDX Keterbukaan Informasi disclosures (already pre-fetched below), company profile, and any "
                "financial statements the tools return. IDX quarterly reports lag the period end by weeks and coverage is sparse: "
                "treat a missing quarter as unavailable (not zero), never annualize or interpolate a figure no filing states, "
                "and say explicitly which periods were actually filed. Provide specific, actionable insights with supporting evidence."
                " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
                " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements, and `get_insider_transactions` for recent insider buying and selling."
                " Pengumuman IDX yang sudah dikumpulkan:\n<start_of_idx_disclosure>\n" + idx_block + "\n<end_of_idx_disclosure>"
                + get_language_instruction()
            )
        else:
            system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements, and `get_insider_transactions` for recent insider buying and selling."
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
