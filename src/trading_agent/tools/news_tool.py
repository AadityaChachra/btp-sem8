"""Tool for fetching recent news for a ticker (Researcher agent — sentiment input)."""

from typing import Type
import yfinance as yf
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class NewsFetchInput(BaseModel):
    """Input schema for NewsFetchTool."""

    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL, MSFT")


class NewsFetchTool(BaseTool):
    """Fetches recent financial news and headlines for a given stock ticker."""

    name: str = "Financial News Fetcher"
    description: str = (
        "Retrieves recent news headlines and links for a stock ticker from Yahoo Finance. "
        "Use this to gather qualitative information for sentiment analysis (bullish/bearish). "
        "Input: stock ticker symbol."
    )
    args_schema: Type[BaseModel] = NewsFetchInput

    def _run(self, ticker: str) -> str:
        ticker = ticker.strip().upper()
        try:
            t = yf.Ticker(ticker)
            news = t.get_news(count=15, tab="news") if hasattr(t, "get_news") else getattr(t, "news", [])
            if not news:
                return f"No recent news found for {ticker}. Consider sentiment as neutral or data unavailable."
            lines = [f"News for {ticker}:\n"]
            for i, n in enumerate(news[:15], 1):
                title = n.get("title", "No title")
                link = n.get("link", "")
                pub = n.get("publisher", "")
                lines.append(f"{i}. {title}")
                if pub:
                    lines.append(f"   Publisher: {pub}")
                if link:
                    lines.append(f"   Link: {link}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error fetching news for {ticker}: {e!s}"
