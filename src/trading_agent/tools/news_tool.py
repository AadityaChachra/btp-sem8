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
                # yfinance >= 0.2.40 nests data under "content"
                content = n.get("content", n)  # fall back to flat dict for older versions
                title = content.get("title", n.get("title", "No title"))
                provider = content.get("provider", {})
                pub = provider.get("displayName", "") if isinstance(provider, dict) else n.get("publisher", "")
                click = content.get("clickThroughUrl", {})
                link = click.get("url", "") if isinstance(click, dict) else n.get("link", "")
                lines.append(f"{i}. {title}")
                if pub:
                    lines.append(f"   Publisher: {pub}")
                if link:
                    lines.append(f"   Link: {link}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error fetching news for {ticker}: {e!s}"
