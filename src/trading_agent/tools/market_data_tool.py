"""Tool for fetching historical price data and computing technical indicators (Quant agent)."""

from typing import Type
import yfinance as yf
import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class MarketDataInput(BaseModel):
    """Input schema for MarketDataTool."""

    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL, MSFT, GOOGL")


class MarketDataTool(BaseTool):
    """Fetches historical OHLCV data and computes technical indicators for a given ticker."""

    name: str = "Market Data & Technical Indicators"
    description: str = (
        "Retrieves historical price data (Open, High, Low, Close, Volume) and computes "
        "technical indicators: Simple Moving Averages (SMA 20, 50), recent volatility, "
        "and price trend. Use this for quantitative analysis only. Input: stock ticker symbol."
    )
    args_schema: Type[BaseModel] = MarketDataInput

    def _run(self, ticker: str) -> str:
        ticker = ticker.strip().upper()
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="6mo", interval="1d")
            if hist.empty or len(hist) < 20:
                return f"Insufficient or no history for {ticker}. Cannot compute indicators."
            df = hist.copy()
            df["SMA_20"] = df["Close"].rolling(20).mean()
            df["SMA_50"] = df["Close"].rolling(50).mean()
            df["Volatility_20d"] = df["Close"].pct_change().rolling(20).std() * (20 ** 0.5) * 100  # annualized %
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            price_chg_pct = ((latest["Close"] - prev["Close"]) / prev["Close"] * 100) if prev["Close"] else 0
            trend_vs_sma20 = "above" if latest["Close"] > latest["SMA_20"] else "below"
            trend_vs_sma50 = "above" if latest["Close"] > latest["SMA_50"] else "below"
            sma20_slope = (df["SMA_20"].iloc[-1] - df["SMA_20"].iloc[-5]) / 5 if len(df) >= 5 else 0
            report = (
                f"Ticker: {ticker}\n"
                f"Latest Close: {latest['Close']:.2f} (1d change: {price_chg_pct:+.2f}%)\n"
                f"SMA(20): {latest['SMA_20']:.2f} — Price is {trend_vs_sma20} SMA20\n"
                f"SMA(50): {latest['SMA_50']:.2f} — Price is {trend_vs_sma50} SMA50\n"
                f"Approx. 20-day annualized volatility: {latest['Volatility_20d']:.2f}%\n"
                f"SMA20 short-term slope (5-day): {sma20_slope:.4f}\n"
                f"Volume (latest): {int(latest['Volume'])}\n"
            )
            return report
        except Exception as e:
            return f"Error fetching market data for {ticker}: {e!s}"
