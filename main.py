#!/usr/bin/env python3
"""
CLI entry point to run the investment committee for a single ticker.
Usage: python main.py [TICKER]
       Default ticker: AAPL
"""

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))

from dotenv import load_dotenv
load_dotenv(root / ".env")

from trading_agent.crew import run_crew


def main():
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").strip().upper()
    model = (os.environ.get("MODEL") or "groq/llama-3.1-70b-versatile").strip()
    if model.startswith("groq/") and not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not set. For Ollama use MODEL=ollama/llama3.2 in .env (no key).", file=sys.stderr)
        sys.exit(1)
    print(f"Running committee for {ticker} (model: {model})...")
    result = run_crew(ticker, api_key=os.environ.get("GROQ_API_KEY"))
    print("\n--- Recommendation ---")
    print(result["recommendation"])
    print("\n--- Justification ---")
    print(result["justification"])
    print("\n--- Researcher (sentiment) ---")
    print(result["research_output"])
    print("\n--- Quant (technical) ---")
    print(result["quant_output"])
    print("\n--- Portfolio Manager (decision) ---")
    print(result["decision_output"])


if __name__ == "__main__":
    main()
