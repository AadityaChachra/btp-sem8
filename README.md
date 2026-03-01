# Autonomous Trading & Portfolio Optimization Agent

A **Multi-Agent System (MAS)** that simulates an investment committee. The system acts as a "Digital Investment Office" where specialized AI agents (Research, Technical Analysis, Risk/Decision) collaborate to produce an **auditable** Buy/Sell/Hold recommendation with full **explainability (XAI)**.

## Key Features

- **Multi-Agent Pipeline** — Three specialized agents (Researcher, Quant, Portfolio Manager) work sequentially, each contributing domain expertise.
- **Explainable AI (XAI)** — Every agent's reasoning is visible in the dashboard, so users can audit how the recommendation was formed.
- **Interactive Dashboard** — Candlestick charts with SMA overlays, volume bars, key metrics (price, volatility, high/low), and live workflow progress.
- **PDF Report Export** — Download a professional PDF report with the recommendation, market overview, and full agent reasoning.
- **Live Agent Communication** — Sidebar shows each agent's output in real-time as it completes.
- **Provider Agnostic** — Switch LLM providers (Groq, Gemini, OpenAI, Anthropic, Ollama) by changing two lines in `.env`.

## System Workflow

```
User Input (Ticker)
       |
       v
+--------------+     +--------------+     +-------------------+
|  Researcher  | --> |    Quant      | --> | Portfolio Manager |
|  (Sentiment) |     |  (Technical)  |     |    (Decision)     |
+--------------+     +--------------+     +-------------------+
       |                    |                       |
       v                    v                       v
  News & NLP          Price data &          BUY / SELL / HOLD
  sentiment          indicators            + justification
```

1. **Researcher** — Fetches recent financial news, analyzes sentiment (Bullish/Bearish/Neutral), reports qualitative factors.
2. **Quant** — Fetches historical prices, computes technical indicators (SMA 20/50, volatility), reports trends and entry/exit levels.
3. **Portfolio Manager** — Synthesizes both reports, weighs risk vs. reward, outputs **Buy / Sell / Hold** with justification.

## Technical Stack

| Component       | Choice                                                                 |
|----------------|------------------------------------------------------------------------|
| Language       | Python 3.11+                                                           |
| Orchestration  | **CrewAI** (agents, tasks, sequential crew)                            |
| LLM            | Configurable via `MODEL` + `API_KEY` in `.env` (Groq, Gemini, OpenAI, Anthropic, Ollama) |
| Market Data    | **yfinance** (OHLCV, news)                                             |
| Charts         | **Plotly** (candlestick, volume, SMA overlays)                         |
| PDF Export     | **fpdf2**                                                              |
| Frontend       | **Streamlit**                                                          |

## Project Structure

```
├── app.py                  # Streamlit dashboard
├── main.py                 # CLI: python main.py [TICKER]
├── requirements.txt
├── .env.example            # Template — copy to .env and add your key
├── .gitignore
├── README.md
└── src/
    └── trading_agent/
        ├── __init__.py
        ├── crew.py             # Crew definition, LLM config, run_crew()
        ├── config/
        │   ├── agents.yaml     # Agent role definitions
        │   └── tasks.yaml      # Task descriptions & expected outputs
        └── tools/
            ├── __init__.py
            ├── market_data_tool.py   # yfinance OHLCV + technical indicators
            └── news_tool.py          # yfinance news headlines
```

## Setup

### 1. Clone & create virtual environment

```bash
git clone <repo-url>
cd btp-sem8
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure LLM

Copy `.env.example` to `.env` and set your `MODEL` and `API_KEY`:

```bash
cp .env.example .env
```

Edit `.env` with your chosen provider:

| Provider | MODEL | API_KEY | Cost |
|----------|-------|---------|------|
| Groq | `groq/llama-3.1-70b-versatile` | `gsk_...` | Free |
| Google Gemini | `gemini/gemini-2.5-pro` | `AIza...` | Free tier |
| OpenAI | `openai/gpt-4o` | `sk-...` | Paid |
| Anthropic | `anthropic/claude-3-5-sonnet-20241022` | `sk-ant-...` | Paid |
| Ollama (local) | `ollama/llama3.2` | *(leave empty)* | Free |

Example `.env`:
```env
MODEL=groq/llama-3.1-70b-versatile
API_KEY=gsk_your_key_here
```

### 4. (Optional) Ollama — Local LLM, no API key

1. Install Ollama from [ollama.com](https://ollama.com)
2. Pull a model: `ollama run llama3.2`
3. Set in `.env`:
   ```env
   MODEL=ollama/llama3.2
   API_KEY=
   ```
4. Ensure Ollama is running before launching the app.

## Running

### Dashboard (recommended)

```bash
streamlit run app.py
```

- Enter a ticker (e.g. `AAPL`) in the sidebar.
- View the candlestick chart, metrics, and volume.
- Click **Run committee** to get the recommendation.
- Watch agent outputs appear live in the sidebar.
- Download a PDF report after the run completes.

### CLI

```bash
python main.py AAPL
```

## Dashboard Features

| Feature | Description |
|---------|-------------|
| **Candlestick Chart** | OHLC prices with SMA 20/50 overlays, selectable period (1M–2Y) |
| **Key Metrics** | Current price, daily change, period high/low, annualized volatility |
| **Volume Chart** | Color-coded volume bars (green = up day, red = down day) |
| **Live Workflow** | Real-time progress indicator showing which agent is working |
| **Agent Communication** | Sidebar panels update live as each agent completes |
| **Recommendation** | Color-coded BUY/SELL/HOLD box with justification |
| **Explainability (XAI)** | Three-column view of full agent reasoning |
| **PDF Export** | Download button for a professional report |

## Explainability (XAI)

The system is designed for transparency:

- **Full reasoning chain** visible: Researcher → Quant → Portfolio Manager
- **No black-box decisions** — every recommendation can be traced back to specific news, indicators, and synthesis logic
- **PDF report** preserves the audit trail for offline review

## License

This project is part of a Bachelor Thesis (BTP). Ensure compliance with API provider terms of service when deploying.
