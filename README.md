# Autonomous Trading & Portfolio Optimization Agent

Bachelor thesis project: a **Multi-Agent System (MAS)** that simulates an investment committee. The system acts as a "Digital Investment Office" where specialized AI agents (Research, Technical Analysis, Risk/Decision) collaborate to produce an **auditable** Buy/Sell/Hold recommendation with full **explainability (XAI)**.

## High-Level Concept

- **Not a black-box predictor**: Decisions are reached by explicit reasoning steps (research → technicals → decision).
- **Explainable**: The dashboard shows each agent’s output so users can see how the recommendation was formed.

## System Workflow

1. **User input** — Stock ticker (e.g. `AAPL`) via Streamlit dashboard or CLI.
2. **Researcher agent** — Fetches recent financial news (via tool), analyzes sentiment (Bullish/Bearish/Neutral), reports qualitative factors.
3. **Quant agent** — Fetches historical price data and computes technical indicators (e.g. SMAs, volatility), reports entry/exit considerations.
4. **Portfolio Manager agent** — Synthesizes Researcher and Quant outputs, weighs risk vs. reward, outputs **Buy / Sell / Hold** with a short justification.
5. **Visualization** — Dashboard displays the final recommendation and the full “chat log” (each agent’s report) for explainability.

## Technical Stack

| Component        | Choice                          |
|-----------------|----------------------------------|
| Language        | Python                           |
| Orchestration   | **CrewAI** (agents, tasks, crew) |
| LLM             | **Configurable** via `MODEL` in `.env` (Groq, OpenAI, Anthropic, Gemini, Ollama, etc.) |
| Market data     | **yfinance**                     |
| Frontend        | **Streamlit**                    |

## Project Structure

```
btp/
├── app.py                 # Streamlit dashboard entry point
├── main.py                # CLI: python main.py [TICKER]
├── requirements.txt
├── .env.example
├── README.md
└── src/
    └── trading_agent/
        ├── __init__.py
        ├── crew.py        # Crew definition, Groq LLM, run_crew()
        ├── config/
        │   ├── agents.yaml
        │   └── tasks.yaml
        └── tools/
            ├── __init__.py
            ├── market_data_tool.py   # yfinance history + technical indicators
            └── news_tool.py         # yfinance news for sentiment
```

## Setup

1. **Clone / open the project** and create a virtual environment:

   ```bash
   cd btp
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies** (use `python -m pip` if `pip` is not in your PATH):

   ```bash
   pip install -r requirements.txt
   # or:
   python3 -m pip install -r requirements.txt
   ```

3. **Configure LLM (pick one provider):**

   - Copy `.env.example` to `.env`.
   - Set `MODEL` and the matching API key. Default is Groq:

   ```bash
   cp .env.example .env
   # Edit .env:
   # MODEL=groq/llama-3.1-70b-versatile
   # GROQ_API_KEY=your_groq_key   # get at console.groq.com/keys
   ```

   **Other models:** set `MODEL` and the provider’s key (see `.env.example`):
   - **OpenAI:** `MODEL=openai/gpt-4o`, `OPENAI_API_KEY=sk-...`
   - **Anthropic:** `MODEL=anthropic/claude-3-5-sonnet-20241022`, `ANTHROPIC_API_KEY=...`
   - **Google Gemini:** `MODEL=gemini/gemini-2.0-flash`, `GOOGLE_API_KEY=...`
   - **Ollama (local):** see **Option 3: Ollama** below (free, no API key).

### Option 3: Run with Ollama (free, local, no API key)

**What is Ollama?**  
Ollama runs open-source LLMs (Llama, Mistral, etc.) **on your own machine**. No cloud, no API key, no account. Everything stays local and is free. The app talks to Ollama on `localhost`.

**Steps:**

1. **Install Ollama**  
   - macOS/Linux: [ollama.com](https://ollama.com) → download and install.  
   - Or: `curl -fsSL https://ollama.com/install.sh | sh`

2. **Pull a model** (one-time, downloads a few GB):
   ```bash
   ollama run llama3.2
   ```
   First run will download the model. You can stop the chat (Ctrl+C); the model stays installed. Other options: `ollama run mistral`, `ollama run llama3.1`, etc.

3. **Point the app at Ollama**  
   In your project `.env`:
   ```bash
   MODEL=ollama/llama3.2
   ```
   Leave `GROQ_API_KEY` empty or commented out. No other key needed.

4. **Run the app** (with Ollama already installed and a model pulled):
   ```bash
   streamlit run app.py
   ```
   The crew will use your local Ollama model. Slower than Groq but fully free and private.

**Troubleshooting:** If you get connection errors, ensure Ollama is running (e.g. open the Ollama app or run `ollama serve` in a terminal) and the model name in `MODEL` matches what you pulled (e.g. `llama3.2`, `mistral`).

## Running the System

### Dashboard (recommended)

```bash
streamlit run app.py
```

- Enter a ticker (e.g. `AAPL`) in the sidebar and click **Run committee**.
- View the recommendation and the three agent outputs (Researcher, Quant, Portfolio Manager) for full explainability.

### CLI

```bash
python main.py AAPL
```

Prints the recommendation and all agent outputs to the terminal.

## Agent Roles (Summary)

- **Researcher** — Uses the news tool; produces a sentiment report (Bullish/Bearish/Neutral) and qualitative themes. Does not recommend Buy/Sell.
- **Quant** — Uses the market data tool; produces a technical summary (trend, SMAs, volatility, entry/exit). No final recommendation.
- **Portfolio Manager** — No tools; reads Researcher and Quant outputs and outputs **BUY / SELL / HOLD** with a short justification.

## Explainability (XAI)

The system exposes:

- The **final recommendation** and **justification**.
- The **full reasoning chain**: Researcher output → Quant output → Portfolio Manager output.

This allows users and auditors to see exactly how the decision was reached, aligning with the thesis goal of moving beyond black-box predictions.

## License and Thesis

This code is for educational use as part of a bachelor thesis. Ensure compliance with Groq and Yahoo Finance terms of use when deploying or extending the project.
