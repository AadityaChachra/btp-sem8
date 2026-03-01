# BTP Presentation Guide — Complete Project Explanation

> **Project**: Autonomous Trading & Portfolio Optimization Agent  
> **Approach**: Multi-Agent System (MAS) with Explainable AI (XAI)

---

## TABLE OF CONTENTS

1. [High-Level Concept (What & Why)](#1-high-level-concept)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [File-by-File Code Explanation](#3-file-by-file-code-explanation)
4. [End-to-End Execution Flow](#4-end-to-end-execution-flow)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Panel Q&A — Expected Questions and Answers](#6-panel-qa)

---

## 1. HIGH-LEVEL CONCEPT

### What is this project?
This project builds a **Digital Investment Office** — a system where three specialized AI agents simulate an **investment committee** to analyze a stock and produce a **Buy / Sell / Hold** recommendation.

### Why Multi-Agent instead of a single model?
- A single LLM prompt would combine sentiment analysis, technical analysis, and decision-making into one black-box call — no explainability.
- By splitting into 3 agents, each agent's reasoning is **independently visible and auditable** — this is our **Explainability (XAI)** feature.
- Each agent is a specialist (just like in real investment firms — research analysts, quant analysts, portfolio managers are different people).

### Core Thesis Argument
> "Traditional ML-based trading models are black-box predictors. Our multi-agent approach produces auditable, explainable decisions where every step of the reasoning chain is visible."

---

## 2. ARCHITECTURE & DATA FLOW

### Pipeline (Sequential)
```
User enters ticker (e.g. "AAPL")
         │
         ▼
┌─────────────────────┐
│   RESEARCHER AGENT  │ ← Uses NewsFetchTool (Yahoo Finance news)
│  Produces: Sentiment│   Output: "Bullish" / "Bearish" / "Neutral"
│  report with themes │   + key themes (earnings, macro, sector)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│     QUANT AGENT     │ ← Uses MarketDataTool (Yahoo Finance OHLCV)
│  Produces: Technical│   Output: SMA20, SMA50, volatility,
│  analysis report    │   trend direction, entry/exit levels
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  PORTFOLIO MANAGER  │ ← No tools; reads both reports (context)
│  Produces: Final    │   Output: BUY / SELL / HOLD
│  recommendation     │   + 2-4 sentence justification
└─────────────────────┘
         │
         ▼
Dashboard / CLI shows everything
```

### How data flows:
1. User gives a ticker → it's injected into task descriptions via `{ticker}` placeholder
2. Researcher calls `NewsFetchTool("AAPL")` → gets 15 headlines → LLM analyzes sentiment
3. Quant calls `MarketDataTool("AAPL")` → gets 6 months OHLCV + indicators → LLM summarizes
4. Portfolio Manager receives BOTH outputs as `context` → LLM weighs them → outputs BUY/SELL/HOLD
5. Dashboard displays all three outputs + charts + PDF download

---

## 3. FILE-BY-FILE CODE EXPLANATION

---

### 3.1 `requirements.txt` — Dependencies

```
crewai          → Multi-agent orchestration framework (creates agents, tasks, crews)
crewai-tools    → Base class for custom tools (BaseTool)
groq            → Groq API client (for fast LLM inference)
litellm         → Universal LLM router (translates groq/*, gemini/*, openai/* into correct API calls)
yfinance        → Yahoo Finance API wrapper (market data + news)
pandas/numpy    → Data manipulation for technical indicators
streamlit       → Web dashboard framework
plotly          → Interactive charts (candlestick, volume)
fpdf2           → PDF generation
python-dotenv   → Loads .env file into os.environ
pydantic        → Data validation (used for tool input schemas)
PyYAML          → Parses agents.yaml and tasks.yaml
```

**Why LiteLLM?** CrewAI uses LiteLLM internally. When we set `MODEL=groq/llama-3.1-70b-versatile`, LiteLLM sees the `groq/` prefix and routes the call to Groq's API. Same logic for `gemini/`, `openai/`, `ollama/`. This makes our system **provider-agnostic** — we only change 2 lines in `.env` to switch providers.

---

### 3.2 `.env` and `.env.example` — Configuration

```env
MODEL=groq/llama-3.1-70b-versatile    # Format: provider/model-name
API_KEY=gsk_your_key_here              # Single API key for whichever provider
```

**How the key mapping works:**
- User sets one `API_KEY` in `.env`
- Our code reads the `MODEL` prefix (e.g., `groq/...` → prefix is `groq`)
- Maps it to the provider-specific env var LiteLLM expects: `groq` → `GROQ_API_KEY`
- Sets `os.environ["GROQ_API_KEY"] = api_key` so LiteLLM picks it up

This is handled by `_set_provider_key()` in crew.py.

---

### 3.3 `src/trading_agent/config/agents.yaml` — Agent Definitions

This YAML defines WHO each agent IS (role, goal, backstory). CrewAI uses these to construct the system prompt for each LLM call.

```yaml
researcher:
  role: "Financial News & Sentiment Researcher"
  goal: "Scrape and analyze recent financial news... produce sentiment assessment"
  backstory: "You are a seasoned research analyst who focuses on qualitative factors..."
```

- **role** → Becomes part of the system prompt: "You are a Financial News & Sentiment Researcher"
- **goal** → Tells the LLM what to accomplish
- **backstory** → Gives the LLM a "persona" for better domain-specific outputs
- **Why YAML?** Separates configuration from code. You can tweak agent behavior without modifying Python.

Three agents defined:
1. **researcher** — Qualitative analysis (news, sentiment). Explicitly told "do not make trading decisions."
2. **quant** — Quantitative analysis (prices, indicators). Told "ignore news, use only numbers."
3. **portfolio_manager** — Synthesis and decision. "You are the final decision-maker."

---

### 3.4 `src/trading_agent/config/tasks.yaml` — Task Definitions

This YAML defines WHAT each agent must DO.

```yaml
research_task:
  description: 'For the stock ticker "{ticker}", use the Financial News tool...'
  expected_output: 'A concise sentiment report: (1) Overall sentiment...'
  agent: researcher
```

- **description** → The actual prompt/instruction sent to the agent. Note `{ticker}` — this is a placeholder replaced at runtime with the actual ticker.
- **expected_output** → Tells the LLM the expected format of its response. This constrains the output.
- **agent** → Links this task to which agent executes it.
- **context** (on decision_task) → `[research_task, quant_task]` means the Portfolio Manager receives outputs from both previous tasks as input.

Key design: `quant_task` does NOT have `context: [research_task]`. The Quant works independently to avoid bias from the Researcher's sentiment.

---

### 3.5 `src/trading_agent/tools/__init__.py` — Tool Exports

```python
from trading_agent.tools.market_data_tool import MarketDataTool
from trading_agent.tools.news_tool import NewsFetchTool
__all__ = ["MarketDataTool", "NewsFetchTool"]
```

Simple re-export so other files can write `from trading_agent.tools import MarketDataTool`.

---

### 3.6 `src/trading_agent/tools/news_tool.py` — News Fetcher Tool

**Purpose**: Gives the Researcher agent access to real financial news.

**Line-by-line:**

```python
class NewsFetchInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL, MSFT")
```
- Pydantic model that validates input. CrewAI uses this schema to tell the LLM what arguments the tool expects.
- `Field(...)` means the field is required.

```python
class NewsFetchTool(BaseTool):
    name: str = "Financial News Fetcher"
    description: str = "Retrieves recent news headlines..."
    args_schema: Type[BaseModel] = NewsFetchInput
```
- Inherits from CrewAI's `BaseTool`.
- `name` and `description` are shown to the LLM so it knows when to use this tool.
- `args_schema` tells the LLM the input format.

```python
def _run(self, ticker: str) -> str:
    t = yf.Ticker(ticker)
    news = t.get_news(count=15, tab="news")
```
- `_run()` is the method CrewAI calls when the agent decides to use this tool.
- Uses yfinance to fetch up to 15 recent news articles for the ticker.

```python
    for i, n in enumerate(news[:15], 1):
        title = n.get("title", "No title")
        pub = n.get("publisher", "")
        lines.append(f"{i}. {title}")
```
- Formats news as a numbered list with title and publisher.
- Returns this as a plain text string — the LLM then analyzes sentiment from these headlines.

**Key point**: The tool only FETCHES news. The ANALYSIS (bullish/bearish/neutral) is done by the LLM based on the agent's instructions.

---

### 3.7 `src/trading_agent/tools/market_data_tool.py` — Market Data Tool

**Purpose**: Gives the Quant agent access to price data and technical indicators.

**Line-by-line:**

```python
def _run(self, ticker: str) -> str:
    t = yf.Ticker(ticker)
    hist = t.history(period="6mo", interval="1d")
```
- Fetches 6 months of daily OHLCV (Open, High, Low, Close, Volume) data.

```python
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
```
- **SMA (Simple Moving Average)**: Averages the closing price over the last N days.
- SMA_20 = short-term trend (20 trading days ≈ 1 month)
- SMA_50 = medium-term trend (50 trading days ≈ 2.5 months)
- **Trading signal**: If price > SMA → bullish (uptrend). If price < SMA → bearish (downtrend). If SMA_20 crosses above SMA_50 → "golden cross" (bullish). If SMA_20 crosses below SMA_50 → "death cross" (bearish).

```python
    df["Volatility_20d"] = df["Close"].pct_change().rolling(20).std() * (20 ** 0.5) * 100
```
- **Volatility**: Standard deviation of daily returns over 20 days, annualized.
- `pct_change()` calculates daily return (e.g., +1.5%, -0.8%)
- `.rolling(20).std()` calculates the standard deviation over a 20-day window
- `* (20 ** 0.5)` annualizes it (multiplied by sqrt of the rolling period)
- `* 100` converts to percentage

```python
    trend_vs_sma20 = "above" if latest["Close"] > latest["SMA_20"] else "below"
    sma20_slope = (df["SMA_20"].iloc[-1] - df["SMA_20"].iloc[-5]) / 5
```
- Determines if price is above/below the moving average.
- **SMA slope**: Direction the moving average is headed (positive = uptrend, negative = downtrend).

The tool returns a formatted text report that the Quant agent's LLM then interprets.

---

### 3.8 `src/trading_agent/crew.py` — Core Orchestration (THE BRAIN)

This is the most important file. It defines agents, tasks, creates the crew, and runs everything.

**Section 1: Environment Setup (Lines 1-42)**

```python
load_dotenv(_project_root / ".env", override=True)
```
- Loads `.env` file so `MODEL` and `API_KEY` are in `os.environ`.

```python
_PROVIDER_KEY_MAP = {
    "groq":      "GROQ_API_KEY",
    "gemini":    "GOOGLE_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
```
- Maps model prefix to the specific env var that LiteLLM expects.
- LiteLLM checks `os.environ["GROQ_API_KEY"]` when model is `groq/*`.

```python
def _set_provider_key(model: str, api_key: str) -> None:
    prefix = model.split("/")[0].lower()  # "groq/llama-3.1..." → "groq"
    env_var = _PROVIDER_KEY_MAP.get(prefix)  # "groq" → "GROQ_API_KEY"
    if env_var and api_key:
        os.environ[env_var] = api_key  # Set GROQ_API_KEY so LiteLLM finds it
```
- This bridges our single `API_KEY` to whatever provider-specific variable is needed.

**Section 2: TradingAgentCrew Class (Lines 44-130)**

```python
@CrewBase
class TradingAgentCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
```
- `@CrewBase` is a CrewAI decorator that auto-discovers agents and tasks based on the YAML config files.
- Paths are relative to this file's directory (`src/trading_agent/`).

```python
def __init__(self, api_key_override=None):
    model = os.environ.get("MODEL", _llm_model).strip()
    api_key = (api_key_override or os.environ.get("API_KEY") or "").strip()
    
    is_local = model.lower().startswith("ollama/")
    if not api_key and not is_local:
        raise ValueError("API_KEY is not set...")
```
- Resolves which model and key to use.
- Ollama (local) doesn't need an API key, so it's exempted.

```python
    self._llm = LLM(model=model, temperature=0.3, max_tokens=1024, api_key=api_key)
```
- Creates the LLM instance used by all agents.
- **temperature=0.3**: Low randomness → more deterministic, consistent outputs (good for financial analysis, not creative writing).
- **max_tokens=1024**: Limits output length to save API quota and keep responses concise.

**Agent definitions:**

```python
@agent
def researcher(self) -> Agent:
    return Agent(
        config=self.agents_config["researcher"],  # Loads from agents.yaml
        llm=self._llm,
        verbose=True,              # Prints agent's thinking to console
        tools=[NewsFetchTool()],   # This agent can use the news tool
        max_iter=3,                # Max 3 tool-use iterations (prevents infinite loops)
        allow_delegation=False,    # Cannot delegate work to other agents
    )
```
- `config=self.agents_config["researcher"]` loads the role/goal/backstory from agents.yaml.
- `tools=[NewsFetchTool()]` gives this agent access to fetch news.
- `max_iter=3` prevents the agent from calling tools excessively (saves API quota).
- `allow_delegation=False` prevents agents from asking each other for help (keeps the pipeline clean and predictable).

```python
@agent
def portfolio_manager(self) -> Agent:
    return Agent(
        config=self.agents_config["portfolio_manager"],
        llm=self._llm,
        verbose=True,
        max_iter=2,              # Fewer iterations — no tools, just synthesis
        allow_delegation=False,
    )
```
- Portfolio Manager has NO tools — it only reads previous outputs and makes a decision.
- `max_iter=2` because it doesn't need tool calls.

**Task definitions:**

```python
@task
def decision_task(self) -> Task:
    return Task(config=self.tasks_config["decision_task"])
```
- The YAML for `decision_task` includes `context: [research_task, quant_task]`, so CrewAI automatically passes both previous task outputs to the Portfolio Manager.

**Crew definition:**

```python
@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.sequential,  # Run tasks one after another
        verbose=True,
    )
```
- `Process.sequential` = tasks run in order: research_task → quant_task → decision_task.
- Alternative would be `Process.hierarchical` (a manager agent assigns tasks dynamically), but sequential is more predictable and explainable.

**Section 3: run_crew() Function (Lines 131-242)**

```python
def run_crew(ticker, api_key=None, progress_file=None) -> dict:
```
- Main entry point called by both `app.py` and `main.py`.

```python
    inputs = {"ticker": ticker.strip().upper()}
```
- This dict is passed to `crew.kickoff(inputs=inputs)`.
- CrewAI replaces `{ticker}` in task descriptions with the actual value.

**Event handlers for live progress:**

```python
def _on_task_started(_source, event):
    task_step[0] += 1
    progress_path.write_text(str(task_step[0]))
```
- CrewAI fires `TaskStartedEvent` when each task begins.
- We write the step number (1, 2, or 3) to a file.
- The Streamlit app polls this file to update the workflow progress bar.

```python
def _on_task_completed(_source, event):
    output_text = event.output.raw  # Get the agent's text output
    existing[agent_key] = output_text
    outputs_path.write_text(json.dumps(existing))
```
- CrewAI fires `TaskCompletedEvent` when each task finishes.
- We write the output to a JSON file.
- The Streamlit app polls this file to update the sidebar's "Agent Communication" panels in real-time.

**Result parsing:**

```python
    tasks_outputs = []
    for t in result.tasks_output:
        tasks_outputs.append(t.raw)
    
    research_output = tasks_outputs[0]
    quant_output = tasks_outputs[1]
    decision_output = tasks_outputs[2]
```
- `result.tasks_output` is a list of task results in order.
- `.raw` gives the plain text output.

```python
    rec = "HOLD"
    for word in ["BUY", "SELL", "HOLD"]:
        if word in decision_output.upper():
            rec = word
            break
```
- Simple keyword extraction to determine the recommendation.
- Checks in order: BUY first, then SELL, then defaults to HOLD.

---

### 3.9 `main.py` — CLI Entry Point

Simple command-line runner:

```python
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))  # Add src/ to Python path
load_dotenv(root / ".env")             # Load environment variables
```
- Sets up the Python path so `from trading_agent.crew import run_crew` works.

```python
def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"  # Default to AAPL
    result = run_crew(ticker, api_key=api_key)
    print(result["recommendation"])    # BUY / SELL / HOLD
    print(result["research_output"])   # Full researcher report
    print(result["quant_output"])      # Full quant report  
    print(result["decision_output"])   # Full portfolio manager reasoning
```
- Runs the crew and prints all outputs to terminal.

---

### 3.10 `app.py` — Streamlit Dashboard (THE UI)

**Section 1: Setup (Lines 1-30)**

```python
root = Path(__file__).resolve().parent
load_dotenv(root / ".env", override=True)
sys.path.insert(0, str(root / "src"))
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
```
- Loads env, sets up path, disables CrewAI telemetry (which causes threading issues with Streamlit).

```python
_crewai_storage = root / ".crewai_storage"
_crewai_storage.mkdir(parents=True, exist_ok=True)
_crewai_paths.db_storage_path = _writable_db_storage_path
```
- CrewAI stores task outputs in an SQLite DB. By default it uses a system path that may be read-only.
- We redirect it to `.crewai_storage/` in our project directory.

**Section 2: Page Config & CSS (Lines 35-100)**

```python
st.set_page_config(page_title="...", page_icon="💹", layout="wide")
```
- Sets the browser tab title, icon, and uses wide layout.

The CSS block styles:
- `.recommendation-buy/sell/hold` — Color-coded recommendation boxes (green/red/yellow)
- `.agent-block` — Card-style containers for agent outputs
- `.workflow-step.running/done/pending` — Progress steps with icons
- `.comm-panel` — Sidebar panels for live agent communication

**Section 3: Company Info (Lines 105-140)**

```python
@st.cache_data(ttl=3600, show_spinner=False)
def get_company_info(t: str):
```
- `@st.cache_data(ttl=3600)` caches the result for 1 hour to avoid repeated API calls.
- Fetches company name (via yfinance) and logo (via public CDN URLs).
- Tries 3 different logo sources as fallbacks.

**Section 4: Stock Dashboard (Lines 160-210)**

```python
@st.cache_data(ttl=900, show_spinner=False)
def get_stock_dashboard_data(t, period="6mo"):
    tk = yf.Ticker(t)
    hist = tk.history(period=period, interval="1d")
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
```
- Cached for 15 minutes (900s).
- Fetches OHLCV data, computes SMA and volatility.
- Returns dataframe + metrics dict.

```python
    volatility = df["Close"].pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5) * 100
```
- `252 ** 0.5` = annualization factor (252 trading days/year).
- This gives the annualized volatility percentage.

**Charts (Lines 300-350):**

```python
fig = go.Figure()
fig.add_trace(go.Candlestick(...))  # OHLC candles
fig.add_trace(go.Scatter(y=df["SMA_20"], name="SMA 20"))  # Blue line
fig.add_trace(go.Scatter(y=df["SMA_50"], name="SMA 50"))  # Orange line
```
- **Candlestick chart**: Each candle shows Open/High/Low/Close for one day.
  - Green candle = Close > Open (price went up)
  - Red candle = Close < Open (price went down)
- SMA lines overlay to show trends.

```python
vol_colors = ["#28a745" if c >= o else "#dc3545" for c, o in zip(df["Close"], df["Open"])]
fig_vol = go.Figure(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors))
```
- Volume bar chart with same green/red coloring as the candles.

**Section 5: Workflow & Live Updates (Lines 350-475)**

```python
WORKFLOW_STEPS = [
    ("1", "Researcher", "News & sentiment"),
    ("2", "Quant", "Technical analysis"),
    ("3", "Portfolio Manager", "Final decision"),
    ("4", "Explainability", "Full agent reasoning"),
]
```
- Defines sidebar progress steps.

```python
th = threading.Thread(target=_run)
th.start()
while th.is_alive():
    step = int(_progress_file.read_text())
    _workflow_placeholder.markdown(render_workflow_steps(step))
    if _outputs_file.exists():
        _live_outputs = json.loads(_outputs_file.read_text())
        _render_comm_panels(_live_outputs)
    time.sleep(0.6)
```
- **Key mechanism**: The crew runs in a background thread.
- Main thread polls 2 files every 0.6 seconds:
  1. `workflow_progress.txt` — contains step number (1/2/3)
  2. `workflow_progress_outputs.json` — contains agent outputs as they complete
- Updates the sidebar in real-time.

**Section 6: Results Display (Lines 475-530)**

```python
rec = result.get("recommendation", "HOLD")
css_class = "recommendation-buy" if rec == "BUY" else ...
st.markdown(f'<div class="recommendation-box {css_class}">Recommendation: {rec}</div>')
```
- Displays the color-coded recommendation.

```python
col1, col2, col3 = st.columns(3)
with col1:
    st.text_area("Research output", value=result.get("research_output"), disabled=True)
```
- Three-column explainability view: each agent's full output side by side.

**PDF Generation (Lines 210-280):**

```python
class PDF(FPDF):
    def header(self):  # Runs at top of each page
    def footer(self):  # Runs at bottom of each page

pdf.cell(0, 14, f"Recommendation: {rec}", fill=True)  # Colored recommendation box
pdf.multi_cell(0, 5, body_clean)  # Agent outputs (multi-line)
return bytes(pdf.output())
```
- Creates a professional PDF with header, footer, colored recommendation, market overview metrics, and all three agent outputs.

---

## 4. END-TO-END EXECUTION FLOW

### When user clicks "Run committee" in Streamlit:

1. **app.py** reads ticker from sidebar input, gets `_api_key` from `os.environ`
2. **app.py** creates a background thread → calls `run_crew(ticker, api_key, progress_file)`
3. **crew.py** `run_crew()`:
   a. Sets `os.environ["GROQ_API_KEY"]` from the generic `API_KEY`
   b. Creates `TradingAgentCrew` → which creates `LLM(model="groq/llama-3.1-70b-versatile", temperature=0.3)`
   c. Builds the `Crew` with 3 agents and 3 tasks in sequential order
   d. Registers event handlers for `TaskStartedEvent` and `TaskCompletedEvent`
   e. Calls `crew.kickoff(inputs={"ticker": "AAPL"})`
4. **CrewAI internal**:
   a. Replaces `{ticker}` in all task descriptions with `"AAPL"`
   b. Starts `research_task` → sends researcher agent's system prompt + task description to the LLM
   c. LLM decides to call `NewsFetchTool("AAPL")` → gets 15 news headlines back
   d. LLM analyzes headlines → produces sentiment report → task complete
   e. `TaskCompletedEvent` fires → handler writes output to JSON file
   f. Starts `quant_task` → sends quant agent's prompt to LLM
   g. LLM calls `MarketDataTool("AAPL")` → gets OHLCV + indicators back
   h. LLM analyzes data → produces technical report → task complete
   i. Starts `decision_task` → sends PM's prompt + researcher output + quant output as context
   j. LLM synthesizes both → outputs "BUY" + justification → task complete
5. **crew.py** parses `result.tasks_output` → extracts 3 outputs + recommendation → returns dict
6. **app.py** receives dict → displays recommendation box, 3-column XAI view, PDF download button
7. **Meanwhile**: The polling loop in app.py was updating the sidebar every 0.6s with live progress

### Number of LLM API calls (approximate):
- Researcher: 1 call to decide to use tool + 1 call to analyze results = **2 calls**
- Quant: 1 call to decide to use tool + 1 call to analyze results = **2 calls**
- Portfolio Manager: 1 call (no tools, just synthesis) = **1 call**
- **Total: ~5-6 API calls per run** (with max_iter limits preventing more)

---

## 5. KEY DESIGN DECISIONS

| Decision | Rationale |
|----------|-----------|
| **Sequential process** (not hierarchical) | More predictable, explainable → each step is auditable |
| **allow_delegation=False** | Prevents agents from creating extra API calls by asking other agents for help |
| **max_iter=3/2** | CrewAI default is 25 iterations — our limit saves API quota, prevents infinite tool-calling loops |
| **temperature=0.3** | Low randomness for consistent, professional financial analysis |
| **max_tokens=1024** | Keeps outputs concise, saves API quota |
| **Quant has no context from Researcher** | Prevents sentiment bias in technical analysis — they should be independent |
| **Single API_KEY** | User-friendly — only 2 env vars instead of provider-specific keys |
| **Progress via file polling** | Streamlit's threading model doesn't allow direct state sharing — file I/O is the bridge |

---

## 6. PANEL Q&A — EXPECTED QUESTIONS AND ANSWERS

---

### Q1: What is a Multi-Agent System (MAS) and why did you use it here?

**A:** A Multi-Agent System is a system where multiple autonomous agents work together to solve a problem. Each agent has a specific role, tools, and knowledge. I used MAS because in real investment firms, different specialists (research analysts, quant analysts, portfolio managers) handle different aspects. This separation of concerns provides:
1. **Specialization** — each agent focuses on one domain (sentiment vs. technicals vs. decision)
2. **Explainability** — we can see each agent's independent reasoning
3. **Modularity** — we can add/replace agents without rewriting the whole system

---

### Q2: What is CrewAI and how does it work?

**A:** CrewAI is an open-source Python framework for building multi-agent systems. It provides:
- **Agent** class: Wraps an LLM with a role/goal/backstory and optional tools
- **Task** class: Defines what an agent should do, with expected output format
- **Crew** class: Orchestrates multiple agents executing tasks in a defined process (sequential or hierarchical)
- **Tool** class: Gives agents the ability to call external APIs/functions

When you call `crew.kickoff()`, CrewAI iterates through tasks, sends prompts to the LLM, handles tool calls (the LLM outputs a JSON tool call → CrewAI executes it → feeds result back to LLM), and collects outputs.

---

### Q3: What is LiteLLM and why is it needed?

**A:** LiteLLM is a universal LLM routing library. It provides a single API interface that works with 100+ LLM providers. When we set `MODEL=groq/llama-3.1-70b-versatile`, LiteLLM:
1. Parses the `groq/` prefix
2. Routes the call to Groq's API endpoint
3. Translates the response back to a standard format

This lets us switch between Groq (free, fast), OpenAI (GPT-4), Google Gemini, Anthropic Claude, or local Ollama without changing any code — just two lines in `.env`.

---

### Q4: How does the tool-calling mechanism work?

**A:** 
1. We define a tool with a `name`, `description`, and `args_schema` (input validation)
2. When an agent receives a task, CrewAI sends the task description + available tools to the LLM
3. The LLM decides whether to use a tool and outputs a structured tool call (e.g., `{"tool": "Financial News Fetcher", "args": {"ticker": "AAPL"}}`)
4. CrewAI intercepts this, calls our `_run()` method with the arguments
5. The tool executes (e.g., fetches news from Yahoo Finance) and returns a string
6. CrewAI feeds the tool result back to the LLM
7. The LLM then generates its final analysis based on the tool's output

This is called **ReAct (Reasoning + Acting)** — the LLM reasons about what action to take, acts (calls tool), observes the result, then produces its final answer.

---

### Q5: What technical indicators do you use and why?

**A:**
- **SMA 20 (Simple Moving Average, 20 days)**: Short-term trend. If price > SMA20, short-term trend is bullish.
- **SMA 50 (50 days)**: Medium-term trend. Price > SMA50 = medium-term bullish.
- **Golden Cross / Death Cross**: SMA20 crossing above SMA50 = bullish signal (golden cross). Below = bearish (death cross).
- **Annualized Volatility**: Standard deviation of daily returns × √252 (252 trading days/year). High volatility means higher risk.
- **SMA Slope**: The direction the SMA is moving (positive = uptrend, negative = downtrend).

These are standard technical analysis indicators used by actual quant analysts. They're simple enough for explainability but meaningful enough for real analysis.

---

### Q6: How is Explainability (XAI) achieved in your system?

**A:** XAI is achieved through:
1. **Transparent pipeline**: The sequential process (Researcher → Quant → PM) is visible in the workflow sidebar
2. **Full reasoning chain**: Each agent's complete output is displayed in 3 columns — users can read every agent's analysis
3. **No hidden logic**: The Portfolio Manager's justification explicitly references both the sentiment and technical reports
4. **Audit trail**: The PDF report captures everything — recommendation, market data, and all three agent outputs
5. **Live communication**: The sidebar updates in real-time showing what each agent concluded

Unlike a single-prompt approach where the LLM's reasoning is opaque, our system makes every decision step inspectable.

---

### Q7: Why did you choose sequential process instead of hierarchical?

**A:** 
- **Sequential**: Tasks run in a fixed order (R→Q→PM). Predictable, reproducible, easy to explain.
- **Hierarchical**: A "manager" agent dynamically assigns tasks to other agents. More flexible but:
  - Uses more API calls (manager agent needs LLM calls too)
  - Less predictable (manager might change task order)
  - Harder to explain in a thesis/presentation

For an investment committee simulation, sequential is more realistic — in real firms, research reports come first, then quant analysis, then the PM decides. The order is natural and explainable.

---

### Q8: How do you handle API rate limits and quota management?

**A:**
1. **max_iter=3/2**: Limits tool-calling iterations per agent (CrewAI default is 25!)
2. **allow_delegation=False**: Prevents agents from creating extra calls by delegating
3. **max_tokens=1024**: Keeps responses concise, reducing token consumption
4. **temperature=0.3**: Low creativity → fewer retry-worthy outputs
5. **Caching**: Dashboard data is cached for 15 minutes, company info for 1 hour — avoids redundant yfinance calls
6. **Quant has no context from Researcher**: Saves one context-passing step

Total API calls per run: ~5-6 (down from potentially 30+ with defaults).

---

### Q9: How does the live progress/communication work in the dashboard?

**A:** It's a **file-based polling mechanism**:
1. CrewAI runs in a **background thread** (because Streamlit's main thread handles UI)
2. When each task starts, a `TaskStartedEvent` fires → we write the step number to `workflow_progress.txt`
3. When each task completes, a `TaskCompletedEvent` fires → we write the agent's output to `workflow_progress_outputs.json`
4. The main Streamlit thread **polls these files every 0.6 seconds** in a while loop
5. Each poll updates the sidebar's workflow steps and agent communication panels using `st.empty()` placeholders

We use file I/O instead of shared variables because Streamlit's execution model re-runs the entire script on each interaction, making in-memory state sharing unreliable across threads.

---

### Q10: Can this system actually make real trades?

**A:** No, this is a **decision support system**, not an automated trading bot. It:
- Analyzes sentiment and technicals
- Produces a recommendation with justification
- Shows the full reasoning for human review

To make real trades, you'd need: a brokerage API (e.g., Alpaca, Interactive Brokers), risk management systems, position sizing, and regulatory compliance. The thesis focuses on **explainable decision-making**, not execution.

---

### Q11: What is yfinance and is it reliable for real-time data?

**A:** yfinance is a Python library that fetches financial data from Yahoo Finance. It provides:
- Historical OHLCV data (Open, High, Low, Close, Volume)
- Company information (name, sector, market cap)
- News headlines

**Limitations**: 
- Data is delayed (15-20 min for free tier)
- News doesn't include full article text, only headlines
- Occasional API throttling
- Not suitable for high-frequency trading

For our use case (daily analysis + recommendation), it's sufficient. For production-grade systems, you'd use Bloomberg Terminal, Reuters, or paid APIs like Alpha Vantage.

---

### Q12: What is the role of Pydantic in this project?

**A:** Pydantic is used for **data validation**:
- `MarketDataInput` and `NewsFetchInput` are Pydantic models that define the input schema for our tools
- When the LLM calls a tool, CrewAI validates the arguments against these schemas before executing
- This prevents runtime errors (e.g., missing ticker, wrong data type)
- The `Field(description=...)` metadata is also sent to the LLM so it knows what arguments to pass

---

### Q13: How does the PDF report generation work?

**A:** We use `fpdf2` (a pure Python PDF library):
1. Create a custom `PDF` class inheriting from `FPDF` with custom header/footer
2. Add a title page with ticker name and timestamp
3. Draw a **color-coded recommendation box** (green=BUY, red=SELL, yellow=HOLD)
4. Add market overview metrics (price, change, volatility, volume)
5. Add each agent's full output as separate sections
6. Return the PDF as `bytes` for Streamlit's download button

The PDF serves as an **audit trail** — users can download and share the analysis.

---

### Q14: What is the role of temperature and max_tokens in your LLM configuration?

**A:**
- **temperature (0.3)**: Controls randomness. 0 = deterministic, 1 = creative. We use 0.3 because:
  - Financial analysis should be consistent and factual
  - Too high → creative/unreliable outputs
  - Too low → might be too rigid and miss nuances
  - 0.3 is a balance between consistency and natural language flow

- **max_tokens (1024)**: Maximum output length. We limit it because:
  - Agents should be concise (2-4 paragraph reports, not essays)
  - Saves API quota (billed per token)
  - Forces the LLM to prioritize important information

---

### Q15: How would you extend this system in the future?

**A:**
1. **More agents**: Risk assessment agent, macroeconomic analyst, options strategist
2. **NLP sentiment scoring**: TextBlob/VADER for quantitative sentiment scores (not just LLM judgment)
3. **Portfolio management**: Multi-ticker support, portfolio optimization (Markowitz), correlation analysis
4. **Backtesting**: Run the system on historical data to measure recommendation accuracy
5. **Real-time data**: WebSocket feeds for intraday analysis
6. **Reinforcement Learning**: Let agents learn from past recommendation outcomes
7. **Brokerage integration**: Alpaca API for paper trading / execution

---

### Q16: What are the limitations of your system?

**A:**
1. **LLM dependency**: Analysis quality depends on the LLM's financial knowledge (which can hallucinate)
2. **Data delay**: yfinance has 15-20 min delay — not suitable for day trading
3. **No backtesting**: We haven't validated recommendation accuracy on historical data
4. **Single-stock**: Analyzes one stock at a time, no portfolio-level optimization
5. **No fundamental analysis**: Doesn't consider P/E ratios, earnings, balance sheets
6. **API rate limits**: Free LLM tiers have limited quota
7. **News headline-only**: No full article text analysis — sentiment is based on titles only

---

### Q17: What is the difference between your approach and traditional algorithmic trading?

**A:**
| Aspect | Traditional Algo Trading | Our MAS Approach |
|--------|-------------------------|------------------|
| Decision | Mathematical rules (if RSI < 30, buy) | LLM reasoning with multiple inputs |
| Explainability | Rule-based (clear but rigid) | Natural language justification |
| Adaptability | Fixed rules, needs re-coding | LLM adapts to new market narratives |
| Data | Only numerical | Numerical + qualitative (news sentiment) |
| Transparency | Transparent rules but no "why" | Full reasoning chain visible |
| Speed | Milliseconds | Several seconds (LLM inference) |

Our system combines the best of both: data-driven analysis (Quant agent) + qualitative reasoning (Researcher agent) + human-readable justification (Portfolio Manager).

---

### Q18: What happens if the agents disagree (e.g., Researcher says bearish, Quant says bullish)?

**A:** This is actually a **feature** of the system. The Portfolio Manager explicitly handles conflicts:
- The PM's task description says: "Consider risk vs. reward and any conflicts between sentiment and technicals"
- When inputs conflict, the PM will note the disagreement in its justification
- Example output: "Despite bullish technicals (price above SMA20/50), bearish news sentiment regarding regulatory concerns suggests caution — recommending HOLD until sentiment clarifies"
- This is exactly how real investment committees work — conflicting opinions are synthesized, not ignored

---

### Q19: Explain the candlestick chart in the dashboard.

**A:** A candlestick chart shows four data points per time period (day):
- **Open**: Price at market open
- **Close**: Price at market close
- **High**: Highest price during the day
- **Low**: Lowest price during the day

Visual representation:
```
     High
      │
  ┌───┤  (Green = Close > Open, price went UP)
  │   │  Body = Open to Close range
  └───┤  
      │
     Low
```

Green candle = bullish day (closed higher than opened)
Red candle = bearish day (closed lower than opened)
The SMA lines help identify trends over time.

---

### Q20: How is this project different from ChatGPT analyzing stocks?

**A:**
1. **Structured pipeline**: ChatGPT is a single conversation. Our system has 3 specialized agents with defined roles, constraints, and tools.
2. **Real data**: Our agents fetch LIVE market data and news via tools. ChatGPT uses training data (potentially outdated).
3. **Separation of concerns**: Researcher can't see price data; Quant can't see news. This prevents cross-contamination.
4. **Reproducibility**: The sequential pipeline with fixed parameters produces more consistent results.
5. **XAI**: Every step is individually visible. In ChatGPT, you get one response — the internal reasoning is hidden.
6. **Audit trail**: PDF report preserves the full analysis for compliance/review.

---

### BONUS: Quick one-line answers for rapid-fire questions

| Question | Answer |
|----------|--------|
| What language? | Python 3.11+ |
| What framework for agents? | CrewAI |
| What LLM? | Configurable — currently Groq/LLaMA-3.1-70B |
| What for charts? | Plotly (candlestick + volume) |
| What for frontend? | Streamlit |
| What for data? | yfinance (Yahoo Finance) |
| How many agents? | 3 (Researcher, Quant, Portfolio Manager) |
| How many API calls per run? | ~5-6 |
| Is it real-time? | Near real-time (15-min delay from yfinance) |
| Can it auto-trade? | No, it's a decision support system |
| What is XAI? | Explainable AI — showing the reasoning chain, not just the answer |
| What is SMA? | Simple Moving Average — mean of closing prices over N days |
| What is volatility? | Standard deviation of returns, annualized — measures risk |
| What is ReAct? | Reasoning + Acting — LLM decides to call a tool, observes result, then answers |
| What is sequential process? | Tasks run in fixed order, one after another |
| What is a golden cross? | SMA20 crosses above SMA50 — bullish signal |
| What is a death cross? | SMA20 crosses below SMA50 — bearish signal |
