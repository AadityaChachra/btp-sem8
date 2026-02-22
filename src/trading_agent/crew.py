"""
Digital Investment Office — Crew definition.
Sequential pipeline: Researcher → Quant → Portfolio Manager.
"""

import os
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from trading_agent.tools import MarketDataTool, NewsFetchTool

# Default key and model loaded at import (can be overridden by run_crew(api_key=...))
_crew_dir = Path(__file__).resolve().parent
_project_root = _crew_dir.parent.parent
if (_project_root / ".env").exists():
    load_dotenv(_project_root / ".env", override=True)
_groq_key = (os.environ.get("GROQ_API_KEY") or "").strip().strip('"').strip("'").strip("\r\n")
# Model: use MODEL env var or default to Groq. Examples: groq/llama-3.1-70b-versatile, openai/gpt-4o, anthropic/claude-3-5-sonnet-20241022, gemini/gemini-2.0-flash
_default_model = "groq/llama-3.1-70b-versatile"
_llm_model = (os.environ.get("MODEL") or _default_model).strip()


@CrewBase
class TradingAgentCrew:
    """Autonomous Trading & Portfolio Optimization Agent — Multi-Agent Investment Committee."""

    # CrewBase uses these as paths relative to this file's directory (trading_agent/).
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, api_key_override: str | None = None):
        model = (os.environ.get("MODEL", _llm_model) or _default_model).strip()
        if model.upper().startswith("MODEL="):
            model = model.split("=", 1)[-1].strip()  # fix typo: MODEL=ollama/... -> ollama/...
        model = model or _default_model
        is_groq = model.lower().startswith("groq/")
        api_key = None
        if is_groq:
            key = (api_key_override or os.environ.get("GROQ_API_KEY") or _groq_key or "").strip().strip('"').strip("'").strip("\r\n")
            if not key:
                raise ValueError(
                    "GROQ_API_KEY is not set. Add it to .env (get one at https://console.groq.com/keys)"
                )
            os.environ["GROQ_API_KEY"] = key
            api_key = key
        llm_kwargs = dict(
            model=model,
            temperature=0.3,
            max_tokens=2048,
        )
        if api_key:
            llm_kwargs["api_key"] = api_key
        self._llm = LLM(**llm_kwargs)

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            llm=self._llm,
            verbose=True,
            tools=[NewsFetchTool()],
        )

    @agent
    def quant(self) -> Agent:
        return Agent(
            config=self.agents_config["quant"],
            llm=self._llm,
            verbose=True,
            tools=[MarketDataTool()],
        )

    @agent
    def portfolio_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["portfolio_manager"],
            llm=self._llm,
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
        )

    @task
    def quant_task(self) -> Task:
        return Task(config=self.tasks_config["quant_task"])

    @task
    def decision_task(self) -> Task:
        return Task(config=self.tasks_config["decision_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


def run_crew(ticker: str, api_key: str | None = None, progress_file: str | Path | None = None) -> dict:
    """
    Run the investment committee crew for a given ticker.
    Returns dict with 'recommendation', 'justification', 'research_output', 'quant_output', 'decision_output'.
    If api_key is provided, it is used and set in os.environ so LiteLLM sees it.
    If progress_file is provided, writes current step (1, 2, 3) on task start for UI progress.
    """
    if api_key:
        api_key = api_key.strip().strip('"').strip("'").strip("\r\n")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
    inputs = {"ticker": ticker.strip().upper()}
    progress_path = Path(progress_file) if progress_file else None
    task_step = [0]  # mutable so handler can increment

    def _on_task_started(_source: Any, event: Any) -> None:
        task_step[0] += 1
        if progress_path:
            try:
                progress_path.write_text(str(task_step[0]), encoding="utf-8")
            except Exception:
                pass

    if progress_path:
        try:
            progress_path.write_text("0", encoding="utf-8")
        except Exception:
            pass
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.task_events import TaskStartedEvent
        crewai_event_bus.register_handler(TaskStartedEvent, _on_task_started)
    try:
        crew_instance = TradingAgentCrew(api_key_override=api_key).crew()
        result = crew_instance.kickoff(inputs=inputs)
    finally:
        if progress_path:
            try:
                progress_path.write_text("3", encoding="utf-8")  # all done
            except Exception:
                pass

    # CrewAI result may have .tasks_output (list of task results with .raw) or .raw (final only)
    tasks_outputs = []
    if hasattr(result, "tasks_output") and result.tasks_output:
        for t in result.tasks_output:
            tasks_outputs.append(getattr(t, "raw", str(t)))
    elif hasattr(result, "raw"):
        tasks_outputs = [result.raw]

    research_output = tasks_outputs[0] if len(tasks_outputs) > 0 else ""
    quant_output = tasks_outputs[1] if len(tasks_outputs) > 1 else ""
    decision_output = tasks_outputs[2] if len(tasks_outputs) > 2 else (result.raw if hasattr(result, "raw") else "")

    # Parse final recommendation from decision_output
    rec = "HOLD"
    for word in ["BUY", "SELL", "HOLD"]:
        if word in (decision_output or "").upper():
            rec = word
            break

    return {
        "recommendation": rec,
        "justification": decision_output or "",
        "research_output": research_output,
        "quant_output": quant_output,
        "decision_output": decision_output or "",
        "ticker": inputs["ticker"],
    }
