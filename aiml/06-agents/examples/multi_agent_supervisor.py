"""
Multi-agent supervisor example.
A supervisor routes tasks to specialised researcher and coder workers.
Run: python examples/multi_agent_supervisor.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.react_agent import ReActAgent
from orchestration.supervisor import SupervisorAgent, Worker
from tools.builtin_tools import CalculatorTool, WikipediaTool


class _MockLLM:
    def complete(self, prompt):
        if "JSON list" in prompt:
            return '["researcher"]'
        return "Here is the synthesised answer based on all worker outputs."

    def chat(self, messages):
        return "Thought: Done.\nFinal Answer: Mock answer."


async def main():
    llm = _MockLLM()

    researcher = ReActAgent(tools=[WikipediaTool()], llm=llm, verbose=False)
    coder = ReActAgent(tools=[CalculatorTool()], llm=llm, verbose=False)

    supervisor = SupervisorAgent(
        workers=[
            Worker(
                "researcher", "Looks up facts and background information", researcher
            ),
            Worker("coder", "Solves math problems and writes Python code", coder),
        ],
        llm=llm,
        verbose=True,
    )

    tasks = [
        "Research the Eiffel Tower and calculate how tall it is in feet (1m = 3.281ft).",
        "Write Python code to generate the first 10 Fibonacci numbers.",
    ]
    for task in tasks:
        print(f"\nTask: {task}")
        answer = await supervisor.run(task)
        print(f"Final: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
