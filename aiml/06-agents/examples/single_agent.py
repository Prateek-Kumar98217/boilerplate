"""
Single agent example — ReAct agent with web search + calculator.
Run: python examples/single_agent.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.react_agent import ReActAgent
from tools.builtin_tools import CalculatorTool, WikipediaTool


class _MockLLM:
    """Minimal mock LLM so the example runs without API keys."""

    def chat(self, messages):
        last = messages[-1]["content"]
        if "Thought:" in last or "Action:" in last:
            return "Thought: I have all the info I need.\nFinal Answer: 42"
        return 'Thought: I need to check Wikipedia.\nAction: wikipedia\nAction Input: {"query": "Eiffel Tower"}'


def main():
    # ── Swap MockLLM for a real client ─────────────────────────────────────
    # from generation.chain import create_llm_client  # (relative path from 05-rag)
    # llm = create_llm_client("groq", api_key=os.getenv("GROQ_API_KEY",""))
    llm = _MockLLM()

    agent = ReActAgent(
        tools=[WikipediaTool(), CalculatorTool()],
        llm=llm,
        max_iterations=6,
        verbose=True,
    )

    questions = [
        "What is the height of the Eiffel Tower?",
        "Calculate 2**16 + sqrt(256)",
    ]
    for q in questions:
        print(f"\nQuery: {q}")
        result = agent.run(q)
        print(f"Answer: {result.answer}")
        print(f"Iterations: {result.total_iterations}  Time: {result.elapsed_s:.2f}s")


if __name__ == "__main__":
    main()
