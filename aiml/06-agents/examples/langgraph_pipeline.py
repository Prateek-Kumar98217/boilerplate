"""
LangGraph pipeline example — sequential researcher → writer workflow.
Run: python examples/langgraph_pipeline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestration.langgraph_workflow import (
    AgentState,
    build_sequential_pipeline,
    make_llm_node,
)


class _MockLLM:
    def chat(self, messages):
        role = messages[0]["content"][:30]
        return f"[{role}...] Mock LLM response for: {messages[-1]['content'][:60]}"


def main():
    try:
        from langgraph.graph import END  # noqa: F401
    except ImportError:
        print("langgraph not installed. Run: pip install langgraph")
        return

    llm = _MockLLM()

    researcher_node = make_llm_node(
        "researcher",
        llm,
        "You are a researcher. Find facts about the topic given.",
    )
    writer_node = make_llm_node(
        "writer",
        llm,
        "You are a technical writer. Using the researcher's findings, write a short blog post.",
    )

    pipeline = build_sequential_pipeline(
        [
            ("researcher", researcher_node),
            ("writer", writer_node),
        ]
    )

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": "Tell me about FAISS."}],
        "task": "Write a blog post about FAISS.",
        "current_agent": "",
        "agent_outputs": {},
        "final_answer": None,
        "error": None,
        "iterations": 0,
    }

    result = pipeline.invoke(initial_state)
    print("Researcher output:", result["agent_outputs"].get("researcher", ""))
    print("\nWriter output:", result["agent_outputs"].get("writer", ""))


if __name__ == "__main__":
    main()
