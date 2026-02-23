"""
ReAct agent — Reason + Act pattern with explicit Thought/Action/Observation loop.

The LLM emits structured "Thought:" / "Action:" / "Action Input:" / "Final Answer:"
blocks, which are parsed to drive tool execution.

Reference: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", 2022.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentHooks, BaseAgent, Message, Tool, ToolCall

logger = logging.getLogger(__name__)

REACT_SYSTEM = """\
You are an intelligent AI assistant that solves tasks step by step using tools.

Use the following format EXACTLY:

Thought: Think about what you need to do next.
Action: The tool to call (one of: {tool_names}).
Action Input: {{"param1": "value1", "param2": "value2"}}
Observation: (result of the tool will appear here)

Repeat Thought/Action/Action Input/Observation as many times as needed.
When you have the final answer, write:

Thought: I now know the final answer.
Final Answer: <your complete answer here>

Available tools:
{tool_descriptions}
"""


class ReActAgent(BaseAgent):
    """
    Implements the ReAct reasoning loop.

    Parses LLM text output for structured Thought/Action/Final Answer blocks.

    Example:
        agent = ReActAgent(
            tools=[WebSearchTool(), CalculatorTool()],
            llm=groq_client,
        )
        result = agent.run("What is 15% of the population of Tokyo?")
        print(result.answer)
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        llm: Optional[Any] = None,
        max_iterations: int = 10,
        verbose: bool = True,
        hooks: Optional[AgentHooks] = None,
    ) -> None:
        super().__init__(
            tools=tools,
            llm=llm,
            system_prompt=self._build_system_prompt(tools or []),
            max_iterations=max_iterations,
            verbose=verbose,
            hooks=hooks,
        )

    # ------------------------------------------------------------------
    # ReAct plan parsing
    # ------------------------------------------------------------------

    async def _plan(self, messages: List[Dict]) -> List[ToolCall]:
        raw = await self._generate(messages)
        # Append the LLM text to message history as assistant turn
        self._history.append(Message(role="assistant", content=raw))

        if "Final Answer:" in raw:
            return []  # signal: done

        # Extract Action / Action Input
        action_match = re.search(r"Action:\s*(.+)", raw)
        input_match = re.search(r"Action Input:\s*(\{.*?\}|\".+\"|\S+)", raw, re.DOTALL)

        if not action_match:
            return []  # no tool call detected → generate final answer

        tool_name = action_match.group(1).strip()
        try:
            raw_input = input_match.group(1).strip() if input_match else "{}"
            arguments: Dict[str, Any] = json.loads(raw_input)
        except (json.JSONDecodeError, AttributeError):
            arguments = {"input": input_match.group(1).strip() if input_match else ""}

        return [
            ToolCall(
                name=tool_name,
                arguments=arguments,
                call_id=f"react_{len(self._history)}",
            )
        ]

    async def _generate(self, messages: List[Dict]) -> str:
        if self._llm is None:
            raise RuntimeError("No LLM client configured.")
        return self._llm.chat(messages)

    # ------------------------------------------------------------------
    # Override run loop to extract final answer from ReAct text
    # ------------------------------------------------------------------

    async def arun(self, query: str, reset_history: bool = True):  # type: ignore[override]
        result = await super().arun(query, reset_history=reset_history)
        # Extract clean final answer from ReAct formatted text
        if result.steps and result.steps[-1].final_answer:
            raw = result.steps[-1].final_answer
            fa_match = re.search(r"Final Answer:\s*(.+)", raw, re.DOTALL)
            if fa_match:
                result = result.__class__(
                    answer=fa_match.group(1).strip(),
                    steps=result.steps,
                    total_iterations=result.total_iterations,
                    elapsed_s=result.elapsed_s,
                )
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt(tools: List[Tool]) -> str:
        tool_names = ", ".join(t.name for t in tools) or "none"
        tool_descriptions = (
            "\n".join(f"  • {t.name}: {t.description}" for t in tools)
            or "  (no tools provided)"
        )
        return REACT_SYSTEM.format(
            tool_names=tool_names,
            tool_descriptions=tool_descriptions,
        )
