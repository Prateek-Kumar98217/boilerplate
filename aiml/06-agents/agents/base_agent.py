"""
BaseAgent — fully-featured customisable agent foundation.

Capabilities:
  • Pluggable tool registry
  • Modular memory (buffer | summary | vector)
  • Multi-turn conversation history
  • Configurable planning mode (none | structured | tree-of-thought)
  • Retry + error handling on tool calls
  • Streaming support (async generator)
  • Hooks (on_start, on_tool_call, on_tool_result, on_end)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    call_id: str = ""


@dataclass
class ToolResult:
    call_id: str
    name: str
    result: Any
    error: Optional[str] = None
    elapsed_s: float = 0.0


@dataclass
class AgentStep:
    iteration: int
    thought: str
    tool_calls: List[ToolCall]
    tool_results: List[ToolResult]
    final_answer: Optional[str] = None


@dataclass
class AgentResult:
    answer: str
    steps: List[AgentStep]
    total_iterations: int
    elapsed_s: float


# ---------------------------------------------------------------------------
# Tool interface
# ---------------------------------------------------------------------------


class Tool(ABC):
    """Base class for all agent tools."""

    name: str = "tool"
    description: str = "A tool."
    parameters_schema: Dict[str, Any] = {}  # JSON schema for parameters

    @abstractmethod
    def run(self, **kwargs: Any) -> Any: ...

    async def arun(self, **kwargs: Any) -> Any:
        """Async wrapper — override for native async tools."""
        return await asyncio.to_thread(self.run, **kwargs)

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


@dataclass
class AgentHooks:
    on_start: Optional[Callable[[str], None]] = None
    on_tool_call: Optional[Callable[[ToolCall], None]] = None
    on_tool_result: Optional[Callable[[ToolResult], None]] = None
    on_iteration: Optional[Callable[[AgentStep], None]] = None
    on_end: Optional[Callable[[AgentResult], None]] = None


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    Fully-featured agent base class.

    Subclasses implement:
        - ``_plan(messages)``    → list of ToolCalls (or empty for direct answer)
        - ``_generate(messages)`` → str final answer from the LLM

    Example:
        class MyAgent(BaseAgent):
            ...
        agent = MyAgent(tools=[WebSearchTool(), CalculatorTool()], llm=groq_client)
        result = agent.run("What is the capital of France and its population squared?")
    """

    STOP_TOKEN = "<FINAL_ANSWER>"

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        llm: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = True,
        retry_on_tool_error: int = 1,
        hooks: Optional[AgentHooks] = None,
    ) -> None:
        self._tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        self._llm = llm
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._max_iterations = max_iterations
        self._verbose = verbose
        self._retry_on_tool_error = retry_on_tool_error
        self._hooks = hooks or AgentHooks()
        self._history: List[Message] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str, reset_history: bool = True) -> AgentResult:
        """Synchronous entry point."""
        return asyncio.run(self.arun(query, reset_history=reset_history))

    async def arun(self, query: str, reset_history: bool = True) -> AgentResult:
        """Async entry point."""
        if reset_history:
            self._history = []

        if self._hooks.on_start:
            self._hooks.on_start(query)

        t0 = time.perf_counter()
        self._history.append(Message(role="user", content=query))
        steps: List[AgentStep] = []

        for iteration in range(1, self._max_iterations + 1):
            messages = self._build_messages()
            tool_calls = await self._plan(messages)

            if not tool_calls:
                answer = await self._generate(messages)
                step = AgentStep(
                    iteration=iteration,
                    thought="",
                    tool_calls=[],
                    tool_results=[],
                    final_answer=answer,
                )
                steps.append(step)
                if self._hooks.on_iteration:
                    self._hooks.on_iteration(step)
                self._history.append(Message(role="assistant", content=answer))
                result = AgentResult(
                    answer=answer,
                    steps=steps,
                    total_iterations=iteration,
                    elapsed_s=time.perf_counter() - t0,
                )
                if self._hooks.on_end:
                    self._hooks.on_end(result)
                return result

            # Execute tools
            results = []
            for tc in tool_calls:
                tr = await self._call_tool(tc)
                results.append(tr)
                self._history.append(
                    Message(
                        role="tool",
                        content=(
                            str(tr.result) if not tr.error else f"ERROR: {tr.error}"
                        ),
                        tool_name=tc.name,
                        tool_call_id=tc.call_id,
                    )
                )

            step = AgentStep(
                iteration=iteration,
                thought="",
                tool_calls=tool_calls,
                tool_results=results,
            )
            steps.append(step)
            if self._hooks.on_iteration:
                self._hooks.on_iteration(step)

        # Max iterations reached — ask for best answer
        answer = await self._generate(self._build_messages())
        result = AgentResult(
            answer=answer,
            steps=steps,
            total_iterations=self._max_iterations,
            elapsed_s=time.perf_counter() - t0,
        )
        if self._hooks.on_end:
            self._hooks.on_end(result)
        return result

    async def astream(self, query: str) -> AsyncGenerator[str, None]:
        """Yield partial answer tokens. Requires async-streaming LLM client."""
        self._history.append(Message(role="user", content=query))
        async for token in self._stream_generate(self._build_messages()):
            yield token

    def add_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def reset(self) -> None:
        self._history = []

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    async def _plan(self, messages: List[Dict]) -> List[ToolCall]: ...

    async def _generate(self, messages: List[Dict]) -> str:
        """Generate a final answer from the LLM."""
        if self._llm is None:
            raise RuntimeError("No LLM client configured.")
        return self._llm.chat(messages)

    async def _stream_generate(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        raise NotImplementedError("Streaming not implemented for this agent.")
        # Make mypy happy
        yield ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self) -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": self._system_prompt}]
        for m in self._history:
            msgs.append({"role": m.role, "content": m.content})
        return msgs

    async def _call_tool(self, tc: ToolCall) -> ToolResult:
        tool = self._tools.get(tc.name)
        if tool is None:
            return ToolResult(
                call_id=tc.call_id,
                name=tc.name,
                result=None,
                error=f"Tool '{tc.name}' not found.",
            )
        t0 = time.perf_counter()
        if self._hooks.on_tool_call:
            self._hooks.on_tool_call(tc)
        for attempt in range(self._retry_on_tool_error + 1):
            try:
                result = await tool.arun(**tc.arguments)
                tr = ToolResult(
                    call_id=tc.call_id,
                    name=tc.name,
                    result=result,
                    elapsed_s=time.perf_counter() - t0,
                )
                if self._hooks.on_tool_result:
                    self._hooks.on_tool_result(tr)
                if self._verbose:
                    logger.info(
                        "Tool '%s' → %s (%.2fs)",
                        tc.name,
                        str(result)[:120],
                        tr.elapsed_s,
                    )
                return tr
            except Exception as exc:
                logger.warning(
                    "Tool '%s' attempt %d failed: %s", tc.name, attempt + 1, exc
                )
                if attempt == self._retry_on_tool_error:
                    return ToolResult(
                        call_id=tc.call_id,
                        name=tc.name,
                        result=None,
                        error=str(exc),
                        elapsed_s=time.perf_counter() - t0,
                    )
        raise RuntimeError("Unreachable")

    def _default_system_prompt(self) -> str:
        return (
            "You are a helpful AI assistant with access to tools. "
            "Use tools when necessary to answer accurately. "
            "When you have enough information, provide a clear, concise final answer."
        )
