"""
ToolCallingAgent — uses OpenAI-compatible function/tool calling API.

The LLM decides which tools to call and returns structured JSON arguments.
Supports parallel tool calls in a single LLM turn (where the API allows it).

Works with: Groq, OpenAI, Cerebras (any OpenAI-compatible API).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentHooks, BaseAgent, Message, Tool, ToolCall

logger = logging.getLogger(__name__)

TOOL_CALLING_SYSTEM = """\
You are a smart AI assistant with access to tools.
Use tools whenever they can help you answer more accurately.
When you have enough information, reply directly to the user with a clear, helpful answer.
Do NOT mention tool names or JSON in your final answer unless explicitly asked.
"""


class ToolCallingAgent(BaseAgent):
    """
    Agent that drives tool use via the LLM's native function/tool calling API.

    Unlike ReAct (which parses free-form text), this agent sends the tool schemas
    to the LLM and parses the structured ``tool_calls`` from the API response.

    Requires an ``OpenAIChatClient`` or compatible client that exposes
    ``chat_with_tools(messages, tools)`` returning a raw API response dict.

    Example:
        client = OpenAIChatClient(api_key="...", model="gpt-4o-mini")
        agent = ToolCallingAgent(tools=[WebSearchTool(), CalculatorTool()], llm=client)
        result = agent.run("Search for the latest PyTorch version and compute its square.")
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        llm: Optional[Any] = None,
        max_iterations: int = 10,
        verbose: bool = True,
        parallel_tool_calls: bool = True,
        hooks: Optional[AgentHooks] = None,
    ) -> None:
        super().__init__(
            tools=tools,
            llm=llm,
            system_prompt=TOOL_CALLING_SYSTEM,
            max_iterations=max_iterations,
            verbose=verbose,
            hooks=hooks,
        )
        self._parallel = parallel_tool_calls
        self._tool_schemas = [t.to_openai_schema() for t in (tools or [])]

    # ------------------------------------------------------------------
    # Core plan: ask LLM with tool schemas → parse tool_calls back
    # ------------------------------------------------------------------

    async def _plan(self, messages: List[Dict]) -> List[ToolCall]:
        if not self._tool_schemas or self._llm is None:
            return []

        raw_response = self._llm.chat_with_tools(
            messages=messages,
            tools=self._tool_schemas,
            parallel_tool_calls=self._parallel,
        )
        return self._parse_tool_calls(raw_response)

    async def _generate(self, messages: List[Dict]) -> str:
        """Called when no tool calls are returned — get final answer."""
        if self._llm is None:
            raise RuntimeError("No LLM configured.")
        return self._llm.chat(messages)

    # ------------------------------------------------------------------
    # Parse OpenAI-style tool_calls list
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(response: Any) -> List[ToolCall]:
        calls = []
        try:
            tool_calls = response.choices[0].message.tool_calls or []
            for tc in tool_calls:
                arguments = json.loads(tc.function.arguments or "{}")
                calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=arguments,
                        call_id=tc.id or str(uuid.uuid4()),
                    )
                )
        except (AttributeError, KeyError, json.JSONDecodeError) as exc:
            logger.debug("Could not parse tool calls: %s", exc)
        return calls


# ---------------------------------------------------------------------------
# OpenAI-compatible client wrapper that exposes chat_with_tools()
# ---------------------------------------------------------------------------


class OpenAIChatClient:
    """
    Thin OpenAI-compatible client that supports both plain chat and tool calling.

    Compatible with Groq, Cerebras, vLLM, LM Studio (any OpenAI-compatible API).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai required: pip install openai")
        self._client = OpenAI(
            api_key=api_key, **({"base_url": base_url} if base_url else {})
        )
        self._model = model

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content

    def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        parallel_tool_calls: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Returns the raw API response (message.tool_calls accessible)."""
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=parallel_tool_calls,
            **kwargs,
        )
