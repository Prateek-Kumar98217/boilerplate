"""
Agent Swarm with handoffs — inspired by OpenAI Swarm.

Agents pass control to each other via explicit ``handoff(next_agent_name)`` returns.
Each agent is lightweight and focused; the swarm loop routes until one agent
produces a final answer.

Reference: https://github.com/openai/swarm
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Swarm agent
# ---------------------------------------------------------------------------


@dataclass
class SwarmAgent:
    """A lightweight specialised agent in a swarm."""

    name: str
    description: str
    llm: Any
    system_prompt: str = ""
    tools: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.system_prompt:
            self.system_prompt = (
                f"You are {self.name}. {self.description}\n\n"
                "If you cannot fully complete the task, respond with:\n"
                'HANDOFF: {"agent": "<agent_name>", "reason": "<why>"}\n\n'
                "Otherwise, provide the complete answer directly."
            )

    def chat(self, messages: List[Dict]) -> str:
        full = [{"role": "system", "content": self.system_prompt}] + messages
        return self.llm.chat(full)


# ---------------------------------------------------------------------------
# Swarm orchestrator
# ---------------------------------------------------------------------------


@dataclass
class SwarmResult:
    answer: str
    agent_path: List[str]
    total_handoffs: int
    elapsed_s: float


class AgentSwarm:
    """
    Routes a task through a swarm of agents via explicit handoffs.

    Example:
        swarm = AgentSwarm(
            agents=[triage_agent, researcher_agent, coder_agent],
            entry="triage",
            max_handoffs=5,
        )
        result = swarm.run("Write a Python quicksort and explain it.")
        print(result.answer)
    """

    HANDOFF_RE = re.compile(r"HANDOFF:\s*(\{.*?\})", re.DOTALL)

    def __init__(
        self,
        agents: List[SwarmAgent],
        entry: str,
        max_handoffs: int = 10,
        verbose: bool = True,
    ) -> None:
        self._agents: Dict[str, SwarmAgent] = {a.name: a for a in agents}
        self._entry = entry
        self._max_handoffs = max_handoffs
        self._verbose = verbose
        if entry not in self._agents:
            raise ValueError(f"Entry agent '{entry}' not in agents list.")

    def run(self, task: str) -> SwarmResult:
        t0 = time.perf_counter()
        messages: List[Dict] = [{"role": "user", "content": task}]
        agent_path: List[str] = []
        current = self._entry
        handoffs = 0

        while handoffs <= self._max_handoffs:
            agent = self._agents.get(current)
            if agent is None:
                logger.error("Swarm: unknown agent '%s'", current)
                break

            agent_path.append(current)
            if self._verbose:
                logger.info("Swarm: running '%s' (handoff %d)", current, handoffs)

            response = agent.chat(messages)
            messages.append({"role": "assistant", "content": response})

            match = self.HANDOFF_RE.search(response)
            if not match:
                # Agent produced a final answer
                return SwarmResult(
                    answer=response,
                    agent_path=agent_path,
                    total_handoffs=handoffs,
                    elapsed_s=time.perf_counter() - t0,
                )

            # Parse handoff
            try:
                handoff_data = json.loads(match.group(1))
                next_agent = handoff_data.get("agent", "")
                reason = handoff_data.get("reason", "")
            except json.JSONDecodeError:
                logger.warning("Swarm: could not parse handoff JSON")
                break

            if self._verbose:
                logger.info("Swarm: handoff %s → %s (%s)", current, next_agent, reason)

            if next_agent not in self._agents:
                logger.error("Swarm: handoff to unknown agent '%s'", next_agent)
                break

            messages.append(
                {
                    "role": "user",
                    "content": f"[Handoff from {current}]: {reason}\n\nContinue with: {task}",
                }
            )
            current = next_agent
            handoffs += 1

        # Fallback: ask the last agent for its best answer
        agent = self._agents.get(current, self._agents[self._entry])
        fallback = agent.chat(
            messages
            + [{"role": "user", "content": "Please give your best final answer now."}]
        )
        return SwarmResult(
            answer=fallback,
            agent_path=agent_path,
            total_handoffs=handoffs,
            elapsed_s=time.perf_counter() - t0,
        )
