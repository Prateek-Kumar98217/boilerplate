"""
Supervisor agent — orchestrates specialised worker agents.

The supervisor receives the task, routes it to the right worker(s), collects
results, and synthesises a final answer. Supports both sequential and parallel
worker execution.

Pattern: OpenAI's "Swarm" paper and LangGraph multi-agent supervisor recipe.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker descriptor
# ---------------------------------------------------------------------------


@dataclass
class Worker:
    """A specialised sub-agent registered with the supervisor."""

    name: str
    description: str  # shown to the supervisor for routing decisions
    agent: Any  # any object with an async arun(task) or run(task) method


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class SupervisorAgent:
    """
    Orchestrates multiple worker agents.

    Routing logic:
      1. Supervisor LLM decides which worker(s) to call.
      2. Workers execute (optionally in parallel).
      3. Supervisor synthesises all results into a final answer.

    Example:
        supervisor = SupervisorAgent(
            workers=[
                Worker("researcher", "Searches the web for facts", researcher_agent),
                Worker("coder",      "Writes and runs Python code", coder_agent),
            ],
            llm=groq_client,
        )
        result = await supervisor.run("Research FAISS and write a Python example using it.")
    """

    ROUTE_PROMPT = """\
You are a task orchestrator. Choose which worker(s) should handle the task below.

Workers available:
{workers}

Task: {task}

Reply with a JSON list of worker names to invoke (in order), e.g. ["researcher", "coder"].
If a single worker suffices, use ["worker_name"].
Reply with ONLY the JSON list, no explanation.
"""

    SYNTHESIS_PROMPT = """\
You are a coordinator. Combine the outputs from your workers into a single, coherent response.

Original task: {task}

Worker outputs:
{outputs}

Final synthesised answer:"""

    def __init__(
        self,
        workers: List[Worker],
        llm: Any,
        max_parallel: int = 4,
        verbose: bool = True,
    ) -> None:
        self._workers: Dict[str, Worker] = {w.name: w for w in workers}
        self._llm = llm
        self._max_parallel = max_parallel
        self._verbose = verbose

    async def run(self, task: str) -> str:
        t0 = time.perf_counter()

        # Step 1 — routing decision
        worker_list = "\n".join(
            f"  • {w.name}: {w.description}" for w in self._workers.values()
        )
        route_prompt = self.ROUTE_PROMPT.format(workers=worker_list, task=task)
        import json, re

        raw = self._llm.complete(route_prompt)
        try:
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            chosen_names: List[str] = json.loads(match.group()) if match else []
        except Exception:
            logger.warning(
                "Supervisor: failed to parse routing response — using all workers"
            )
            chosen_names = list(self._workers)

        if self._verbose:
            logger.info("Supervisor routing to: %s", chosen_names)

        # Step 2 — run workers (parallel up to max_parallel)
        outputs: Dict[str, str] = {}
        semaphore = asyncio.Semaphore(self._max_parallel)

        async def _run_worker(name: str) -> None:
            worker = self._workers.get(name)
            if worker is None:
                outputs[name] = f"ERROR: worker '{name}' not found."
                return
            async with semaphore:
                try:
                    if hasattr(worker.agent, "arun"):
                        result = await worker.agent.arun(task)
                        outputs[name] = (
                            result.answer if hasattr(result, "answer") else str(result)
                        )
                    else:
                        result = await asyncio.to_thread(worker.agent.run, task)
                        outputs[name] = (
                            result.answer if hasattr(result, "answer") else str(result)
                        )
                except Exception as exc:
                    logger.error("Worker '%s' failed: %s", name, exc)
                    outputs[name] = f"ERROR: {exc}"

        await asyncio.gather(*[_run_worker(n) for n in chosen_names])

        # Step 3 — synthesis
        outputs_text = "\n\n".join(f"[{k}]\n{v}" for k, v in outputs.items())
        synthesis_prompt = self.SYNTHESIS_PROMPT.format(task=task, outputs=outputs_text)
        answer = self._llm.complete(synthesis_prompt)

        if self._verbose:
            logger.info("Supervisor done in %.2fs", time.perf_counter() - t0)
        return answer

    def run_sync(self, task: str) -> str:
        return asyncio.run(self.run(task))
