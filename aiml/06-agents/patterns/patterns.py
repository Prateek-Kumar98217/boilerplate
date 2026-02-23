"""
Multi-agent patterns — parallelisation, map-reduce, human-in-the-loop, plan-and-execute.

  ParallelAgents     — fan-out same task to N agents, aggregate results
  MapReduceAgents    — map subtasks across agents, reduce into final answer
  HumanInTheLoop     — pause for human feedback at configurable checkpoints
  PlanAndExecute     — LLM generates a plan, agent executes step-by-step
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern 1: Parallel agents
# ---------------------------------------------------------------------------


@dataclass
class ParallelResult:
    agent_name: str
    answer: str
    error: Optional[str] = None


class ParallelAgents:
    """
    Fan the same query out to N agents concurrently and collect all responses.
    Useful for ensemble voting, diversity sampling, or cross-validation.

    Example:
        pa = ParallelAgents(
            agents={"gpt": gpt_agent, "llama": llama_agent, "gemini": gemini_agent}
        )
        results = asyncio.run(pa.run("What is the capital of France?"))
    """

    def __init__(self, agents: Dict[str, Any], timeout: float = 30.0) -> None:
        self._agents = agents
        self._timeout = timeout

    async def run(self, query: str) -> List[ParallelResult]:
        async def _call(name: str, agent: Any) -> ParallelResult:
            try:
                if hasattr(agent, "arun"):
                    res = await asyncio.wait_for(
                        agent.arun(query), timeout=self._timeout
                    )
                    answer = res.answer if hasattr(res, "answer") else str(res)
                else:
                    res = await asyncio.wait_for(
                        asyncio.to_thread(agent.run, query), timeout=self._timeout
                    )
                    answer = res.answer if hasattr(res, "answer") else str(res)
                return ParallelResult(agent_name=name, answer=answer)
            except asyncio.TimeoutError:
                return ParallelResult(agent_name=name, answer="", error="timeout")
            except Exception as exc:
                return ParallelResult(agent_name=name, answer="", error=str(exc))

        return await asyncio.gather(*[_call(n, a) for n, a in self._agents.items()])

    def vote(self, results: List[ParallelResult]) -> str:
        """Majority vote: return the most common non-empty answer."""
        from collections import Counter

        answers = [r.answer for r in results if r.answer]
        if not answers:
            return ""
        return Counter(answers).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Pattern 2: Map-Reduce
# ---------------------------------------------------------------------------


class MapReduceAgents:
    """
    Split a large task into subtasks (map), run each on a worker agent,
    then aggregate with a reducer LLM.

    Example:
        mr = MapReduceAgents(worker_agent=worker, reducer_llm=groq)
        answer = asyncio.run(mr.run(
            task="Summarise the key points of these 5 documents",
            subtasks=[f"Summarise document {i}" for i in range(5)]
        ))
    """

    def __init__(
        self,
        worker_agent: Any,
        reducer_llm: Any,
        max_parallel: int = 4,
    ) -> None:
        self._worker = worker_agent
        self._reducer = reducer_llm
        self._max_parallel = max_parallel

    async def run(self, task: str, subtasks: List[str]) -> str:
        semaphore = asyncio.Semaphore(self._max_parallel)
        results: Dict[int, str] = {}

        async def _map(i: int, subtask: str) -> None:
            async with semaphore:
                try:
                    if hasattr(self._worker, "arun"):
                        res = await self._worker.arun(subtask)
                        results[i] = res.answer if hasattr(res, "answer") else str(res)
                    else:
                        res = await asyncio.to_thread(self._worker.run, subtask)
                        results[i] = res.answer if hasattr(res, "answer") else str(res)
                except Exception as exc:
                    results[i] = f"ERROR: {exc}"

        await asyncio.gather(*[_map(i, st) for i, st in enumerate(subtasks)])

        # Reduce
        mapped_text = "\n\n".join(
            f"[Subtask {i+1}: {subtasks[i]}]\n{results[i]}"
            for i in range(len(subtasks))
        )
        reduce_prompt = (
            f"Original task: {task}\n\n"
            f"Results from workers:\n{mapped_text}\n\n"
            "Synthesise all results into one comprehensive, coherent final answer:"
        )
        return self._reducer.complete(reduce_prompt)


# ---------------------------------------------------------------------------
# Pattern 3: Human-in-the-Loop
# ---------------------------------------------------------------------------


@dataclass
class CheckpointResult:
    approved: bool
    human_feedback: str = ""
    modified_content: str = ""


class HumanInTheLoopAgent:
    """
    Wraps any agent with a human approval gate at configurable checkpoints.

    In production, replace ``_request_human_input`` with a UI callback,
    Slack message, or webhook integration.

    Example:
        agent = HumanInTheLoopAgent(
            inner_agent=react_agent,
            checkpoint_fn=my_review_callback,
        )
        result = agent.run("Draft a company blog post about our new release.")
    """

    def __init__(
        self,
        inner_agent: Any,
        checkpoint_fn: Optional[Callable[[str, str], CheckpointResult]] = None,
        require_approval_for: Optional[List[str]] = None,
    ) -> None:
        self._agent = inner_agent
        self._checkpoint = checkpoint_fn or self._cli_checkpoint
        self._require_for = set(require_approval_for or [])

    def run(self, task: str) -> Any:
        logger.info("HITL: starting task: %s", task[:100])
        result = self._agent.run(task)

        for step in result.steps:
            for tc in step.tool_calls:
                if tc.name in self._require_for:
                    cp = self._checkpoint(tc.name, str(tc.arguments))
                    if not cp.approved:
                        logger.info("HITL: tool '%s' blocked by human.", tc.name)
                        return result  # early exit

        # Final answer gate
        cp = self._checkpoint("final_answer", result.answer)
        if not cp.approved:
            result = result.__class__(
                answer=cp.modified_content or result.answer,
                steps=result.steps,
                total_iterations=result.total_iterations,
                elapsed_s=result.elapsed_s,
            )
        return result

    @staticmethod
    def _cli_checkpoint(action: str, content: str) -> CheckpointResult:
        """Default: interactive CLI approval (replace in production)."""
        print(f"\n[Human Review Required]\nAction: {action}\nContent: {content[:300]}")
        answer = input("Approve? (y/n/edit): ").strip().lower()
        if answer == "n":
            return CheckpointResult(approved=False)
        if answer == "edit":
            modified = input("Enter modified content: ")
            return CheckpointResult(approved=True, modified_content=modified)
        return CheckpointResult(approved=True)


# ---------------------------------------------------------------------------
# Pattern 4: Plan-and-Execute
# ---------------------------------------------------------------------------


class PlanAndExecuteAgent:
    """
    Two-phase agent:
      1. Planner LLM breaks the goal into numbered steps.
      2. Executor agent runs each step sequentially, passing results forward.

    Example:
        agent = PlanAndExecuteAgent(planner_llm=groq, executor_agent=react_agent)
        answer = agent.run("Research LangGraph, then write a Python Hello World example using it.")
    """

    PLAN_PROMPT = """\
You are a task planner. Break the following goal into clear, numbered executable steps.
Each step should be independently actionable.
Output ONLY the numbered list. No explanation.

Goal: {goal}
"""

    def __init__(
        self,
        planner_llm: Any,
        executor_agent: Any,
        max_plan_steps: int = 8,
        verbose: bool = True,
    ) -> None:
        self._planner = planner_llm
        self._executor = executor_agent
        self._max_steps = max_plan_steps
        self._verbose = verbose

    def run(self, goal: str) -> str:
        import re

        # Phase 1: Plan
        plan_text = self._planner.complete(self.PLAN_PROMPT.format(goal=goal))
        steps = re.findall(r"\d+\.\s*(.+)", plan_text)[: self._max_steps]
        if not steps:
            steps = [goal]

        if self._verbose:
            logger.info(
                "Plan:\n%s", "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
            )

        # Phase 2: Execute
        context = ""
        for i, step in enumerate(steps, 1):
            task_with_context = (
                f"{step}\n\nContext from previous steps:\n{context}"
                if context
                else step
            )
            if self._verbose:
                logger.info("Executing step %d/%d: %s", i, len(steps), step[:80])

            result = self._executor.run(task_with_context)
            answer = result.answer if hasattr(result, "answer") else str(result)
            context += f"\nStep {i} result: {answer}"

        # Final synthesis
        synthesis_prompt = (
            f"Goal: {goal}\n\nExecution results:\n{context}\n\n"
            "Provide a complete, clear final answer:"
        )
        return self._planner.complete(synthesis_prompt)
