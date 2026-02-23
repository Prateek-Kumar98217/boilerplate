"""
LangGraph multi-agent workflow.

Implements a stateful StateGraph where multiple specialised agents are nodes,
connected via conditional routing on the graph state.

Install: pip install langgraph langchain-core
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Shared state passed between graph nodes."""

    messages: Annotated[List[Dict[str, str]], operator.add]
    task: str
    current_agent: str
    agent_outputs: Dict[str, str]
    final_answer: Optional[str]
    error: Optional[str]
    iterations: int


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


def make_agent_node(agent_name: str, agent_callable: Any):
    """
    Wrap any callable agent (f(state) -> state) as a LangGraph node.

    Args:
        agent_name: Identifier stored in state.current_agent.
        agent_callable: Callable receiving AgentState, returning partial update dict.
    """

    def node(state: AgentState) -> Dict:
        logger.info("LangGraph: running agent '%s'", agent_name)
        result = agent_callable(state)
        return {
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                agent_name: str(result),
            },
            "current_agent": agent_name,
            "iterations": state.get("iterations", 0) + 1,
        }

    node.__name__ = agent_name
    return node


def make_llm_node(agent_name: str, llm: Any, system_prompt: str):
    """Convenience: create a simple LLM node that calls llm.chat(messages)."""

    def node(state: AgentState) -> Dict:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(state["messages"])
        answer = llm.chat(messages)
        return {
            "messages": [{"role": "assistant", "content": answer}],
            "agent_outputs": {**state.get("agent_outputs", {}), agent_name: answer},
            "current_agent": agent_name,
            "iterations": state.get("iterations", 0) + 1,
        }

    node.__name__ = agent_name
    return node


# ---------------------------------------------------------------------------
# Conditional routing helpers
# ---------------------------------------------------------------------------


def route_by_keyword(state: AgentState) -> str:
    """
    Example routing function: routes to agent based on task keywords.
    Meant to be customised for your use case.
    """
    task = state.get("task", "").lower()
    if any(kw in task for kw in ("search", "find", "look up", "google")):
        return "search_agent"
    if any(kw in task for kw in ("code", "write", "implement", "script")):
        return "code_agent"
    if any(kw in task for kw in ("calculate", "compute", "math", "number")):
        return "math_agent"
    return "general_agent"


def should_continue(state: AgentState) -> str:
    """Route to END if final_answer is set, else continue."""
    if state.get("final_answer") or state.get("iterations", 0) >= 10:
        return "end"
    return "continue"


# ---------------------------------------------------------------------------
# Pre-built pipeline factory
# ---------------------------------------------------------------------------


def build_sequential_pipeline(
    nodes: List[tuple[str, Any]],
    max_iterations: int = 10,
) -> Any:
    """
    Build a simple linear StateGraph: node0 → node1 → ... → END.

    Args:
        nodes: List of (name, callable) pairs where callable accepts AgentState.
        max_iterations: Safety cap on total node invocations.

    Returns:
        A compiled LangGraph CompiledGraph ready to call with ``invoke()``.

    Example:
        graph = build_sequential_pipeline([
            ("researcher", researcher_node),
            ("writer",     writer_node),
        ])
        result = graph.invoke({"task": "Write a blog post about FAISS", "messages": [], ...})
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError("langgraph required: pip install langgraph")

    graph = StateGraph(AgentState)

    for i, (name, fn) in enumerate(nodes):
        graph.add_node(name, fn)

    # Linear edges
    for i in range(len(nodes) - 1):
        graph.add_edge(nodes[i][0], nodes[i + 1][0])

    graph.set_entry_point(nodes[0][0])
    graph.add_edge(nodes[-1][0], END)
    return graph.compile()


def build_conditional_pipeline(
    nodes: Dict[str, Any],
    entry: str,
    router: Any,  # (state) -> str
    terminal_nodes: Optional[List[str]] = None,
) -> Any:
    """
    Build a StateGraph with conditional routing.

    Args:
        nodes: {name: callable} mapping.
        entry: Name of the entry node.
        router: A function (state) → next_node_name (or "end").
        terminal_nodes: Nodes that route to END unconditionally.

    Returns:
        Compiled LangGraph.
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError("langgraph required: pip install langgraph")

    graph = StateGraph(AgentState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point(entry)
    routing_map = {name: name for name in nodes}
    routing_map["end"] = END

    for name in nodes:
        if terminal_nodes and name in terminal_nodes:
            graph.add_edge(name, END)
        else:
            graph.add_conditional_edges(name, router, routing_map)

    return graph.compile()
