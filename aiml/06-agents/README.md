# 06 — Agents

Full-featured agent boilerplate covering single agents, orchestrated multi-agent systems, and advanced workflow patterns with LangGraph.

---

## Directory layout

```
06-agents/
├── .env.example
├── config.py
├── agents/
│   ├── base_agent.py          # Abstract BaseAgent — tools, memory, hooks, streaming
│   ├── react_agent.py         # ReAct (Reason + Act) text parsing loop
│   └── tool_calling_agent.py  # OpenAI native function/tool calling
├── tools/
│   └── builtin_tools.py       # WebSearch, Calculator, CodeExecutor, FileR/W, Wikipedia, RAG
├── memory/
│   └── memory.py              # BufferMemory, SummaryMemory, VectorMemory + factory
├── orchestration/
│   ├── langgraph_workflow.py  # LangGraph StateGraph builder + routing helpers
│   ├── supervisor.py          # SupervisorAgent — routes + parallelises workers
│   └── swarm.py               # AgentSwarm — handoff-based routing
├── patterns/
│   └── patterns.py            # Parallel, MapReduce, HumanInTheLoop, PlanAndExecute
└── examples/
    ├── single_agent.py
    ├── multi_agent_supervisor.py
    └── langgraph_pipeline.py
```

---

## Quick start

```bash
pip install langgraph langchain-core groq sentence-transformers faiss-cpu \
            tavily-python wikipedia rank-bm25

export GROQ_API_KEY="..."
export TAVILY_API_KEY="..."

python examples/single_agent.py
python examples/multi_agent_supervisor.py
python examples/langgraph_pipeline.py
```

---

## Agents

### BaseAgent (`agents/base_agent.py`)

Abstract foundation every agent builds on. Pluggable design:

| Feature       | How                                                |
| ------------- | -------------------------------------------------- |
| Tool registry | `add_tool(Tool)` — any `Tool` subclass             |
| Memory        | Pass `BufferMemory / SummaryMemory / VectorMemory` |
| Hooks         | `AgentHooks(on_tool_call=..., on_end=...)`         |
| Streaming     | `async for token in agent.astream(query)`          |
| Retry         | `retry_on_tool_error=N`                            |
| Async         | `await agent.arun(query)`                          |

### ReAct Agent (`agents/react_agent.py`)

Implements the **Thought → Action → Observation** loop via free-form LLM text parsing. Compatible with any LLM (no tool-calling API required).

```
Thought: I need to find the population of Tokyo.
Action: wikipedia
Action Input: {"query": "Tokyo population"}
Observation: Tokyo has a population of approximately 13.96 million.
Thought: I now know the final answer.
Final Answer: The population of Tokyo is approximately 13.96 million.
```

### Tool Calling Agent (`agents/tool_calling_agent.py`)

Uses the LLM's native function/tool calling API (OpenAI-compatible). Supports **parallel tool calls** in one LLM turn.

---

## Memory

| Type    | Class           | Best for                                     |
| ------- | --------------- | -------------------------------------------- |
| Buffer  | `BufferMemory`  | Short conversations, fast                    |
| Summary | `SummaryMemory` | Long conversations — compresses via LLM      |
| Vector  | `VectorMemory`  | Episodic retrieval of relevant past messages |

```python
from memory.memory import create_memory
mem = create_memory("summary", llm=groq_client, buffer_turns=6)
mem.add("user", "What is FAISS?")
context = mem.get_context(query="vector similarity")
```

---

## Orchestration

### Supervisor (`orchestration/supervisor.py`)

```
User Query
    │
    ▼
SupervisorAgent (LLM routing)
    ├──→ Worker A (researcher)
    ├──→ Worker B (coder)       ← parallel execution
    └──→ Worker C (analyst)
    │
    ▼
Synthesis LLM → Final Answer
```

### Swarm (`orchestration/swarm.py`)

Agents explicitly hand off to each other until one returns a final answer:

```
triage_agent → researcher_agent → coder_agent → FINAL ANSWER
```

### LangGraph (`orchestration/langgraph_workflow.py`)

```python
pipeline = build_sequential_pipeline([
    ("researcher", researcher_node),
    ("writer",     writer_node),
])
result = pipeline.invoke(initial_state)
```

---

## Patterns (`patterns/patterns.py`)

| Pattern           | Class                 | Use case                      |
| ----------------- | --------------------- | ----------------------------- |
| Parallel          | `ParallelAgents`      | Ensemble / diversity sampling |
| Map-Reduce        | `MapReduceAgents`     | Large data split + summarise  |
| Human-in-the-Loop | `HumanInTheLoopAgent` | Approval gates / review       |
| Plan-and-Execute  | `PlanAndExecuteAgent` | Complex multi-step goals      |

---

## Built-in tools

| Tool                 | Class               | Install                             |
| -------------------- | ------------------- | ----------------------------------- |
| Web Search (Tavily)  | `TavilySearchTool`  | `pip install tavily-python`         |
| Web Search (SerpAPI) | `SerpAPISearchTool` | `pip install google-search-results` |
| Math Calculator      | `CalculatorTool`    | _(stdlib only)_                     |
| Python REPL          | `CodeExecutorTool`  | _(stdlib only)_                     |
| File Reader          | `FileReaderTool`    | _(stdlib only)_                     |
| File Writer          | `FileWriterTool`    | _(stdlib only)_                     |
| Wikipedia            | `WikipediaTool`     | `pip install wikipedia`             |
| RAG Knowledge Base   | `RAGTool`           | depends on 05-rag setup             |

### Custom tool

```python
from agents.base_agent import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "Does something useful."
    parameters_schema = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    }

    def run(self, input: str) -> str:
        return f"Processed: {input}"
```

---

## Key configuration (`.env`)

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.1-70b-versatile

AGENT_MAX_ITERATIONS=10
MEMORY_TYPE=buffer          # buffer|summary|vector

TAVILY_API_KEY=...
LANGSMITH_API_KEY=...       # optional LangGraph tracing
```
