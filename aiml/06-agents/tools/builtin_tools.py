"""
Built-in tools for agents.

Includes:
  • WebSearchTool     — Tavily or SerpAPI web search
  • CalculatorTool    — safe math expression evaluator
  • CodeExecutorTool  — sandboxed Python code execution
  • FileReaderTool    — read local text files
  • FileWriterTool    — write / append to local files
  • RAGTool           — query a RAG pipeline as a tool
  • WikipediaTool     — Wikipedia summary lookup
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
import os
import signal
import textwrap
import time
from typing import Any, Dict, Optional

from agents.base_agent import Tool


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------


class TavilySearchTool(Tool):
    """Web search via Tavily API (optimised for AI agents)."""

    name = "web_search"
    description = (
        "Search the web for current information. "
        "Input: a natural language search query string."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {"type": "integer", "default": 3},
        },
        "required": ["query"],
    }

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key or os.getenv("TAVILY_API_KEY", "")

    def run(self, query: str, max_results: int = 3) -> str:
        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError("tavily-python required: pip install tavily-python")
        client = TavilyClient(api_key=self._api_key)
        resp = client.search(query=query, max_results=max_results)
        results = resp.get("results", [])
        lines = [f"[{r['title']}] {r['content'][:300]}" for r in results]
        return "\n\n".join(lines) if lines else "No results found."


class SerpAPISearchTool(Tool):
    """Web search via SerpAPI (Google search results)."""

    name = "web_search"
    description = "Search the web using Google. Input: search query string."
    parameters_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self, api_key: str = "", num_results: int = 5) -> None:
        self._api_key = api_key or os.getenv("SERPAPI_API_KEY", "")
        self._num = num_results

    def run(self, query: str) -> str:
        try:
            from serpapi import GoogleSearch
        except ImportError:
            raise ImportError(
                "google-search-results required: pip install google-search-results"
            )
        params = {"q": query, "api_key": self._api_key, "num": self._num}
        results = GoogleSearch(params).get_dict().get("organic_results", [])
        lines = [
            f"[{r.get('title','')}] {r.get('snippet','')} — {r.get('link','')}"
            for r in results[: self._num]
        ]
        return "\n".join(lines) if lines else "No results found."


# ---------------------------------------------------------------------------
# Calculator — safe math evaluator
# ---------------------------------------------------------------------------

_SAFE_MATH = {
    name: getattr(math, name) for name in dir(math) if not name.startswith("_")
}
_SAFE_MATH.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})


class CalculatorTool(Tool):
    """Evaluate arithmetic/math expressions safely (no exec, no eval of arbitrary code)."""

    name = "calculator"
    description = (
        "Evaluate a mathematical expression. "
        "Supports +,-,*,/,**,%, sqrt, sin, cos, log, etc. "
        "Input: a Python math expression string, e.g. '2**10 + sqrt(144)'."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"}
        },
        "required": ["expression"],
    }

    def run(self, expression: str) -> str:
        try:
            tree = ast.parse(expression, mode="eval")
            # Whitelist allowed AST node types
            allowed = {
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Call,
                ast.Constant,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Pow,
                ast.Mod,
                ast.FloorDiv,
                ast.USub,
                ast.UAdd,
                ast.Name,
                ast.Load,
            }
            for node in ast.walk(tree):
                if type(node) not in allowed:
                    return f"ERROR: unsupported expression node {type(node).__name__}"
            result = eval(
                compile(tree, "<expr>", "eval"), {"__builtins__": {}}, _SAFE_MATH
            )  # noqa: S307
            return str(result)
        except Exception as exc:
            return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Python code executor — sandboxed (timeout + no dangerous builtins)
# ---------------------------------------------------------------------------


class CodeExecutorTool(Tool):
    """Execute Python code in a restricted sandbox and return stdout."""

    name = "python_repl"
    description = (
        "Execute Python code and return the output. "
        "Use for data processing, calculations, or generating text programmatically."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to run"},
            "timeout": {
                "type": "integer",
                "default": 10,
                "description": "Timeout in seconds",
            },
        },
        "required": ["code"],
    }

    # Builtins to block
    BLOCKED = {"open", "exec", "eval", "__import__", "compile", "input", "breakpoint"}

    def __init__(self, timeout: int = 10) -> None:
        self._timeout = timeout

    def run(self, code: str, timeout: Optional[int] = None) -> str:
        safe_globals: Dict[str, Any] = (
            {
                k: v
                for k, v in __builtins__.items()  # type: ignore[union-attr]
                if k not in self.BLOCKED
            }
            if isinstance(__builtins__, dict)
            else {
                k: getattr(__builtins__, k)
                for k in dir(__builtins__)  # type: ignore[call-overload]
                if k not in self.BLOCKED
            }
        )
        safe_globals["__builtins__"] = safe_globals

        buf = io.StringIO()
        t = timeout or self._timeout

        def _handler(signum, frame):
            raise TimeoutError(f"Code execution exceeded {t}s timeout.")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(t)
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, safe_globals)  # noqa: S102
        except TimeoutError as exc:
            return f"TIMEOUT: {exc}"
        except Exception as exc:
            return f"ERROR: {exc}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

        return buf.getvalue() or "(no output)"


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------


class FileReaderTool(Tool):
    name = "read_file"
    description = (
        "Read the contents of a local text file. Input: absolute or relative file path."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"},
            "max_chars": {"type": "integer", "default": 4000},
        },
        "required": ["path"],
    }

    def run(self, path: str, max_chars: int = 4000) -> str:
        try:
            text = open(path).read()  # noqa: WPS515
            return text[:max_chars] + ("…[truncated]" if len(text) > max_chars else "")
        except FileNotFoundError:
            return f"ERROR: file not found — {path}"
        except Exception as exc:
            return f"ERROR: {exc}"


class FileWriterTool(Tool):
    name = "write_file"
    description = "Write or append text to a local file."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "mode": {"type": "string", "enum": ["write", "append"], "default": "write"},
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str, mode: str = "write") -> str:
        flag = "w" if mode == "write" else "a"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, flag) as f:
                f.write(content)
            return f"Written {len(content)} chars to {path}."
        except Exception as exc:
            return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------


class WikipediaTool(Tool):
    name = "wikipedia"
    description = "Look up a Wikipedia summary for a topic."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Topic to look up"},
            "sentences": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def run(self, query: str, sentences: int = 5) -> str:
        try:
            import wikipedia  # type: ignore

            return wikipedia.summary(query, sentences=sentences, auto_suggest=True)
        except ImportError:
            raise ImportError("wikipedia required: pip install wikipedia")
        except Exception as exc:
            return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# RAG tool — query a RAG pipeline as an agent capability
# ---------------------------------------------------------------------------


class RAGTool(Tool):
    """Wrap a LocalRAGPipeline or OnlineRAGPipeline as an agent tool."""

    name = "knowledge_base_search"
    description = (
        "Search the internal knowledge base for relevant information. "
        "Use this before web search for company/domain-specific questions."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question to answer from docs",
            },
            "top_k": {"type": "integer", "default": 3},
        },
        "required": ["query"],
    }

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    def run(self, query: str, top_k: int = 3) -> str:
        response = self._pipeline.query(query, top_k=top_k)
        return response.answer
