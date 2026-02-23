"""
Cerebras client — runnable examples.

Run from the clients/ directory:
    python examples/cerebras_example.py
"""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from cerebras import CerebrasClient, CEREBRAS_MODELS
from _prompt import Templates


# ---------------------------------------------------------------------------
# 1. Text chat with persistent memory
# ---------------------------------------------------------------------------
def text_chat_example():
    print("=" * 60)
    print("1. Text chat with rolling memory")
    print("=" * 60)
    client = CerebrasClient.from_env(enable_memory=True)
    turns = [
        "I'm building a distributed ML training system.",
        "What are the key bottlenecks I should worry about?",
        "How would you specifically address the memory bottleneck?",
    ]
    for msg in turns:
        print(f"\nUser: {msg}")
        reply = client.chat(msg)
        print(f"Cerebras: {reply[:300]}")


# ---------------------------------------------------------------------------
# 2. Streaming response
# ---------------------------------------------------------------------------
async def streaming_example():
    print("\n" + "=" * 60)
    print("2. Streaming response")
    print("=" * 60)
    client = CerebrasClient.from_env(enable_memory=False)
    print("Cerebras: ", end="", flush=True)
    async for token in client.astream("Explain mixture-of-experts (MoE) briefly."):
        print(token, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 3. Completion (no memory, single turn)
# ---------------------------------------------------------------------------
def completion_example():
    print("\n" + "=" * 60)
    print("3. Completion (single turn, no memory)")
    print("=" * 60)
    client = CerebrasClient.from_env(enable_memory=False)
    result = client.complete(
        'def fibonacci(n):\n    """Return the nth Fibonacci number."""'
    )
    print(f"Cerebras completion:\n{result[:400]}")


# ---------------------------------------------------------------------------
# 4. Model switching per call
# ---------------------------------------------------------------------------
def model_switch_example():
    print("\n" + "=" * 60)
    print("4. Model switching per call")
    print("=" * 60)
    client = CerebrasClient.from_env(enable_memory=False)
    prompt = "What is sparsity in neural networks? Two sentences."
    for mid, mfull in [
        ("llama3.1-8b", "llama3.1-8b"),
        ("llama3.3-70b", "llama3.3-70b"),
    ]:
        try:
            reply = client.chat(prompt, model=mfull)
            print(f"[{mid}]: {reply[:200]}")
        except Exception as e:
            print(f"[{mid}] unavailable: {e}")


# ---------------------------------------------------------------------------
# 5. Chain-of-thought template
# ---------------------------------------------------------------------------
def template_example():
    print("\n" + "=" * 60)
    print("5. Chain-of-thought template")
    print("=" * 60)
    client = CerebrasClient.from_env(enable_memory=False)
    client.set_template(Templates.chain_of_thought)
    reply = client.chat(
        message="",
        problem="A train travels 60 km/h for 2 hours, then 80 km/h for 1.5 hours. What is the total distance?",
    )
    print(f"Cerebras: {reply[:500]}")


# ---------------------------------------------------------------------------
# 6. Code generation template
# ---------------------------------------------------------------------------
def code_gen_example():
    print("\n" + "=" * 60)
    print("6. Code generation template")
    print("=" * 60)
    client = CerebrasClient.from_env(enable_memory=False)
    client.set_template(Templates.code_gen)
    reply = client.chat(
        message="",
        language="Python",
        task="Implement a thread-safe LRU cache with a configurable max size.",
        context="Use only the standard library.",
    )
    print(f"Cerebras:\n{reply[:600]}")


# ---------------------------------------------------------------------------
# 7. Temperature & token control via default_params
# ---------------------------------------------------------------------------
def generation_params_example():
    print("\n" + "=" * 60)
    print("7. Custom generation parameters")
    print("=" * 60)
    client = CerebrasClient.from_env(
        enable_memory=False,
        default_params={"temperature": 0.2, "max_tokens": 256},
    )
    reply = client.chat("List 5 hyperparameters important for training large LLMs.")
    print(f"Cerebras (temp=0.2, max_tokens=256):\n{reply[:400]}")


# ---------------------------------------------------------------------------
# 8. Key status
# ---------------------------------------------------------------------------
def key_status_example():
    print("\n" + "=" * 60)
    print("8. Key rotation status")
    print("=" * 60)
    client = CerebrasClient.from_env()
    for entry in client.key_status():
        print(f"  Key ...{entry['key'][-6:]}: rpm_used={entry.get('rpm_used', 0)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    text_chat_example()
    asyncio.run(streaming_example())
    completion_example()
    model_switch_example()
    template_example()
    code_gen_example()
    generation_params_example()
    key_status_example()
