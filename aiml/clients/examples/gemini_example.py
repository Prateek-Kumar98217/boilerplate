"""
Gemini client — runnable examples.

Run from the clients/ directory:
    python examples/gemini_example.py
"""

import asyncio
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from gemini import GeminiClient
from _prompt import Templates


# ---------------------------------------------------------------------------
# 1. Basic text chat with memory
# ---------------------------------------------------------------------------
def text_chat_example():
    print("=" * 60)
    print("1. Text chat with rolling memory")
    print("=" * 60)
    client = GeminiClient.from_env(enable_memory=True)
    turns = [
        "My name is Alex and I love astronomy.",
        "What is Olbers' paradox?",
        "Given what you know about me, suggest a good book.",
    ]
    for msg in turns:
        print(f"\nUser: {msg}")
        reply = client.chat(msg)
        print(f"Gemini: {reply[:200]}")
    print(f"\nMemory short-term count: {len(client.memory._short)}")


# ---------------------------------------------------------------------------
# 2. Streaming
# ---------------------------------------------------------------------------
async def streaming_example():
    print("\n" + "=" * 60)
    print("2. Streaming response")
    print("=" * 60)
    client = GeminiClient.from_env(enable_memory=False)
    print("Gemini: ", end="", flush=True)
    async for token in client.astream("Write a haiku about neural networks."):
        print(token, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 3. Vision — image from file path
# ---------------------------------------------------------------------------
def vision_example(image_path: str | None = None):
    print("\n" + "=" * 60)
    print("3. Vision")
    print("=" * 60)
    if not image_path:
        print("Skipped — provide IMAGE_PATH env var to enable this demo.")
        return
    client = GeminiClient.from_env()
    reply = client.vision("Describe what you see in detail.", image=image_path)
    print(f"Gemini: {reply[:400]}")


# ---------------------------------------------------------------------------
# 4. Multimodal (two images side by side)
# ---------------------------------------------------------------------------
def multimodal_example(img1: str | None = None, img2: str | None = None):
    print("\n" + "=" * 60)
    print("4. Multimodal — compare two images")
    print("=" * 60)
    if not (img1 and img2):
        print("Skipped — set IMAGE1 and IMAGE2 env vars to enable.")
        return
    client = GeminiClient.from_env()
    reply = client.multimodal(["Compare these two images:", img1, img2])
    print(f"Gemini: {reply[:400]}")


# ---------------------------------------------------------------------------
# 5. Embedding
# ---------------------------------------------------------------------------
def embedding_example():
    print("\n" + "=" * 60)
    print("5. Text embedding")
    print("=" * 60)
    client = GeminiClient.from_env()
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    vecs = client.embed_batch(texts)
    for text, vec in zip(texts, vecs):
        print(f"  '{text[:40]}...' → vector dim={len(vec)}, first3={vec[:3]}")


# ---------------------------------------------------------------------------
# 6. Custom prompt template
# ---------------------------------------------------------------------------
def template_example():
    print("\n" + "=" * 60)
    print("6. Custom prompt template (RAG QA)")
    print("=" * 60)
    client = GeminiClient.from_env(enable_memory=False)
    client.set_template(Templates.rag_qa)
    reply = client.chat(
        message="",  # unused — template provides user message
        question="What is gradient descent?",
        context=(
            "Gradient descent is an optimisation algorithm used to minimise "
            "a function by iteratively moving in the direction of steepest descent."
        ),
    )
    print(f"Gemini: {reply[:400]}")


# ---------------------------------------------------------------------------
# 7. Per-call system & model override
# ---------------------------------------------------------------------------
def override_example():
    print("\n" + "=" * 60)
    print("7. Per-call system & model override")
    print("=" * 60)
    client = GeminiClient.from_env(system_prompt="You are a concise assistant.")
    reply = client.chat(
        "List five programming languages.",
        system_override="You are an expert in programming languages. Be brief.",
        model="gemini-2.0-flash",
    )
    print(f"Gemini: {reply[:400]}")


# ---------------------------------------------------------------------------
# 8. Key rotation status
# ---------------------------------------------------------------------------
def key_status_example():
    print("\n" + "=" * 60)
    print("8. Key rotation status")
    print("=" * 60)
    client = GeminiClient.from_env()
    for entry in client.key_status():
        print(
            f"  Key ...{entry['key'][-6:]}: rpm_used={entry.get('rpm_used', 0)}, "
            f"blocked={entry.get('blocked_until', 0) > 0}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    text_chat_example()
    asyncio.run(streaming_example())
    vision_example(os.environ.get("IMAGE_PATH"))
    multimodal_example(os.environ.get("IMAGE1"), os.environ.get("IMAGE2"))
    embedding_example()
    template_example()
    override_example()
    key_status_example()
