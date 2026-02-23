"""
Groq client — runnable examples covering both LLM and Whisper audio modes.

Run from the clients/ directory:
    python examples/groq_example.py

For audio examples set the environment variable:
    AUDIO_FILE=/path/to/audio.mp3
"""

import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from groq import GroqClient
from _prompt import Templates


# ---------------------------------------------------------------------------
# 1. LLM chat with memory
# ---------------------------------------------------------------------------
def llm_chat_example():
    print("=" * 60)
    print("1. LLM chat with rolling memory")
    print("=" * 60)
    client = GroqClient.from_env(enable_memory=True)
    turns = [
        "I'm learning Rust coming from Python.",
        "What are the most important concepts I need to understand first?",
        "Give me a concrete example of the borrow checker catching a real bug.",
    ]
    for msg in turns:
        print(f"\nUser: {msg}")
        reply = client.chat(msg)
        print(f"Groq: {reply[:300]}")


# ---------------------------------------------------------------------------
# 2. Streaming
# ---------------------------------------------------------------------------
async def streaming_example():
    print("\n" + "=" * 60)
    print("2. Streaming LLM response")
    print("=" * 60)
    client = GroqClient.from_env(enable_memory=False)
    print("Groq: ", end="", flush=True)
    async for token in client.astream("Explain attention mechanisms in transformers."):
        print(token, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# 3. Whisper transcription
# ---------------------------------------------------------------------------
def transcription_example(audio_path: str | None = None):
    print("\n" + "=" * 60)
    print("3. Whisper transcription")
    print("=" * 60)
    if not audio_path:
        print("Skipped — set AUDIO_FILE env var to enable this demo.")
        return
    client = GroqClient.from_env()
    text = client.transcribe(audio_path, language="en")
    print(f"Transcript: {text[:400]}")


# ---------------------------------------------------------------------------
# 4. Whisper transcription — word-level timestamps
# ---------------------------------------------------------------------------
def transcription_timestamps_example(audio_path: str | None = None):
    print("\n" + "=" * 60)
    print("4. Whisper transcription with word-level timestamps")
    print("=" * 60)
    if not audio_path:
        print("Skipped — set AUDIO_FILE env var.")
        return
    client = GroqClient.from_env()
    result = client.transcribe(
        audio_path,
        response_format="verbose_json",
        timestamp_granularities=["word"],
        model="whisper-large-v3",
    )
    print(f"Result type: {type(result)}")
    print(str(result)[:500])


# ---------------------------------------------------------------------------
# 5. Whisper translation → English
# ---------------------------------------------------------------------------
def translation_example(audio_path: str | None = None):
    print("\n" + "=" * 60)
    print("5. Whisper translation → English")
    print("=" * 60)
    if not audio_path:
        print("Skipped — set AUDIO_FILE env var.")
        return
    client = GroqClient.from_env()
    text = client.translate(audio_path)
    print(f"English translation: {text[:400]}")


# ---------------------------------------------------------------------------
# 6. Async transcription
# ---------------------------------------------------------------------------
async def async_transcription_example(audio_path: str | None = None):
    print("\n" + "=" * 60)
    print("6. Async transcription")
    print("=" * 60)
    if not audio_path:
        print("Skipped — set AUDIO_FILE env var.")
        return
    client = GroqClient.from_env()
    text = await client.atranscribe(audio_path)
    print(f"Async transcript: {text[:400]}")


# ---------------------------------------------------------------------------
# 7. LLM model switching
# ---------------------------------------------------------------------------
def model_switch_example():
    print("\n" + "=" * 60)
    print("7. LLM model switching")
    print("=" * 60)
    client = GroqClient.from_env(enable_memory=False)
    prompt = "What is KV-cache? One paragraph."
    for model_id in ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]:
        try:
            reply = client.chat(prompt, model=model_id)
            print(f"\n[{model_id}]\n{reply[:250]}")
        except Exception as e:
            print(f"[{model_id}] error: {e}")


# ---------------------------------------------------------------------------
# 8. Whisper model switching
# ---------------------------------------------------------------------------
def whisper_model_switch_example(audio_path: str | None = None):
    print("\n" + "=" * 60)
    print("8. Whisper model switching")
    print("=" * 60)
    if not audio_path:
        print("Skipped — set AUDIO_FILE env var.")
        return
    client = GroqClient.from_env()
    for whisper_model in ["whisper-large-v3-turbo", "whisper-large-v3"]:
        client.set_whisper_model(whisper_model)
        text = client.transcribe(audio_path)
        print(f"\n[{whisper_model}]: {text[:150]}")


# ---------------------------------------------------------------------------
# 9. Prompt template (transcript clean-up)
# ---------------------------------------------------------------------------
def transcript_clean_example():
    print("\n" + "=" * 60)
    print("9. Transcript clean-up template")
    print("=" * 60)
    client = GroqClient.from_env(enable_memory=False)
    client.set_template(Templates.transcript_clean)
    raw_transcript = (
        "uh so today we gonna talk about um reinforcement learning, "
        "like basically its uh an agent that learns from like... rewards "
        "and stuff you know what i mean"
    )
    reply = client.chat(message="", transcript=raw_transcript)
    print(f"Cleaned: {reply[:400]}")


# ---------------------------------------------------------------------------
# 10. Key rotation status
# ---------------------------------------------------------------------------
def key_status_example():
    print("\n" + "=" * 60)
    print("10. Key rotation status")
    print("=" * 60)
    client = GroqClient.from_env()
    for entry in client.key_status():
        print(
            f"  Key ...{entry['key'][-6:]}: rpm_used={entry.get('rpm_used', 0)}, "
            f"tpm_used={entry.get('tpm_used', 0)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    audio = os.environ.get("AUDIO_FILE")

    llm_chat_example()
    asyncio.run(streaming_example())
    transcription_example(audio)
    transcription_timestamps_example(audio)
    translation_example(audio)
    asyncio.run(async_transcription_example(audio))
    model_switch_example()
    whisper_model_switch_example(audio)
    transcript_clean_example()
    key_status_example()
