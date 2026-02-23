# `clients/` — Inference Clients

Production-grade inference clients for **Gemini**, **Cerebras**, and **Groq**, all sharing a unified architecture:

- **Key rotation** — sliding-window rate limiting per key, proactive + reactive 429 handling
- **Memory management** — short-term verbatim buffer + long-term LLM-compressed summary
- **Prompt templates** — composable templates with variable substitution, 16 built-ins
- **Multiple input types** — each client exposes the full API surface of its provider
- **Async-first** — every method has both a sync and an `a`-prefixed async variant

---

## Directory layout

```
clients/
├── .env.example              ← environment variable reference
├── config.py                 ← Pydantic Settings for all providers
│
├── _key_rotator.py           ← rate-limit-aware API key pool
├── _memory.py                ← short + long-term memory with summarisation
├── _prompt.py                ← prompt template engine (16 built-ins)
├── _base.py                  ← abstract base client (retry, memory, prompt wiring)
│
├── gemini.py                 ← Google Gemini  (text, vision, audio, embedding)
├── cerebras.py               ← Cerebras       (fast LLM inference)
├── groq.py                   ← Groq           (LLM + Whisper audio)
│
└── examples/
    ├── gemini_example.py
    ├── cerebras_example.py
    └── groq_example.py
```

---

## Installation

```bash
# Core clients
pip install google-generativeai cerebras-cloud-sdk groq

# Optional
pip install python-dotenv pydantic-settings pillow
```

---

## Quick start

Copy `.env.example` to `.env` and fill in your keys:

```bash
GEMINI_API_KEYS=key1,key2,key3
GROQ_API_KEYS=key1,key2
CEREBRAS_API_KEYS=key1
```

Then:

```python
from gemini import GeminiClient
from cerebras import CerebrasClient
from groq import GroqClient

# All three are constructed identically
gemini  = GeminiClient.from_env()
cerebras = CerebrasClient.from_env()
groq    = GroqClient.from_env()

# Basic chat (sync)
print(gemini.chat("What is attention in transformers?"))
print(cerebras.chat("Explain sparse autoencoders."))
print(groq.chat("What is KV-cache?"))
```

---

## Architecture

```
BaseClient  (abstract)
├── _rotator:    KeyRotator   ← manages API key pool
├── _memory:     MemoryManager ← short + long-term history
├── _template:   PromptTemplate ← message rendering
│
├── chat() / achat()          ← text round-trip
├── complete() / acomplete()  ← single-turn completion
├── astream()                 ← async token generator
│
├── [abstract] _raw_chat()    ← provider SDK call
└── [optional] _raw_stream()  ← provider streaming call
```

### Key rotation (`_key_rotator.py`)

```python
from _key_rotator import KeyRotator

rotator = KeyRotator(
    keys=["key1", "key2", "key3"],
    rpm_limit=30,        # requests per minute per key
    rpd_limit=14400,     # requests per day per key (optional)
    tpm_limit=131072,    # tokens per minute per key (optional)
)
```

The rotator maintains **sliding-window counters** per key (60s for RPM/TPM, 24 h for RPD).  
When a 429 is received, the offending key is blocked with **exponential backoff** (60 → 120 → 240 s, max 1 h).

```python
key = await rotator.acquire(tokens_hint=500)  # blocks until a key is free
# ... use key ...
rotator.release(key, tokens_used=423)

# Check status
for entry in rotator.status():
    print(entry)  # {"key": "...", "rpm_used": 12, "blocked_until": 0.0}
```

### Memory management (`_memory.py`)

Short-term memory keeps the last _N_ turns verbatim. When the turn count or
character budget is exceeded, the oldest half is **summarised** by the client
itself and folded into a long-term summary sent as a special system message.

```python
# Memory is wired automatically through chat() / achat()

# Manual access
client.memory.add("user", "My name is Alex.")
context = client.memory.get_context()   # list of messages

# Persistence
snapshot = client.memory.snapshot()     # serialisable dict
client.memory.restore(snapshot)

# Reset
client.reset_memory()                   # wipe everything
client.memory.clear_short_term()        # keep long-term summary
```

### Prompt templates (`_prompt.py`)

````python
from _prompt import Templates, PromptTemplate

# Use a built-in template
client.set_template(Templates.rag_qa)
reply = client.chat("", question="What is RLHF?", context="...")

# Compose templates
my_tpl = Templates.chain_of_thought.extend_system(
    "\n\nAlways show your work step by step."
).with_defaults(language="Python")

# Custom template
custom = PromptTemplate(
    name="bug_finder",
    system="You are a senior code reviewer. Find bugs in the provided code.",
    user="Language: {language}\n\nCode:\n```\n{code}\n```",
)
client.set_template(custom)
reply = client.chat("", language="Python", code="def add(a,b): return a-b")
````

Built-in templates: `chat`, `completion`, `qa`, `rag_qa`, `rag_citation`,
`chain_of_thought`, `classification`, `summarise`, `bullet_summary`,
`code_gen`, `code_review`, `json_extract`, `translate`, `persona`,
`transcript_clean`.

---

## Client reference

### GeminiClient

| Method                      | Input        | Description                        |
| --------------------------- | ------------ | ---------------------------------- |
| `chat(message)`             | text         | LLM conversation with memory       |
| `complete(prompt)`          | text         | Single-turn completion             |
| `astream(message)`          | text         | Async token generator              |
| `vision(prompt, image)`     | text + image | Analyse an image                   |
| `audio(prompt, audio_path)` | text + audio | Analyse / transcribe audio         |
| `multimodal(parts)`         | mixed list   | Arbitrary text + image + audio mix |
| `embed(text)`               | text         | Single embedding vector            |
| `embed_batch(texts)`        | text list    | Batch embeddings                   |

```python
client = GeminiClient.from_env(
    model="gemini-2.0-flash",
    vision_model="gemini-2.0-flash",
    embedding_model="text-embedding-004",
    rpm_limit=15,
    rpd_limit=1500,
)

# Vision
reply = client.vision("What is the street name?", image="photo.jpg")

# Audio
transcript = client.audio("Transcribe this.", "lecture.mp3", mime_type="audio/mp3")

# Embeddings
vec: list[float] = client.embed("Hello, world!")
vecs: list[list[float]] = client.embed_batch(["Hello", "World"])
```

### CerebrasClient

| Method             | Input | Description                  |
| ------------------ | ----- | ---------------------------- |
| `chat(message)`    | text  | LLM conversation with memory |
| `complete(prompt)` | text  | Single-turn completion       |
| `astream(message)` | text  | Async token generator        |

```python
client = CerebrasClient.from_env(
    model="llama3.1-70b",         # default
    # model="llama3.3-70b"
    # model="qwen-3-32b"
    # model="deepseek-r1-distill-llama-70b"
    default_params={"temperature": 0.3, "max_tokens": 1024},
)
reply = client.chat("Explain parameter-efficient fine-tuning.")
```

### GroqClient

| Method               | Input      | Description                   |
| -------------------- | ---------- | ----------------------------- |
| `chat(message)`      | text       | LLM conversation with memory  |
| `complete(prompt)`   | text       | Single-turn LLM completion    |
| `astream(message)`   | text       | Async token generator         |
| `transcribe(audio)`  | audio file | Whisper speech-to-text        |
| `translate(audio)`   | audio file | Whisper translation → English |
| `atranscribe(audio)` | audio file | Async transcription           |
| `atranslate(audio)`  | audio file | Async translation             |

```python
client = GroqClient.from_env(
    model="llama-3.3-70b-versatile",
    whisper_model="whisper-large-v3-turbo",
)

# LLM
reply = client.chat("What is speculative decoding?")

# Whisper transcription
text = client.transcribe("interview.mp3", language="en")

# Whisper with timestamps
result = client.transcribe(
    "lecture.mp3",
    model="whisper-large-v3",
    response_format="verbose_json",
    timestamp_granularities=["word"],
)

# Translation to English
english = client.translate("german_audio.m4a")

# Switch whisper model at runtime
client.set_whisper_model("whisper-large-v3")
```

---

## Configuration (`.env`)

```env
# --- Gemini ---
GEMINI_API_KEYS=key1,key2,key3   # comma-separated for rotation
GEMINI_DEFAULT_MODEL=gemini-2.0-flash
GEMINI_RPM_LIMIT=15
GEMINI_RPD_LIMIT=1500

# --- Groq ---
GROQ_API_KEYS=key1,key2
GROQ_DEFAULT_MODEL=llama-3.3-70b-versatile
GROQ_WHISPER_MODEL=whisper-large-v3-turbo
GROQ_RPM_LIMIT=30
GROQ_TPM_LIMIT=131072

# --- Cerebras ---
CEREBRAS_API_KEYS=key1
CEREBRAS_DEFAULT_MODEL=llama3.1-70b
CEREBRAS_RPM_LIMIT=30

# --- Memory ---
MEMORY_SHORT_TERM_TURNS=20
MEMORY_MAX_SHORT_CHARS=8000
MEMORY_SUMMARY_MODEL=         # leave blank to use the client's own default model

# --- Prompts ---
DEFAULT_SYSTEM_PROMPT=You are a helpful AI assistant.
```

---

## Running examples

```bash
cd clients/

# Gemini
GEMINI_API_KEYS=your_key python examples/gemini_example.py

# Cerebras
CEREBRAS_API_KEYS=your_key python examples/cerebras_example.py

# Groq — LLM only
GROQ_API_KEYS=your_key python examples/groq_example.py

# Groq — with audio demo
GROQ_API_KEYS=your_key AUDIO_FILE=lecture.mp3 python examples/groq_example.py
```
