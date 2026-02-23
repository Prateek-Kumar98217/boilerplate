# AI/ML Boilerplate

Production-grade, modular boilerplate for AI/ML projects using Python, FastAPI, PyTorch, HuggingFace, LangChain, LangGraph, and more.

## Tech Stack

| Category            | Libraries                                                 |
| ------------------- | --------------------------------------------------------- |
| Core ML             | PyTorch, scikit-learn, NumPy, SciPy, Pandas               |
| LLM/NLP             | HuggingFace Transformers, PEFT, TRL, LangChain, LangGraph |
| Vector DBs          | ChromaDB, FAISS, Pinecone                                 |
| Serving             | FastAPI, Uvicorn, Pydantic                                |
| LLM SDKs            | Ollama, Groq, Cerebras, Google Gemini                     |
| Config              | Pydantic Settings, python-dotenv                          |
| Experiment Tracking | MLflow, Weights & Biases                                  |

## Sections

| #   | Module                                       | Description                                                         |
| --- | -------------------------------------------- | ------------------------------------------------------------------- |
| 01  | [deployment](01-deployment/)                 | FastAPI model serving, MLOps/AIOps pipelines                        |
| 02  | [data-preprocessing](02-data-preprocessing/) | Text, audio, video, image pipelines + ML data analysis              |
| 03  | [training](03-training/)                     | Production PyTorch training with full VRAM optimization             |
| 04  | [finetuning](04-finetuning/)                 | LoRA, QLoRA, full fine-tune, adapters, prompt tuning                |
| 05  | [rag](05-rag/)                               | End-to-end RAG — ingestion, retrieval, generation (local + cloud)   |
| 06  | [agents](06-agents/)                         | Customizable agents, multi-agent orchestration, LangGraph workflows |
| 07  | [clients](clients/)                          | Gemini, Cerebras & Groq clients — key rotation, memory, templates   |

## Conventions

- Every section has its own `config.py` (Pydantic Settings) and `.env.example`
- Secrets/keys are **never** hardcoded — always loaded from environment
- Each section is self-contained and independently runnable
- `examples/` directories contain runnable end-to-end scripts

## Quick Setup

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install core deps
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv \
    torch torchvision torchaudio transformers peft trl \
    langchain langchain-community langgraph \
    chromadb faiss-cpu pinecone-client \
    pandas numpy scipy scikit-learn \
    groq cerebras-cloud-sdk google-generativeai ollama \
    mlflow wandb accelerate bitsandbytes

# Copy and fill env
cp <section>/.env.example <section>/.env
```
