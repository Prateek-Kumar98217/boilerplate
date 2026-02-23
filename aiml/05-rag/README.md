# 05 — RAG (Retrieval-Augmented Generation)

Production-grade RAG stack: **ingestion → chunking → embedding → vector store → retrieval → reranking → generation**.  
Supports local-only (ChromaDB / FAISS + Ollama) and cloud (Pinecone + Groq / Gemini / Cerebras) modes.

---

## Directory layout

```
05-rag/
├── .env.example
├── config.py
├── ingestion/
│   ├── document_loader.py   # TextFile, PDF, Web, CSV, Directory loaders
│   ├── chunker.py           # Recursive, Sentence, Token splitters
│   └── embedder.py          # SentenceTransformer + LangChain wrapper
├── vectorstores/
│   ├── chromadb_store.py    # PersistentClient, cosine HNSW
│   ├── faiss_store.py       # IndexFlatIP, L2-normalised cosine
│   └── pinecone_store.py    # Serverless Pinecone, batched upsert
├── retrieval/
│   ├── retriever.py         # Dense, BM25, Hybrid (RRF)
│   └── reranker.py          # CrossEncoderReranker (ms-marco MiniLM)
├── generation/
│   ├── prompt_templates.py  # QA, Citation, Conversational, Structured
│   ├── chain.py             # LLMClient factory + RAGChain
│   └── structured_output.py # Pydantic schemas + instructor / JSON extractor
├── pipelines/
│   ├── local_rag.py         # LocalRAGPipeline (offline)
│   └── online_rag.py        # OnlineRAGPipeline (cloud)
└── examples/
    ├── local_rag_example.py
    └── pinecone_rag_example.py
```

---

## Quick start

### Local (offline)

```bash
pip install sentence-transformers chromadb faiss-cpu rank-bm25 ollama pymupdf beautifulsoup4 requests tiktoken
ollama pull llama3.2

python examples/local_rag_example.py
```

### Cloud (Pinecone + Groq)

```bash
pip install pinecone-client groq sentence-transformers rank-bm25

export PINECONE_API_KEY="..."
export LLM_PROVIDER="groq"
export LLM_API_KEY="..."

python examples/pinecone_rag_example.py
```

---

## Supported LLM providers

| Provider       | Install                           | Env var                 |
| -------------- | --------------------------------- | ----------------------- |
| Groq           | `pip install groq`                | `GROQ_API_KEY`          |
| Google Gemini  | `pip install google-generativeai` | `GEMINI_API_KEY`        |
| Cerebras       | `pip install cerebras-cloud-sdk`  | `CEREBRAS_API_KEY`      |
| Ollama (local) | `pip install ollama`              | _(none — runs locally)_ |
| OpenAI / vLLM  | `pip install openai`              | `OPENAI_API_KEY`        |

---

## Retrieval strategies

| Strategy    | Class                  | Description                                 |
| ----------- | ---------------------- | ------------------------------------------- |
| Dense       | `DenseRetriever`       | Semantic search via cosine similarity       |
| Sparse      | `BM25Retriever`        | Lexical keyword search (rank-bm25)          |
| Hybrid      | `HybridRetriever`      | RRF fusion of dense + sparse                |
| + Reranking | `CrossEncoderReranker` | Cross-encoder rescoring of top-k candidates |

### Reciprocal Rank Fusion score

$$\text{RRF}(d) = \sum_{\text{retriever}} \frac{w}{rank(d) + k}$$

where $k = 60$ (default) prevents high scores from dominating.

---

## Pipeline diagram

```
Documents
   │
   ▼
DirectoryLoader ──→ Document(page_content, metadata)
   │
   ▼
Chunker (recursive|sentence|token)
   │
   ▼
SentenceTransformerEmbedder
   │
   ▼
VectorStore (Chroma | FAISS | Pinecone)
   │           query time
   ▼
DenseRetriever ──┐
BM25Retriever ───┤ HybridRetriever (RRF)
                 │
                 ▼
         CrossEncoderReranker (optional)
                 │
                 ▼
         PromptTemplate.format(context, question)
                 │
                 ▼
         LLMClient (Groq|Gemini|Ollama|Cerebras|OpenAI)
                 │
                 ▼
         RAGResponse(answer, sources, scores)
```

---

## Key configuration (`.env`)

```dotenv
# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db

# FAISS
FAISS_INDEX_PATH=./faiss.index

# Pinecone
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=my-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# LLM
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.1-70b-versatile
OLLAMA_MODEL=llama3.2
GEMINI_API_KEY=...
CEREBRAS_API_KEY=...
```
