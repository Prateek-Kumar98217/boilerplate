# 09 · RAG Utility

A `VectorStore` interface with **ChromaDB** (local) and **Pinecone** (cloud)
implementations, plus a single `upsertDocument()` entry point that handles
embedding generation.

Switch between providers with one function call. All embedding is done via
**OpenAI `text-embedding-3-small`** (1536-dimensional, batched).

---

## File map

```
09-rag/
├── types.ts                   VectorStore interface, Document, RagError
├── embeddings.ts              OpenAI batch embedding wrapper
├── upsert-document.ts         upsertDocument() + retrieveDocuments()
├── index.ts                   Barrel + createVectorStore() factory
└── providers/
    ├── chroma.ts              ChromaVectorStore (local)
    └── pinecone.ts            PineconeVectorStore (cloud)
```

---

## How it works

### 1. Upsert

```ts
import { createVectorStore, upsertDocument } from "@/lib/rag";

const store = createVectorStore(
  process.env.VECTOR_STORE === "pinecone" ? "pinecone" : "chroma",
);

await upsertDocument(
  {
    documents: [
      {
        id: "page-about",
        content: "We are a SaaS company building...",
        metadata: { source: "website", section: "about" },
      },
    ],
    namespace: "product-docs",
  },
  store,
);
```

Internally: `embedBatch([content])` → `store.upsert(namespace, docs, vectors)`.

### 2. Retrieve (RAG query)

```ts
import { createVectorStore, retrieveDocuments } from "@/lib/rag";

const store = createVectorStore("pinecone");
const results = await retrieveDocuments(
  "What does your company do?",
  "product-docs",
  store,
  5, // topK
);

// Pass results.matches to your LLM as context
for (const match of results.matches) {
  console.log(match.score, match.content);
}
```

### 3. Feed into LLM

```ts
import OpenAI from "openai";

const context = results.matches.map((m) => m.content).join("\n\n---\n\n");
const openai = new OpenAI();

const completion = await openai.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [
    { role: "system", content: `Answer based on this context:\n\n${context}` },
    { role: "user", content: "What does your company do?" },
  ],
});
```

---

## Installation

```bash
# Embeddings
npm install openai

# ChromaDB (local dev)
npm install chromadb
npx chroma run --path ./chroma-data

# Pinecone (production)
npm install @pinecone-database/pinecone
```

---

## Environment variables

| Variable           | Provider | Purpose                                               |
| ------------------ | -------- | ----------------------------------------------------- |
| `OPENAI_API_KEY`   | Both     | Embedding generation                                  |
| `CHROMA_HOST`      | Chroma   | ChromaDB server URL (default `http://localhost:8000`) |
| `PINECONE_API_KEY` | Pinecone | API auth                                              |
| `PINECONE_INDEX`   | Pinecone | Index name (must exist in dashboard)                  |
| `VECTOR_STORE`     | Both     | `"chroma"` (default) or `"pinecone"`                  |

---

## Chunking strategy (not included)

Documents should be chunked before calling `upsertDocument`. Common strategies:

- **Fixed-size**: 512 tokens with 50-token overlap
- **Semantic**: split on paragraph/section boundaries
- **Recursive**: LangChain `RecursiveCharacterTextSplitter`

Each chunk becomes one `Document` with a stable id (e.g. `"${sourceId}-chunk-${i}"`).

---

## What can go wrong

| Issue                                | Cause                                | Fix                                                                                            |
| ------------------------------------ | ------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `EMBEDDING_FAILED` on first call     | `OPENAI_API_KEY` not set             | Add to `.env.local`                                                                            |
| Chroma connection refused            | ChromaDB server not running          | `npx chroma run --path ./chroma-data`                                                          |
| Pinecone `NAMESPACE_NOT_FOUND`       | Wrong index name or no index created | Create index in Pinecone dashboard; set `PINECONE_INDEX`                                       |
| Scores all near 0                    | Querying with mismatched dimensions  | Pass the same `dimensions` value to both upsert and retrieve                                   |
| Duplicate inserts grow the index     | Same `id` upserted twice             | Upsert is idempotent by id — safe to re-run, old vector is replaced                            |
| Slow batch embedding                 | Sending 1000s of docs in one call    | The embedder already batches at 512; further chunking upstream reduces single-doc token counts |
| Metadata values rejected by Pinecone | Object/array values in metadata      | Flatten to `string \| number \| boolean` before upsert                                         |
