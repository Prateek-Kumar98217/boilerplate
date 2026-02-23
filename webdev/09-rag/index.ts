/**
 * index.ts — barrel export + factory for the RAG utility.
 *
 * Usage:
 *   // Pick a provider via env var (defaults to chroma for local dev)
 *   const store = createVectorStore(
 *     process.env.VECTOR_STORE === "pinecone" ? "pinecone" : "chroma"
 *   );
 *
 *   await upsertDocument({
 *     documents: [{ id: "page-1", content: "...", metadata: { source: "docs" } }],
 *     namespace: "product-docs",
 *   }, store);
 *
 *   const results = await retrieveDocuments("How do I reset my password?", "product-docs", store);
 */
export type {
  VectorStore,
  Document,
  EmbeddingResult,
  UpsertDocumentInput,
  UpsertResult,
  QueryInput,
  QueryResult,
  QueryMatch,
  RagErrorCode,
} from "./types";
export { RagError } from "./types";
export { embedText, embedBatch } from "./embeddings";
export { ChromaVectorStore } from "./providers/chroma";
export { PineconeVectorStore } from "./providers/pinecone";
export { upsertDocument, retrieveDocuments } from "./upsert-document";

import { ChromaVectorStore } from "./providers/chroma";
import { PineconeVectorStore } from "./providers/pinecone";
import type { VectorStore } from "./types";

// ─── Factory ──────────────────────────────────────────────────────────────────

/**
 * Instantiate a VectorStore for the named provider.
 *
 * "chroma"  → requires CHROMA_HOST (default http://localhost:8000)
 * "pinecone" → requires PINECONE_API_KEY + PINECONE_INDEX
 */
export function createVectorStore(
  provider: "chroma" | "pinecone",
): VectorStore {
  switch (provider) {
    case "chroma":
      return new ChromaVectorStore();
    case "pinecone":
      return new PineconeVectorStore();
    default: {
      const _unreachable: never = provider;
      throw new Error(`Unknown vector store provider: ${String(_unreachable)}`);
    }
  }
}
