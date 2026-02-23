/**
 * upsertDocument — single entry point for embedding + storing documents.
 *
 * Both ChromaDB and Pinecone are supported; switch by passing a different
 * VectorStore instance. Use createVectorStore() from index.ts to build one.
 *
 * Flow:
 *   1. Validate input
 *   2. Batch-embed all document contents via OpenAI (text-embedding-3-small)
 *   3. Upsert vectors into the chosen store
 *   4. Return count of upserted documents
 *
 * Usage:
 *   const store = createVectorStore("pinecone");
 *   const result = await upsertDocument({
 *     documents: [{ id: "doc-1", content: "Hello world", metadata: {} }],
 *     namespace: "my-collection",
 *   }, store);
 */
import { embedBatch } from "./embeddings";
import { RagError } from "./types";
import type { UpsertDocumentInput, UpsertResult, VectorStore } from "./types";

/**
 * Embed documents and store them in the given vector store.
 *
 * @param input    - documents + namespace to upsert into
 * @param store    - ChromaVectorStore | PineconeVectorStore
 * @param dimensions - optional embedding dimension reduction (OpenAI only)
 */
export async function upsertDocument(
  input: UpsertDocumentInput,
  store: VectorStore,
  dimensions?: number,
): Promise<UpsertResult> {
  if (input.documents.length === 0) {
    return { upsertedCount: 0, namespace: input.namespace };
  }

  // Validate: all docs must have non-empty content
  const emptyContentIdx = input.documents.findIndex((d) => !d.content.trim());
  if (emptyContentIdx !== -1) {
    throw new RagError(
      `Document at index ${emptyContentIdx} has empty content.`,
      "INVALID_INPUT",
    );
  }

  // Validate: ids must be unique within the batch
  const ids = input.documents.map((d) => d.id);
  const uniqueIds = new Set(ids);
  if (uniqueIds.size !== ids.length) {
    throw new RagError(
      "Duplicate document ids found in the input batch.",
      "INVALID_INPUT",
    );
  }

  // Embed all documents in one batched call
  const embeddings = await embedBatch(
    input.documents.map((d) => d.content),
    dimensions,
  );

  const vectors = embeddings.map((e) => e.vector);

  // Store in the chosen vector store
  return store.upsert(input.namespace, input.documents, vectors);
}

/**
 * Convenience function: embed a query and retrieve top-K matches.
 *
 * @param query      - natural-language question
 * @param namespace  - collection / Pinecone namespace to search
 * @param store      - the vector store to query
 * @param topK       - number of results (default 5)
 * @param dimensions - must match the dimensions used during upsert
 */
export async function retrieveDocuments(
  query: string,
  namespace: string,
  store: VectorStore,
  topK = 5,
  dimensions?: number,
) {
  if (!query.trim()) {
    throw new RagError("Query string cannot be empty.", "INVALID_INPUT");
  }

  const [embedding] = await embedBatch([query], dimensions);
  if (!embedding) {
    throw new RagError("Failed to embed query.", "EMBEDDING_FAILED");
  }

  return store.query({ query, namespace, topK }, embedding.vector);
}
