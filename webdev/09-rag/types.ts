/**
 * Shared types for the RAG (Retrieval-Augmented Generation) utility.
 *
 * One abstraction layer: VectorStore interface + two implementations:
 *   - ChromaVectorStore (local, dev-friendly, no API key)
 *   - PineconeVectorStore (cloud, production-ready)
 *
 * The single public entry point is upsertDocument() in upsert-document.ts.
 */

// ─── Document ─────────────────────────────────────────────────────────────────

/** A chunk of text ready to embed and store. */
export interface Document {
  /** Stable, unique identifier — used for upsert deduplication. */
  id: string;
  /** The text content to embed. */
  content: string;
  /** Arbitrary key/value pairs stored alongside the embedding for filtering. */
  metadata: Record<string, string | number | boolean>;
}

// ─── Embeddings ───────────────────────────────────────────────────────────────

export interface EmbeddingResult {
  vector: number[];
  /** Token count reported by the embedding provider. */
  totalTokens: number;
}

// ─── Upsert ───────────────────────────────────────────────────────────────────

export interface UpsertDocumentInput {
  /** Documents to embed and store. Multiple docs are batched in one API call. */
  documents: Document[];
  /**
   * Name of the collection / index to upsert into.
   * Chroma: collection name. Pinecone: index name (must exist).
   */
  namespace: string;
}

export interface UpsertResult {
  upsertedCount: number;
  namespace: string;
}

// ─── Query ────────────────────────────────────────────────────────────────────

export interface QueryInput {
  /** Natural-language query — will be embedded before search. */
  query: string;
  namespace: string;
  /** Maximum number of results to return. Default 5. */
  topK?: number;
  /** Metadata filter (provider-dependent syntax). */
  filter?: Record<string, string | number | boolean>;
}

export interface QueryMatch {
  id: string;
  score: number;
  content: string;
  metadata: Record<string, string | number | boolean>;
}

export interface QueryResult {
  matches: QueryMatch[];
}

// ─── VectorStore interface ────────────────────────────────────────────────────

export interface VectorStore {
  readonly provider: "chroma" | "pinecone";

  /**
   * Upsert pre-computed embeddings into the store.
   * Documents are indexed by `id`; existing documents are overwritten.
   */
  upsert(
    namespace: string,
    documents: Document[],
    vectors: number[][],
  ): Promise<UpsertResult>;

  /**
   * Retrieve the top-K closest documents to a pre-computed query vector.
   */
  query(input: QueryInput, queryVector: number[]): Promise<QueryResult>;

  /**
   * Delete documents by id from a namespace.
   */
  delete(namespace: string, ids: string[]): Promise<void>;
}

// ─── Error ────────────────────────────────────────────────────────────────────

export type RagErrorCode =
  | "EMBEDDING_FAILED"
  | "UPSERT_FAILED"
  | "QUERY_FAILED"
  | "NAMESPACE_NOT_FOUND"
  | "INVALID_INPUT"
  | "NETWORK_ERROR"
  | "UNKNOWN";

export class RagError extends Error {
  constructor(
    message: string,
    public readonly code: RagErrorCode,
    public readonly cause?: unknown,
  ) {
    super(message);
    this.name = "RagError";
  }
}
