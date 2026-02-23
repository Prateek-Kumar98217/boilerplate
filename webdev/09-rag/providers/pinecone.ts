/**
 * PineconeVectorStore — cloud vector store using Pinecone.
 *
 * Install: npm install @pinecone-database/pinecone
 * Env:
 *   PINECONE_API_KEY=pcsk_...
 *   PINECONE_INDEX=your-index-name   (must already exist in Pinecone dashboard)
 *
 * Notes:
 *  - Pinecone uses "namespaces" to partition vectors within a single index.
 *    The `namespace` arg in upsert / query maps to a Pinecone namespace.
 *  - Serverless indexes do not support pod-based operations; pod indexes do.
 *    This implementation targets serverless (the default in the current SDK).
 *  - Metadata values must be strings, numbers, or booleans (no arrays/objects).
 */
import { Pinecone } from "@pinecone-database/pinecone";
import { RagError } from "../types";
import type {
  VectorStore,
  Document,
  UpsertResult,
  QueryInput,
  QueryResult,
} from "../types";

// ─── Client singleton ─────────────────────────────────────────────────────────

let _pinecone: Pinecone | null = null;

function getPineconeClient(): Pinecone {
  if (!_pinecone) {
    const apiKey = process.env.PINECONE_API_KEY;
    if (!apiKey) {
      throw new RagError(
        "PINECONE_API_KEY environment variable is not set.",
        "UPSERT_FAILED",
      );
    }
    _pinecone = new Pinecone({ apiKey });
  }
  return _pinecone;
}

function getIndex() {
  const indexName = process.env.PINECONE_INDEX;
  if (!indexName) {
    throw new RagError(
      "PINECONE_INDEX environment variable is not set.",
      "NAMESPACE_NOT_FOUND",
    );
  }
  return getPineconeClient().index(indexName);
}

// ─── Batch helpers ────────────────────────────────────────────────────────────

// Pinecone recommends batches ≤ 100 vectors
const PINECONE_BATCH_SIZE = 100;

function chunk<T>(arr: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    chunks.push(arr.slice(i, i + size));
  }
  return chunks;
}

// ─── Implementation ───────────────────────────────────────────────────────────

export class PineconeVectorStore implements VectorStore {
  readonly provider = "pinecone" as const;

  async upsert(
    namespace: string,
    documents: Document[],
    vectors: number[][],
  ): Promise<UpsertResult> {
    if (documents.length !== vectors.length) {
      throw new RagError(
        "documents and vectors arrays must have the same length.",
        "INVALID_INPUT",
      );
    }

    const index = getIndex().namespace(namespace);

    // Build Pinecone vector records
    const records = documents.map((doc, i) => ({
      id: doc.id,
      values: vectors[i]!,
      metadata: {
        content: doc.content,
        ...doc.metadata,
      },
    }));

    const batches = chunk(records, PINECONE_BATCH_SIZE);

    try {
      for (const batch of batches) {
        await index.upsert(batch);
      }
    } catch (err) {
      throw new RagError(
        `Pinecone upsert failed: ${String(err)}`,
        "UPSERT_FAILED",
        err,
      );
    }

    return { upsertedCount: documents.length, namespace };
  }

  async query(input: QueryInput, queryVector: number[]): Promise<QueryResult> {
    const index = getIndex().namespace(input.namespace);

    try {
      const response = await index.query({
        vector: queryVector,
        topK: input.topK ?? 5,
        filter: input.filter,
        includeMetadata: true,
        includeValues: false,
      });

      return {
        matches: (response.matches ?? []).map((match) => {
          const meta = (match.metadata ?? {}) as Record<
            string,
            string | number | boolean
          >;
          // Extract the content we stored alongside the metadata
          const content =
            typeof meta["content"] === "string" ? meta["content"] : "";
          const { content: _content, ...restMeta } = meta;
          void _content;

          return {
            id: match.id,
            score: match.score ?? 0,
            content,
            metadata: restMeta,
          };
        }),
      };
    } catch (err) {
      throw new RagError(
        `Pinecone query failed: ${String(err)}`,
        "QUERY_FAILED",
        err,
      );
    }
  }

  async delete(namespace: string, ids: string[]): Promise<void> {
    try {
      const index = getIndex().namespace(namespace);
      await index.deleteMany(ids);
    } catch (err) {
      throw new RagError(
        `Pinecone delete failed: ${String(err)}`,
        "UNKNOWN",
        err,
      );
    }
  }
}
