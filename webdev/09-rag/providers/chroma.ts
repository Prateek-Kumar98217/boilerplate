/**
 * ChromaVectorStore — local vector store using ChromaDB.
 *
 * Install: npm install chromadb
 * Start: npx chroma run --path ./chroma-data
 * Env: CHROMA_HOST=http://localhost:8000 (default)
 *
 * Collections are created automatically on first upsert.
 * Distances metric: cosine (change to "l2" or "ip" if needed).
 */
import { ChromaClient, Collection } from "chromadb";
import { RagError } from "../types";
import type {
  VectorStore,
  Document,
  UpsertResult,
  QueryInput,
  QueryResult,
} from "../types";

// ─── Client singleton ─────────────────────────────────────────────────────────

let _chroma: ChromaClient | null = null;

function getChromaClient(): ChromaClient {
  if (!_chroma) {
    const host = process.env.CHROMA_HOST ?? "http://localhost:8000";
    _chroma = new ChromaClient({ path: host });
  }
  return _chroma;
}

// ─── Collection cache (avoid redundant HTTP round-trips) ─────────────────────

const _collectionCache = new Map<string, Collection>();

async function getOrCreateCollection(name: string): Promise<Collection> {
  const cached = _collectionCache.get(name);
  if (cached) return cached;

  const client = getChromaClient();
  const collection = await client.getOrCreateCollection({
    name,
    metadata: { "hnsw:space": "cosine" },
  });
  _collectionCache.set(name, collection);
  return collection;
}

// ─── Implementation ───────────────────────────────────────────────────────────

export class ChromaVectorStore implements VectorStore {
  readonly provider = "chroma" as const;

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

    let collection: Collection;
    try {
      collection = await getOrCreateCollection(namespace);
    } catch (err) {
      throw new RagError(
        `Failed to get/create Chroma collection "${namespace}": ${String(err)}`,
        "UPSERT_FAILED",
        err,
      );
    }

    try {
      await collection.upsert({
        ids: documents.map((d) => d.id),
        embeddings: vectors,
        documents: documents.map((d) => d.content),
        metadatas: documents.map((d) => d.metadata),
      });
    } catch (err) {
      throw new RagError(
        `Chroma upsert failed: ${String(err)}`,
        "UPSERT_FAILED",
        err,
      );
    }

    return { upsertedCount: documents.length, namespace };
  }

  async query(input: QueryInput, queryVector: number[]): Promise<QueryResult> {
    let collection: Collection;
    try {
      collection = await getOrCreateCollection(input.namespace);
    } catch (err) {
      throw new RagError(
        `Chroma collection "${input.namespace}" not found.`,
        "NAMESPACE_NOT_FOUND",
        err,
      );
    }

    try {
      const result = await collection.query({
        queryEmbeddings: [queryVector],
        nResults: input.topK ?? 5,
        where: input.filter as Record<
          string,
          string | number | boolean | Record<string, unknown>
        >,
        include: ["documents", "metadatas", "distances"] as Parameters<
          typeof collection.query
        >[0]["include"],
      });

      const ids = result.ids[0] ?? [];
      const distances = result.distances?.[0] ?? [];
      const docs = result.documents[0] ?? [];
      const metas = result.metadatas[0] ?? [];

      return {
        matches: ids.map((id, i) => ({
          id,
          // Chroma returns cosine distance (0=identical, 2=opposite).
          // Convert to similarity score 0–1.
          score: 1 - (distances[i] ?? 0) / 2,
          content: docs[i] ?? "",
          metadata: (metas[i] ?? {}) as Record<
            string,
            string | number | boolean
          >,
        })),
      };
    } catch (err) {
      throw new RagError(
        `Chroma query failed: ${String(err)}`,
        "QUERY_FAILED",
        err,
      );
    }
  }

  async delete(namespace: string, ids: string[]): Promise<void> {
    try {
      const collection = await getOrCreateCollection(namespace);
      await collection.delete({ ids });
    } catch (err) {
      throw new RagError(
        `Chroma delete failed: ${String(err)}`,
        "UNKNOWN",
        err,
      );
    }
  }
}
