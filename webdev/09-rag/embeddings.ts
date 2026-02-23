/**
 * embeddings.ts — OpenAI text-embedding-3-small wrapper.
 *
 * Install: npm install openai
 * Env: OPENAI_API_KEY=sk-...
 *
 * Batching:
 *   OpenAI allows up to 2048 inputs per request. The helper automatically
 *   splits larger arrays into batches of BATCH_SIZE.
 *
 * Dimensions:
 *   text-embedding-3-small → 1536 dimensions by default.
 *   Pass `dimensions` to reduce (e.g. 512) for faster queries at lower recall.
 */
import OpenAI from "openai";
import { RagError } from "./types";
import type { EmbeddingResult } from "./types";

// ─── Config ───────────────────────────────────────────────────────────────────

const MODEL = "text-embedding-3-small";
const BATCH_SIZE = 512; // stay well below the 2048 limit

// ─── Client (lazy singleton) ──────────────────────────────────────────────────

let _client: OpenAI | null = null;
function getClient(): OpenAI {
  if (!_client) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new RagError(
        "OPENAI_API_KEY environment variable is not set.",
        "EMBEDDING_FAILED",
      );
    }
    _client = new OpenAI({ apiKey });
  }
  return _client;
}

// ─── Public helpers ───────────────────────────────────────────────────────────

/**
 * Embed a single string. Returns the vector and token usage.
 */
export async function embedText(
  text: string,
  dimensions?: number,
): Promise<EmbeddingResult> {
  const results = await embedBatch([text], dimensions);
  if (results.length === 0) {
    throw new RagError("Embedding returned no vectors.", "EMBEDDING_FAILED");
  }
  return results[0]!;
}

/**
 * Embed multiple strings in optimally-sized batches.
 * Returns one EmbeddingResult per input string in the same order.
 */
export async function embedBatch(
  texts: string[],
  dimensions?: number,
): Promise<EmbeddingResult[]> {
  if (texts.length === 0) return [];

  const client = getClient();
  const results: EmbeddingResult[] = [];

  for (let start = 0; start < texts.length; start += BATCH_SIZE) {
    const batch = texts.slice(start, start + BATCH_SIZE);

    let response: OpenAI.Embeddings.CreateEmbeddingResponse;
    try {
      response = await client.embeddings.create({
        model: MODEL,
        input: batch,
        ...(dimensions !== undefined ? { dimensions } : {}),
      });
    } catch (err) {
      throw new RagError(
        `OpenAI embeddings request failed: ${err instanceof Error ? err.message : String(err)}`,
        "EMBEDDING_FAILED",
        err,
      );
    }

    // Sort by index in case the API reorders them (it shouldn't, but be safe).
    const sorted = [...response.data].sort((a, b) => a.index - b.index);
    const totalTokens = response.usage.total_tokens;

    for (const item of sorted) {
      results.push({
        vector: item.embedding,
        totalTokens,
      });
    }
  }

  return results;
}
