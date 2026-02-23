/**
 * Example API route: /api/posts/[id]
 *
 * GET    /api/posts/[id]   — get single post (public)
 * PATCH  /api/posts/[id]   — update a post (requires auth + ownership)
 * DELETE /api/posts/[id]   — delete a post (requires auth + ownership)
 */
import { NextRequest } from "next/server";
import { z } from "zod";
import { withApiHandler, parseBody, ok, noContent, ApiErrors } from "../lib";
import type { RouteContext } from "../lib";

// ─── Schema ───────────────────────────────────────────────────────────────────

const UpdateBodySchema = z.object({
  title: z.string().min(3).max(255).optional(),
  content: z.string().min(10).optional(),
});

// ─── Fake data store ──────────────────────────────────────────────────────────

type PostRecord = {
  id: string;
  title: string;
  content: string;
  authorId: string;
};
const fakePosts: PostRecord[] = [
  {
    id: "abc-123",
    title: "Hello World",
    content: "First post.",
    authorId: "user-1",
  },
];

// ─── GET /api/posts/[id] ──────────────────────────────────────────────────────

export const GET = withApiHandler(
  async (_req: NextRequest, ctx: RouteContext) => {
    const id = ctx.params["id"];
    if (typeof id !== "string") throw ApiErrors.badRequest("Invalid post ID.");

    const post = fakePosts.find((p) => p.id === id);
    if (!post) throw ApiErrors.notFound("Post");

    return ok(post);
  },
);

// ─── PATCH /api/posts/[id] ────────────────────────────────────────────────────

export const PATCH = withApiHandler(
  async (req: NextRequest, ctx) => {
    const id = ctx.params["id"];
    if (typeof id !== "string") throw ApiErrors.badRequest("Invalid post ID.");

    const postIndex = fakePosts.findIndex((p) => p.id === id);
    if (postIndex === -1) throw ApiErrors.notFound("Post");

    const post = fakePosts[postIndex];

    // Ownership check — only the author can update.
    if (post?.authorId !== ctx.userId) throw ApiErrors.forbidden();

    const body = await parseBody(req, UpdateBodySchema);

    const updated: PostRecord = {
      ...post,
      ...(body.title !== undefined ? { title: body.title } : {}),
      ...(body.content !== undefined ? { content: body.content } : {}),
    };

    fakePosts[postIndex] = updated;

    return ok(updated, { message: "Post updated." });
  },
  { requireAuth: true },
);

// ─── DELETE /api/posts/[id] ───────────────────────────────────────────────────

export const DELETE = withApiHandler(
  async (_req: NextRequest, ctx) => {
    const id = ctx.params["id"];
    if (typeof id !== "string") throw ApiErrors.badRequest("Invalid post ID.");

    const postIndex = fakePosts.findIndex((p) => p.id === id);
    if (postIndex === -1) throw ApiErrors.notFound("Post");

    const post = fakePosts[postIndex];
    if (post?.authorId !== ctx.userId) throw ApiErrors.forbidden();

    fakePosts.splice(postIndex, 1);

    return noContent();
  },
  { requireAuth: true },
);
