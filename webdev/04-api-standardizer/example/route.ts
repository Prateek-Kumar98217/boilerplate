/**
 * Example API route: /api/posts
 *
 * GET  /api/posts        — list posts (paginated, searchable)
 * POST /api/posts        — create a post (requires auth)
 *
 * Every response is guaranteed to match ApiResponse<T>.
 */
import { NextRequest } from "next/server";
import { z } from "zod";
import {
  withApiHandler,
  parseBody,
  parseSearchParams,
  ok,
  created,
  ApiErrors,
} from "../lib";

// ─── Schemas ──────────────────────────────────────────────────────────────────

const ListQuerySchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  pageSize: z.coerce.number().int().min(1).max(100).default(20),
  search: z.string().trim().optional(),
});

const CreatePostBodySchema = z.object({
  title: z.string().min(3, "Title must be at least 3 characters.").max(255),
  content: z.string().min(10, "Content must be at least 10 characters."),
});

// ─── Fake data store (replace with real DB calls) ─────────────────────────────

type PostRecord = {
  id: string;
  title: string;
  content: string;
  authorId: string;
};

const fakePosts: PostRecord[] = [
  {
    id: "1",
    title: "Hello World",
    content: "My first post.",
    authorId: "user-1",
  },
  {
    id: "2",
    title: "TypeScript Tips",
    content: "Always use strict mode.",
    authorId: "user-1",
  },
];

// ─── GET /api/posts ───────────────────────────────────────────────────────────

export const GET = withApiHandler(async (req: NextRequest) => {
  const query = parseSearchParams(req, ListQuerySchema);

  const filtered = query.search
    ? fakePosts.filter((p) =>
        p.title.toLowerCase().includes(query.search!.toLowerCase()),
      )
    : fakePosts;

  const start = (query.page - 1) * query.pageSize;
  const items = filtered.slice(start, start + query.pageSize);
  const total = filtered.length;
  const totalPages = Math.ceil(total / query.pageSize);

  return ok(items, {
    pagination: {
      page: query.page,
      pageSize: query.pageSize,
      total,
      totalPages,
      hasNextPage: query.page < totalPages,
      hasPreviousPage: query.page > 1,
    },
  });
});

// ─── POST /api/posts ──────────────────────────────────────────────────────────

export const POST = withApiHandler(
  async (req: NextRequest, ctx) => {
    const body = await parseBody(req, CreatePostBodySchema);

    const newPost: PostRecord = {
      id: crypto.randomUUID(),
      title: body.title,
      content: body.content,
      authorId: ctx.userId,
    };

    // In a real app: await db.insert(posts).values(newPost);
    fakePosts.push(newPost);

    return created(newPost, { message: "Post created successfully." });
  },
  { requireAuth: true },
);
