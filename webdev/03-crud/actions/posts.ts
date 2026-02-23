/**
 * Post CRUD server actions.
 * Full create/read/update/delete with pagination, auth ownership, and error handling.
 */
"use server";

import { eq, and, ilike, count, desc } from "drizzle-orm";
import { z } from "zod";
import { revalidatePath } from "next/cache";
import { db } from "@/db"; // adjust import path to your db client
import { posts } from "../db/schema";
import type { Post } from "../db/schema";
import {
  requireAuth,
  runAction,
  validateInput,
  buildPaginatedResult,
  PaginationSchema,
  CrudError,
} from "./base";

// ─── Schemas ──────────────────────────────────────────────────────────────────

export const CreatePostSchema = z.object({
  title: z.string().min(3, "Title needs at least 3 characters.").max(255),
  slug: z
    .string()
    .min(3)
    .max(300)
    .regex(
      /^[a-z0-9-]+$/,
      "Slug may only contain lowercase letters, numbers, and hyphens.",
    ),
  content: z.string().min(10, "Content needs at least 10 characters."),
  excerpt: z
    .string()
    .max(500, "Excerpt cannot exceed 500 characters.")
    .optional(),
  publishedAt: z.string().datetime().optional().nullable(),
});

export const UpdatePostSchema = z.object({
  id: z.string().uuid("Invalid post ID."),
  title: z.string().min(3).max(255).optional(),
  slug: z
    .string()
    .min(3)
    .max(300)
    .regex(
      /^[a-z0-9-]+$/,
      "Slug may only contain lowercase letters, numbers, and hyphens.",
    )
    .optional(),
  content: z.string().min(10).optional(),
  excerpt: z.string().max(500).optional().nullable(),
  publishedAt: z.string().datetime().optional().nullable(),
});

export const ListPostsSchema = PaginationSchema.extend({
  authorId: z.string().uuid().optional(),
});

// ─── List posts (public, paginated, optional search & filter) ─────────────────

export async function listPosts(input: unknown) {
  const validated = validateInput(ListPostsSchema, input);
  if (!validated.success) return validated;

  return runAction(async () => {
    const { page, pageSize, search, authorId } = validated.data;
    const offset = (page - 1) * pageSize;

    // Build where conditions dynamically.
    const conditions = [];
    if (search) conditions.push(ilike(posts.title, `%${search}%`));
    if (authorId) conditions.push(eq(posts.authorId, authorId));

    const whereClause = conditions.length > 0 ? and(...conditions) : undefined;

    const [rows, [countRow]] = await Promise.all([
      db
        .select()
        .from(posts)
        .where(whereClause)
        .orderBy(desc(posts.createdAt))
        .limit(pageSize)
        .offset(offset),
      db.select({ count: count() }).from(posts).where(whereClause),
    ]);

    const total = Number(countRow?.count ?? 0);
    return buildPaginatedResult(rows, total, { page, pageSize });
  });
}

// ─── Get single post ──────────────────────────────────────────────────────────

export async function getPostById(id: unknown) {
  const validated = validateInput(z.string().uuid("Invalid post ID."), id);
  if (!validated.success) return validated;

  return runAction(async () => {
    const post = await db.query.posts.findFirst({
      where: eq(posts.id, validated.data),
      with: { author: { columns: { id: true, name: true, avatarUrl: true } } },
    });

    if (!post) throw new CrudError("Post not found.", "NOT_FOUND");
    return post;
  });
}

// ─── Get post by slug ─────────────────────────────────────────────────────────

export async function getPostBySlug(slug: unknown) {
  const validated = validateInput(z.string().min(1), slug);
  if (!validated.success) return validated;

  return runAction(async () => {
    const post = await db.query.posts.findFirst({
      where: eq(posts.slug, validated.data),
      with: {
        author: { columns: { id: true, name: true, avatarUrl: true } },
        postTags: { with: { tag: true } },
      },
    });

    if (!post) throw new CrudError("Post not found.", "NOT_FOUND");
    return post;
  });
}

// ─── Create ───────────────────────────────────────────────────────────────────

export async function createPost(input: unknown) {
  const validated = validateInput(CreatePostSchema, input);
  if (!validated.success) return validated;

  return runAction(async () => {
    const { userId } = await requireAuth();

    const [post] = await db
      .insert(posts)
      .values({
        ...validated.data,
        authorId: userId,
        publishedAt: validated.data.publishedAt
          ? new Date(validated.data.publishedAt)
          : null,
      })
      .returning();

    if (!post)
      throw new CrudError("Post could not be created.", "INTERNAL_ERROR");

    revalidatePath("/posts");
    return post;
  });
}

// ─── Update ───────────────────────────────────────────────────────────────────

export async function updatePost(input: unknown) {
  const validated = validateInput(UpdatePostSchema, input);
  if (!validated.success) return validated;

  return runAction(async () => {
    const { userId } = await requireAuth();
    const { id, ...fields } = validated.data;

    // Ensure the post exists and belongs to this user.
    const existing = await db.query.posts.findFirst({
      where: and(eq(posts.id, id), eq(posts.authorId, userId)),
    });
    if (!existing) throw new CrudError("Post not found.", "NOT_FOUND");

    // Build a clean update payload — only include provided fields.
    const updateData: Partial<Post> = {};
    if (fields.title !== undefined) updateData.title = fields.title;
    if (fields.slug !== undefined) updateData.slug = fields.slug;
    if (fields.content !== undefined) updateData.content = fields.content;
    if ("excerpt" in fields) updateData.excerpt = fields.excerpt ?? null;
    if ("publishedAt" in fields) {
      updateData.publishedAt = fields.publishedAt
        ? new Date(fields.publishedAt)
        : null;
    }

    if (Object.keys(updateData).length === 0) {
      throw new CrudError(
        "Provide at least one field to update.",
        "VALIDATION_ERROR",
      );
    }

    const [updated] = await db
      .update(posts)
      .set(updateData)
      .where(eq(posts.id, id))
      .returning();

    if (!updated) throw new CrudError("Update failed.", "INTERNAL_ERROR");

    revalidatePath("/posts");
    revalidatePath(`/posts/${updated.slug}`);
    return updated;
  });
}

// ─── Delete ───────────────────────────────────────────────────────────────────

export async function deletePost(id: unknown) {
  const validated = validateInput(z.string().uuid("Invalid post ID."), id);
  if (!validated.success) return validated;

  return runAction(async () => {
    const { userId } = await requireAuth();

    const existing = await db.query.posts.findFirst({
      where: and(eq(posts.id, validated.data), eq(posts.authorId, userId)),
    });
    if (!existing) throw new CrudError("Post not found.", "NOT_FOUND");

    await db.delete(posts).where(eq(posts.id, validated.data));

    revalidatePath("/posts");
    return { id: validated.data };
  });
}
