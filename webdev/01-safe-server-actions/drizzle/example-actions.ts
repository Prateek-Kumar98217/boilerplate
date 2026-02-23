/**
 * Example: Post CRUD actions using createSafeAction.
 *
 * This file is a "use server" module — every exported function becomes a
 * Next.js Server Action callable from Client Components.
 */
"use server";

import { eq, and } from "drizzle-orm";
import { z } from "zod";
import { revalidatePath } from "next/cache";
import { db } from "./db";
import { posts } from "./schema";
import { createSafeAction, createActionError } from "./create-safe-action";
import type { Post } from "./schema";

// ─── Zod schemas ─────────────────────────────────────────────────────────────

const CreatePostSchema = z.object({
  title: z
    .string()
    .min(3, "Title must be at least 3 characters.")
    .max(255, "Title cannot exceed 255 characters."),
  content: z
    .string()
    .min(10, "Content must be at least 10 characters.")
    .max(50_000, "Content cannot exceed 50 000 characters."),
});

const UpdatePostSchema = z.object({
  id: z.string().uuid("Invalid post ID."),
  title: z
    .string()
    .min(3, "Title must be at least 3 characters.")
    .max(255, "Title cannot exceed 255 characters.")
    .optional(),
  content: z
    .string()
    .min(10, "Content must be at least 10 characters.")
    .max(50_000, "Content cannot exceed 50 000 characters.")
    .optional(),
});

const DeletePostSchema = z.object({
  id: z.string().uuid("Invalid post ID."),
});

const GetPostSchema = z.object({
  id: z.string().uuid("Invalid post ID."),
});

// ─── Actions ─────────────────────────────────────────────────────────────────

/**
 * createPost — validates input, confirms auth, inserts row, revalidates cache.
 */
export const createPost = createSafeAction(
  CreatePostSchema,
  async (input, ctx) => {
    const [post] = await db
      .insert(posts)
      .values({
        title: input.title,
        content: input.content,
        authorId: ctx.userId,
      })
      .returning();

    if (!post) {
      throw createActionError.internal("Post could not be created.");
    }

    revalidatePath("/posts");
    return post;
  },
);

/**
 * updatePost — only the post's author can update it.
 */
export const updatePost = createSafeAction(
  UpdatePostSchema,
  async (input, ctx) => {
    // Confirm existence and ownership in a single query.
    const existing = await db.query.posts.findFirst({
      where: and(eq(posts.id, input.id), eq(posts.authorId, ctx.userId)),
    });

    if (!existing) {
      // Deliberately ambiguous: don't reveal whether the post exists but belongs
      // to someone else, or simply doesn't exist.
      throw createActionError.notFound("Post not found.");
    }

    const updateData: Partial<Pick<Post, "title" | "content">> = {};
    if (input.title !== undefined) updateData.title = input.title;
    if (input.content !== undefined) updateData.content = input.content;

    if (Object.keys(updateData).length === 0) {
      throw createActionError.validation(
        "Provide at least one field to update.",
      );
    }

    const [updated] = await db
      .update(posts)
      .set(updateData)
      .where(eq(posts.id, input.id))
      .returning();

    if (!updated) {
      throw createActionError.internal("Update failed.");
    }

    revalidatePath("/posts");
    revalidatePath(`/posts/${input.id}`);
    return updated;
  },
);

/**
 * deletePost — only the post's author can delete it.
 */
export const deletePost = createSafeAction(
  DeletePostSchema,
  async (input, ctx) => {
    const existing = await db.query.posts.findFirst({
      where: and(eq(posts.id, input.id), eq(posts.authorId, ctx.userId)),
    });

    if (!existing) {
      throw createActionError.notFound("Post not found.");
    }

    await db.delete(posts).where(eq(posts.id, input.id));

    revalidatePath("/posts");
    return { id: input.id };
  },
);

/**
 * getPost — public action, no auth required.
 */
export const getPost = createSafeAction(
  GetPostSchema,
  async (input) => {
    const post = await db.query.posts.findFirst({
      where: eq(posts.id, input.id),
    });

    if (!post) {
      throw createActionError.notFound("Post not found.");
    }

    return post;
  },
  { requireAuth: false },
);
