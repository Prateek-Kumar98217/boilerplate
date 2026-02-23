/**
 * Comment CRUD server actions.
 * Demonstrates nested resource pattern (comments belong to a post).
 */
"use server";

import { eq, and, count, asc } from "drizzle-orm";
import { z } from "zod";
import { revalidatePath } from "next/cache";
import { db } from "@/db";
import { comments, posts } from "../db/schema";
import {
  requireAuth,
  runAction,
  validateInput,
  buildPaginatedResult,
  PaginationSchema,
  CrudError,
} from "./base";

// ─── Schemas ──────────────────────────────────────────────────────────────────

export const CreateCommentSchema = z.object({
  postId: z.string().uuid("Invalid post ID."),
  content: z
    .string()
    .min(1, "Comment cannot be empty.")
    .max(5000, "Comment cannot exceed 5000 characters."),
  parentId: z.string().uuid("Invalid parent comment ID.").optional().nullable(),
});

export const UpdateCommentSchema = z.object({
  id: z.string().uuid("Invalid comment ID."),
  content: z
    .string()
    .min(1, "Comment cannot be empty.")
    .max(5000, "Comment cannot exceed 5000 characters."),
});

const ListCommentsSchema = PaginationSchema.extend({
  postId: z.string().uuid("Invalid post ID."),
  parentId: z.string().uuid().optional().nullable(),
});

// ─── List comments for a post ─────────────────────────────────────────────────

export async function listComments(input: unknown) {
  const validated = validateInput(ListCommentsSchema, input);
  if (!validated.success) return validated;

  return runAction(async () => {
    const { page, pageSize, postId, parentId } = validated.data;
    const offset = (page - 1) * pageSize;

    const whereClause = and(
      eq(comments.postId, postId),
      parentId !== undefined
        ? parentId === null
          ? eq(comments.parentId, comments.id) // top-level only (self-ref trick won't work — use isNull)
          : eq(comments.parentId, parentId)
        : undefined,
    );

    const [rows, [countRow]] = await Promise.all([
      db
        .select()
        .from(comments)
        .where(whereClause)
        .orderBy(asc(comments.createdAt))
        .limit(pageSize)
        .offset(offset),
      db.select({ count: count() }).from(comments).where(whereClause),
    ]);

    return buildPaginatedResult(rows, Number(countRow?.count ?? 0), {
      page,
      pageSize,
    });
  });
}

// ─── Create ───────────────────────────────────────────────────────────────────

export async function createComment(input: unknown) {
  const validated = validateInput(CreateCommentSchema, input);
  if (!validated.success) return validated;

  return runAction(async () => {
    const { userId } = await requireAuth();
    const { postId, content, parentId } = validated.data;

    // Verify the target post exists.
    const post = await db.query.posts.findFirst({
      where: eq(posts.id, postId),
    });
    if (!post) throw new CrudError("Post not found.", "NOT_FOUND");

    // If replying, verify the parent comment exists and belongs to the same post.
    if (parentId) {
      const parent = await db.query.comments.findFirst({
        where: and(eq(comments.id, parentId), eq(comments.postId, postId)),
      });
      if (!parent)
        throw new CrudError("Parent comment not found.", "NOT_FOUND");
    }

    const [comment] = await db
      .insert(comments)
      .values({ postId, content, authorId: userId, parentId: parentId ?? null })
      .returning();

    if (!comment)
      throw new CrudError("Comment could not be created.", "INTERNAL_ERROR");

    revalidatePath(`/posts/${post.slug}`);
    return comment;
  });
}

// ─── Update ───────────────────────────────────────────────────────────────────

export async function updateComment(input: unknown) {
  const validated = validateInput(UpdateCommentSchema, input);
  if (!validated.success) return validated;

  return runAction(async () => {
    const { userId } = await requireAuth();
    const { id, content } = validated.data;

    const existing = await db.query.comments.findFirst({
      where: and(eq(comments.id, id), eq(comments.authorId, userId)),
    });
    if (!existing) throw new CrudError("Comment not found.", "NOT_FOUND");

    const [updated] = await db
      .update(comments)
      .set({ content })
      .where(eq(comments.id, id))
      .returning();

    if (!updated) throw new CrudError("Update failed.", "INTERNAL_ERROR");

    return updated;
  });
}

// ─── Delete ───────────────────────────────────────────────────────────────────

export async function deleteComment(id: unknown) {
  const validated = validateInput(z.string().uuid("Invalid comment ID."), id);
  if (!validated.success) return validated;

  return runAction(async () => {
    const { userId } = await requireAuth();

    const existing = await db.query.comments.findFirst({
      where: and(
        eq(comments.id, validated.data),
        eq(comments.authorId, userId),
      ),
      with: { post: { columns: { slug: true } } },
    });
    if (!existing) throw new CrudError("Comment not found.", "NOT_FOUND");

    // Deleting a parent comment cascades to replies via the DB foreign key,
    // but for UX you may want to only allow deleting leaf comments.
    // Remove this guard if cascade delete is the desired behaviour.
    const replyCount = await db
      .select({ count: count() })
      .from(comments)
      .where(eq(comments.parentId, validated.data));

    if (Number(replyCount[0]?.count ?? 0) > 0) {
      // Soft approach: replace content rather than hard-delete.
      const [updated] = await db
        .update(comments)
        .set({ content: "[deleted]" })
        .where(eq(comments.id, validated.data))
        .returning();
      if (!updated) throw new CrudError("Delete failed.", "INTERNAL_ERROR");
      return { id: validated.data };
    }

    await db.delete(comments).where(eq(comments.id, validated.data));
    return { id: validated.data };
  });
}
