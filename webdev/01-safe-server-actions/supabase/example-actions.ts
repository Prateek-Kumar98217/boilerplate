/**
 * Example: Post CRUD actions using createSafeAction (Supabase variant).
 *
 * All DB operations go through the authenticated Supabase client in ctx.supabase.
 * Row Level Security (RLS) on the posts table enforces ownership automatically,
 * but we still do explicit checks where needed for clear error messages.
 */
"use server";

import { z } from "zod";
import { revalidatePath } from "next/cache";
import { createSafeAction, createActionError } from "./create-safe-action";
import type { SupabaseActionContext } from "./create-safe-action";

// ─── Supabase table row type (replace with generated types in production) ─────

type PostRow = {
  id: string;
  title: string;
  content: string;
  author_id: string;
  created_at: string;
  updated_at: string;
};

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
  title: z.string().min(3).max(255).optional(),
  content: z.string().min(10).max(50_000).optional(),
});

const DeletePostSchema = z.object({
  id: z.string().uuid("Invalid post ID."),
});

const GetPostSchema = z.object({
  id: z.string().uuid("Invalid post ID."),
});

// ─── Actions ─────────────────────────────────────────────────────────────────

export const createPost = createSafeAction(
  CreatePostSchema,
  async (input, ctx: SupabaseActionContext) => {
    const { data, error } = await ctx.supabase
      .from("posts")
      .insert({
        title: input.title,
        content: input.content,
        author_id: ctx.userId,
      })
      .select()
      .single<PostRow>();

    if (error) {
      // Supabase error is re-thrown and caught by the factory's error handler.
      throw error;
    }

    revalidatePath("/posts");
    return data;
  },
);

export const updatePost = createSafeAction(
  UpdatePostSchema,
  async (input, ctx: SupabaseActionContext) => {
    const updateData: Partial<Pick<PostRow, "title" | "content">> = {};
    if (input.title !== undefined) updateData.title = input.title;
    if (input.content !== undefined) updateData.content = input.content;

    if (Object.keys(updateData).length === 0) {
      throw createActionError.validation(
        "Provide at least one field to update.",
      );
    }

    // RLS will block updates to posts owned by other users.
    const { data, error } = await ctx.supabase
      .from("posts")
      .update(updateData)
      .eq("id", input.id)
      .eq("author_id", ctx.userId) // belt-and-suspenders alongside RLS
      .select()
      .single<PostRow>();

    if (error) throw error;

    revalidatePath("/posts");
    revalidatePath(`/posts/${input.id}`);
    return data;
  },
);

export const deletePost = createSafeAction(
  DeletePostSchema,
  async (input, ctx: SupabaseActionContext) => {
    const { error } = await ctx.supabase
      .from("posts")
      .delete()
      .eq("id", input.id)
      .eq("author_id", ctx.userId);

    if (error) throw error;

    revalidatePath("/posts");
    return { id: input.id };
  },
);

export const getPost = createSafeAction(
  GetPostSchema,
  async (input, ctx: SupabaseActionContext) => {
    const { data, error } = await ctx.supabase
      .from("posts")
      .select("*")
      .eq("id", input.id)
      .single<PostRow>();

    if (error || !data) {
      throw createActionError.notFound("Post not found.");
    }

    return data;
  },
  { requireAuth: false },
);
