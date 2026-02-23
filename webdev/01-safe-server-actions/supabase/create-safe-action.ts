/**
 * createSafeAction — Supabase variant.
 *
 * Key difference from the Drizzle variant:
 *  - Auth is verified via supabase.auth.getUser() (server-side, not a JWT decode).
 *    This is the ONLY safe way — getSession() should not be trusted server-side
 *    because it reads from cookies without re-validating with the Supabase server.
 *  - The same Supabase client instance is available inside the handler via context
 *    so handlers don't need to create their own client.
 *
 * Usage:
 *
 *   export const createPost = createSafeAction(CreatePostSchema, async (input, ctx) => {
 *     const { data, error } = await ctx.supabase
 *       .from("posts")
 *       .insert({ title: input.title, content: input.content, author_id: ctx.userId })
 *       .select()
 *       .single();
 *
 *     if (error) throw classifySupabaseError(error);
 *     return data;
 *   });
 */
"use server";

import { z } from "zod";
import { createSupabaseServerClient } from "./supabase-server";
import {
  ActionError,
  createActionError,
  classifySupabaseError,
} from "./action-error";
import type {
  ActionContext,
  ActionHandler,
  ActionResult,
  SafeActionOptions,
} from "./types";
import type { SupabaseClient } from "@supabase/supabase-js";
import type { Database } from "./database.types";

// ─── Extended context with Supabase client ────────────────────────────────────

export type SupabaseActionContext = ActionContext & {
  /** Pre-authenticated Supabase client — use this for all DB operations. */
  supabase: SupabaseClient<Database>;
};

export type SupabaseActionHandler<TInput, TData> = (
  validatedData: TInput,
  context: SupabaseActionContext,
) => Promise<TData>;

// ─── Factory ─────────────────────────────────────────────────────────────────

export function createSafeAction<TInput, TData>(
  schema: z.ZodType<TInput>,
  handler: SupabaseActionHandler<TInput, TData>,
  options: SafeActionOptions = {},
): (input: unknown) => Promise<ActionResult<TData>> {
  const { requireAuth = true } = options;

  return async (input: unknown): Promise<ActionResult<TData>> => {
    // ── Step 1: Validate input ────────────────────────────────────────────
    const parsed = schema.safeParse(input);

    if (!parsed.success) {
      const flat = parsed.error.flatten();
      return {
        success: false,
        error: "Validation failed. Please check the highlighted fields.",
        fieldErrors: flat.fieldErrors as Partial<Record<string, string[]>>,
        code: "VALIDATION_ERROR",
      };
    }

    // ── Step 2: Create Supabase client (reads cookies) ────────────────────
    const supabase = await createSupabaseServerClient();

    // ── Step 3: Check authentication ──────────────────────────────────────
    let context: SupabaseActionContext = {
      userId: "",
      userEmail: null,
      supabase,
    };

    if (requireAuth) {
      // IMPORTANT: Always use getUser(), not getSession().
      // getUser() makes a network call to Supabase to validate the JWT;
      // getSession() only reads from cookies and can be spoofed.
      const {
        data: { user },
        error: authError,
      } = await supabase.auth.getUser();

      if (authError || !user) {
        return {
          success: false,
          error: "You must be signed in to perform this action.",
          code: "UNAUTHORIZED",
        };
      }

      context = {
        userId: user.id,
        userEmail: user.email ?? null,
        supabase,
      };
    }

    // ── Step 4 & 5: Execute handler and handle errors ─────────────────────
    try {
      const data = await handler(parsed.data, context);
      return { success: true, data };
    } catch (err) {
      if (err instanceof ActionError) {
        return {
          success: false,
          error: err.message,
          fieldErrors: err.fieldErrors,
          code: err.code,
        };
      }

      // Supabase PostgREST errors have { code, message, details, hint }.
      if (
        typeof err === "object" &&
        err !== null &&
        "code" in err &&
        "message" in err &&
        typeof (err as { code: unknown }).code === "string"
      ) {
        const supaErr = err as { code: string; message: string };
        const code = classifySupabaseError(supaErr);
        const safeMessages: Record<string, string> = {
          CONFLICT: "A record with these details already exists.",
          NOT_FOUND: "The requested resource was not found.",
          FORBIDDEN: "You do not have permission to perform this action.",
          INTERNAL_ERROR: "Something went wrong. Please try again.",
        };
        return {
          success: false,
          error: safeMessages[code] ?? "Something went wrong.",
          code,
        };
      }

      console.error("[createSafeAction/supabase] Unhandled error:", err);
      return {
        success: false,
        error: "Something went wrong. Please try again.",
        code: "INTERNAL_ERROR",
      };
    }
  };
}

export { createActionError, ActionError, classifySupabaseError };
export type {
  ActionResult,
  ActionContext,
  SafeActionOptions,
  SupabaseActionContext,
};
