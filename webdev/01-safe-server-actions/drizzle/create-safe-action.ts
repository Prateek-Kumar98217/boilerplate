/**
 * createSafeAction — the core factory.
 *
 * Usage (Next.js App Router, "use server" file):
 *
 *   export const createPost = createSafeAction(CreatePostSchema, async (input, ctx) => {
 *     const [post] = await db.insert(posts).values({ ...input, authorId: ctx.userId }).returning();
 *     if (!post) throw createActionError.internal("Insert returned nothing.");
 *     return post;
 *   });
 *
 * The returned function can be called directly from a Client Component via
 * `startTransition` + `useActionState` (React 19) or a plain async call.
 */
"use server";

import { z } from "zod";
import { auth } from "@/auth"; // Next Auth v5 — swap for your own session helper
import { ActionError, createActionError } from "./action-error";
import type {
  ActionContext,
  ActionHandler,
  ActionResult,
  SafeActionOptions,
} from "./types";

// ─── Factory ─────────────────────────────────────────────────────────────────

export function createSafeAction<TInput, TData>(
  schema: z.ZodType<TInput>,
  handler: ActionHandler<TInput, TData>,
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

    // ── Step 2: Check authentication ──────────────────────────────────────
    let context: ActionContext = { userId: "", userEmail: null };

    if (requireAuth) {
      const session = await auth();

      if (!session?.user?.id) {
        return {
          success: false,
          error: "You must be signed in to perform this action.",
          code: "UNAUTHORIZED",
        };
      }

      context = {
        userId: session.user.id,
        userEmail: session.user.email ?? null,
      };
    }

    // ── Step 3 & 4: Execute handler and handle errors ─────────────────────
    try {
      const data = await handler(parsed.data, context);
      return { success: true, data };
    } catch (err) {
      // Known, intentional errors thrown by the handler.
      if (err instanceof ActionError) {
        return {
          success: false,
          error: err.message,
          fieldErrors: err.fieldErrors,
          code: err.code,
        };
      }

      // Drizzle / node-postgres unique constraint violation.
      if (
        typeof err === "object" &&
        err !== null &&
        "code" in err &&
        (err as { code: unknown }).code === "23505"
      ) {
        return {
          success: false,
          error: "A record with these details already exists.",
          code: "CONFLICT",
        };
      }

      // Drizzle / node-postgres foreign-key violation.
      if (
        typeof err === "object" &&
        err !== null &&
        "code" in err &&
        (err as { code: unknown }).code === "23503"
      ) {
        return {
          success: false,
          error: "A related record was not found.",
          code: "NOT_FOUND",
        };
      }

      // Log unexpected errors server-side; never expose to client.
      console.error("[createSafeAction] Unhandled error:", err);

      return {
        success: false,
        error: "Something went wrong. Please try again.",
        code: "INTERNAL_ERROR",
      };
    }
  };
}

// ─── Convenience re-export ────────────────────────────────────────────────────
export { createActionError, ActionError };
export type { ActionResult, ActionContext, SafeActionOptions };
