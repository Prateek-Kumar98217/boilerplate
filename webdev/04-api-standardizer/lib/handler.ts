/**
 * withApiHandler — wraps a Next.js App Router route handler.
 *
 * Guarantees:
 *  1. Every response is an ApiResponse (success or error shape).
 *  2. Unhandled throws are caught and converted to 500 INTERNAL_ERROR.
 *  3. ApiError throws are automatically converted to the correct status + shape.
 *  4. Zod parse errors surfaced as 422 VALIDATION_ERROR.
 *  5. Request body is parsed and typed once — no need to call req.json() in handlers.
 *
 * Usage:
 *
 *   // app/api/posts/route.ts
 *   export const GET = withApiHandler(async (req, ctx) => {
 *     const posts = await listPosts();
 *     return ok(posts);
 *   });
 *
 *   export const POST = withApiHandler(async (req, ctx) => {
 *     const body = await parseBody(req, CreatePostSchema);
 *     const post = await createPost(body);
 *     return created(post);
 *   }, { requireAuth: true });
 */
import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/auth"; // swap for your own session helper
import { ApiError, ApiErrors } from "./api-error";
import { apiError } from "./response";
import type { ApiErrorCode } from "./types";

// ─── Context ──────────────────────────────────────────────────────────────────

export type RouteContext = {
  params: Record<string, string | string[]>;
};

export type AuthedRouteContext = RouteContext & {
  userId: string;
  userEmail: string | null;
};

// ─── Handler signatures ───────────────────────────────────────────────────────

type RouteHandler = (
  req: NextRequest,
  ctx: RouteContext,
) => Promise<Response | NextResponse>;

type AuthedRouteHandler = (
  req: NextRequest,
  ctx: AuthedRouteContext,
) => Promise<Response | NextResponse>;

type WithApiHandlerOptions = {
  /**
   * When true, the handler receives an AuthedRouteContext with userId/userEmail.
   * Returns 401 if no valid session is found.
   */
  requireAuth?: boolean;
};

// ─── Overloads ────────────────────────────────────────────────────────────────

export function withApiHandler(
  handler: AuthedRouteHandler,
  options: WithApiHandlerOptions & { requireAuth: true },
): (req: NextRequest, ctx: RouteContext) => Promise<Response>;

export function withApiHandler(
  handler: RouteHandler,
  options?: WithApiHandlerOptions & { requireAuth?: false | undefined },
): (req: NextRequest, ctx: RouteContext) => Promise<Response>;

// ─── Implementation ───────────────────────────────────────────────────────────

export function withApiHandler(
  handler: RouteHandler | AuthedRouteHandler,
  options: WithApiHandlerOptions = {},
): (req: NextRequest, ctx: RouteContext) => Promise<Response> {
  return async (req: NextRequest, ctx: RouteContext): Promise<Response> => {
    try {
      if (options.requireAuth) {
        const session = await auth();

        if (!session?.user?.id) {
          return apiError("UNAUTHORIZED", "Authentication required.");
        }

        const authedCtx: AuthedRouteContext = {
          ...ctx,
          userId: session.user.id,
          userEmail: session.user.email ?? null,
        };

        return await (handler as AuthedRouteHandler)(req, authedCtx);
      }

      return await (handler as RouteHandler)(req, ctx);
    } catch (err) {
      // ApiError — intentional, structured errors.
      if (err instanceof ApiError) {
        return apiError(err.code, err.message, {
          fieldErrors: err.fieldErrors,
        });
      }

      // ZodError — e.g. thrown by parseBody if called manually without the helper.
      if (err instanceof z.ZodError) {
        return apiError("VALIDATION_ERROR", "Validation failed.", {
          fieldErrors: err.flatten().fieldErrors as Partial<
            Record<string, string[]>
          >,
        });
      }

      // Log and return a safe 500.
      console.error("[withApiHandler] Unhandled error:", err);
      return apiError("INTERNAL_ERROR", "An unexpected error occurred.");
    }
  };
}

// ─── Body parsing helper ──────────────────────────────────────────────────────

/**
 * Parses and validates the JSON request body against a Zod schema.
 * Throws ApiError(VALIDATION_ERROR) if parsing fails — caught by withApiHandler.
 */
export async function parseBody<TSchema extends z.ZodTypeAny>(
  req: NextRequest,
  schema: TSchema,
): Promise<z.infer<TSchema>> {
  let rawBody: unknown;

  try {
    rawBody = await req.json();
  } catch {
    throw ApiErrors.badRequest("Request body must be valid JSON.");
  }

  const parsed = schema.safeParse(rawBody);

  if (!parsed.success) {
    throw new ApiError(
      "Validation failed.",
      "VALIDATION_ERROR",
      parsed.error.flatten().fieldErrors as Partial<Record<string, string[]>>,
    );
  }

  return parsed.data;
}

/**
 * Parses and validates URL search params against a Zod schema.
 * Throws ApiError(VALIDATION_ERROR) if parsing fails.
 */
export function parseSearchParams<TSchema extends z.ZodTypeAny>(
  req: NextRequest,
  schema: TSchema,
): z.infer<TSchema> {
  const rawParams: Record<string, string> = {};
  req.nextUrl.searchParams.forEach((value, key) => {
    rawParams[key] = value;
  });

  const parsed = schema.safeParse(rawParams);

  if (!parsed.success) {
    throw new ApiError(
      "Invalid query parameters.",
      "VALIDATION_ERROR",
      parsed.error.flatten().fieldErrors as Partial<Record<string, string[]>>,
    );
  }

  return parsed.data;
}

export type { ApiErrorCode };
