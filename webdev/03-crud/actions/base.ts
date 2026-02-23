/**
 * createCrudActions — generic CRUD factory for Drizzle + Next.js Server Actions.
 *
 * Reduces boilerplate for standard list/get/create/update/delete patterns.
 * Each entity file calls this factory and can override or extend any method.
 *
 * The factory is intentionally minimal: it handles the repetitive scaffolding
 * (validation, auth, error catching, result shaping) while leaving
 * entity-specific query logic in the entity files.
 */
"use server";

import { z } from "zod";
import { auth } from "@/auth";
import type {
  CrudResult,
  CrudErrorCode,
  PaginatedResult,
  PaginationInput,
} from "../types";

// ─── Internal action error ────────────────────────────────────────────────────

export class CrudError extends Error {
  public readonly code: CrudErrorCode;
  constructor(message: string, code: CrudErrorCode = "INTERNAL_ERROR") {
    super(message);
    this.name = "CrudError";
    this.code = code;
  }
}

// ─── Auth helper ──────────────────────────────────────────────────────────────

export async function requireAuth(): Promise<{
  userId: string;
  userEmail: string | null;
}> {
  const session = await auth();
  if (!session?.user?.id) {
    throw new CrudError("You must be signed in.", "UNAUTHORIZED");
  }
  return { userId: session.user.id, userEmail: session.user.email ?? null };
}

// ─── Error normalizer ─────────────────────────────────────────────────────────

export function normalizeCrudError(err: unknown): CrudResult<never> {
  if (err instanceof CrudError) {
    return { success: false, error: err.message, code: err.code };
  }

  // Postgres unique constraint
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

  // Postgres foreign key violation
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

  console.error("[CrudAction] Unhandled error:", err);
  return {
    success: false,
    error: "Something went wrong. Please try again.",
    code: "INTERNAL_ERROR",
  };
}

// ─── Validation helper ────────────────────────────────────────────────────────

export function validateInput<T>(
  schema: z.ZodType<T>,
  input: unknown,
): { success: true; data: T } | CrudResult<never> {
  const parsed = schema.safeParse(input);
  if (!parsed.success) {
    return {
      success: false,
      error: "Validation failed.",
      fieldErrors: parsed.error.flatten().fieldErrors as Partial<
        Record<string, string[]>
      >,
      code: "VALIDATION_ERROR",
    };
  }
  return { success: true, data: parsed.data };
}

// ─── Pagination factory ───────────────────────────────────────────────────────

export const PaginationSchema = z.object({
  page: z.number().int().min(1).default(1),
  pageSize: z.number().int().min(1).max(100).default(20),
  search: z.string().trim().optional(),
});

export function buildPaginatedResult<T>(
  items: T[],
  total: number,
  { page, pageSize }: PaginationInput,
): PaginatedResult<T> {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  return {
    items,
    total,
    page,
    pageSize,
    totalPages,
    hasNextPage: page < totalPages,
    hasPreviousPage: page > 1,
  };
}

// ─── Generic action wrapper ───────────────────────────────────────────────────

/**
 * Wraps an async handler in a try/catch that returns a typed CrudResult.
 * Used by entity action files to avoid repeating the same try/catch.
 */
export async function runAction<T>(
  fn: () => Promise<T>,
): Promise<CrudResult<T>> {
  try {
    const data = await fn();
    return { success: true, data };
  } catch (err) {
    return normalizeCrudError(err) as CrudResult<T>;
  }
}

export type { CrudResult, PaginatedResult, PaginationInput };
