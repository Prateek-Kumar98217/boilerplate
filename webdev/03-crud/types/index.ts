/**
 * Shared types for the CRUD pattern.
 */
import type { z } from "zod";

// ─── Pagination ───────────────────────────────────────────────────────────────

export type PaginationInput = {
  page: number;
  pageSize: number;
};

export type PaginatedResult<T> = {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
};

// ─── CRUD action result (same discriminated union as safe-server-actions) ─────

export type CrudResult<T> =
  | { success: true; data: T; message?: string }
  | {
      success: false;
      error: string;
      fieldErrors?: Partial<Record<string, string[]>>;
      code: CrudErrorCode;
    };

export type CrudErrorCode =
  | "VALIDATION_ERROR"
  | "UNAUTHORIZED"
  | "FORBIDDEN"
  | "NOT_FOUND"
  | "CONFLICT"
  | "INTERNAL_ERROR";

// ─── Generic CRUD actions interface ──────────────────────────────────────────

export type CrudActions<
  TSelect,
  TCreateSchema extends z.ZodTypeAny,
  TUpdateSchema extends z.ZodTypeAny,
> = {
  list: (
    input: PaginationInput & { search?: string },
  ) => Promise<CrudResult<PaginatedResult<TSelect>>>;
  getById: (id: string) => Promise<CrudResult<TSelect>>;
  create: (input: unknown) => Promise<CrudResult<TSelect>>;
  update: (input: unknown) => Promise<CrudResult<TSelect>>;
  delete: (id: string) => Promise<CrudResult<{ id: string }>>;
  createSchema: TCreateSchema;
  updateSchema: TUpdateSchema;
};
