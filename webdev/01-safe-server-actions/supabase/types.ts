/**
 * Core types for the Safe Server Action pattern (Supabase variant).
 * Identical shape to the Drizzle variant so both can be used interchangeably.
 */

// ─── Result shape ────────────────────────────────────────────────────────────

export type ActionResult<TData> =
  | { success: true; data: TData; message?: string }
  | {
      success: false;
      error: string;
      fieldErrors?: Partial<Record<string, string[]>>;
      code: ActionErrorCode;
    };

// ─── Error codes ─────────────────────────────────────────────────────────────

export type ActionErrorCode =
  | "VALIDATION_ERROR"
  | "UNAUTHORIZED"
  | "FORBIDDEN"
  | "NOT_FOUND"
  | "CONFLICT"
  | "RATE_LIMITED"
  | "INTERNAL_ERROR";

// ─── Handler context ─────────────────────────────────────────────────────────

export type ActionContext = {
  userId: string;
  userEmail: string | null;
};

export type ActionHandler<TInput, TData> = (
  validatedData: TInput,
  context: ActionContext,
) => Promise<TData>;

export type SafeActionOptions = {
  requireAuth?: boolean;
};
