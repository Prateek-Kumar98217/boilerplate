/**
 * Core types for the Safe Server Action pattern (Drizzle + PostgreSQL variant).
 *
 * ActionResult is a discriminated union so callers can narrow with `if (result.success)`.
 * Never expose raw DB errors to the client — use ActionErrorCode for machine-readable
 * classification and a safe human-readable `error` string.
 */

// ─── Result shape ────────────────────────────────────────────────────────────

export type ActionResult<TData> =
  | { success: true; data: TData; message?: string }
  | {
      success: false;
      error: string;
      /** Per-field validation messages, keyed by field name. */
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

/**
 * Populated after auth is verified and injected into every handler so handlers
 * never have to re-fetch session data themselves.
 */
export type ActionContext = {
  userId: string;
  userEmail: string | null;
};

// ─── Handler signature ───────────────────────────────────────────────────────

/**
 * The actual business-logic function you pass to `createSafeAction`.
 * Receives validated, typed input and the authenticated context.
 * Should throw `ActionError` for expected failure paths.
 */
export type ActionHandler<TInput, TData> = (
  validatedData: TInput,
  context: ActionContext,
) => Promise<TData>;

// ─── Options ─────────────────────────────────────────────────────────────────

export type SafeActionOptions = {
  /**
   * Default: true. Set to false for public actions that don't require a session.
   * Context.userId will be empty string "" when auth is skipped.
   */
  requireAuth?: boolean;
};
