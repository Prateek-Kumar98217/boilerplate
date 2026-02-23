import type { ActionErrorCode } from "./types";

export class ActionError extends Error {
  public readonly code: ActionErrorCode;
  public readonly fieldErrors?: Partial<Record<string, string[]>>;

  constructor(
    message: string,
    code: ActionErrorCode = "INTERNAL_ERROR",
    fieldErrors?: Partial<Record<string, string[]>>,
  ) {
    super(message);
    this.name = "ActionError";
    this.code = code;
    this.fieldErrors = fieldErrors;
  }
}

export const createActionError = {
  unauthorized: (msg = "You must be signed in.") =>
    new ActionError(msg, "UNAUTHORIZED"),

  forbidden: (msg = "You do not have permission to do this.") =>
    new ActionError(msg, "FORBIDDEN"),

  notFound: (msg = "Resource not found.") => new ActionError(msg, "NOT_FOUND"),

  conflict: (msg = "This resource already exists.") =>
    new ActionError(msg, "CONFLICT"),

  validation: (
    msg = "Invalid input.",
    fieldErrors?: Partial<Record<string, string[]>>,
  ) => new ActionError(msg, "VALIDATION_ERROR", fieldErrors),

  rateLimited: (msg = "Too many requests. Please try again later.") =>
    new ActionError(msg, "RATE_LIMITED"),

  internal: (msg = "Something went wrong. Please try again.") =>
    new ActionError(msg, "INTERNAL_ERROR"),
};

/**
 * Maps Supabase PostgREST error codes to ActionErrorCode.
 * https://postgrest.org/en/stable/references/errors.html
 */
export function classifySupabaseError(error: {
  code: string;
  message: string;
}): ActionErrorCode {
  switch (error.code) {
    case "23505":
      return "CONFLICT";
    case "23503":
      return "NOT_FOUND";
    case "PGRST116": // 0 rows returned by .single()
      return "NOT_FOUND";
    case "42501": // RLS policy violation
      return "FORBIDDEN";
    default:
      return "INTERNAL_ERROR";
  }
}
