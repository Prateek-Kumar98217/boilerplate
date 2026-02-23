/**
 * ActionError — the only error class safe-action handlers should throw.
 *
 * Throwing anything else will result in an INTERNAL_ERROR response so that
 * unhandled/unexpected errors never leak implementation details to the client.
 */
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

/**
 * Narrow-tested helpers so handlers can throw with one line.
 * e.g.  throw ActionError.notFound("Post not found")
 */
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

/** Returns true if the Postgres error code maps to a unique-constraint violation. */
export function isUniqueConstraintError(err: unknown): boolean {
  return (
    typeof err === "object" &&
    err !== null &&
    "code" in err &&
    (err as { code: unknown }).code === "23505"
  );
}

/** Returns true if the Postgres error code maps to a foreign-key violation. */
export function isForeignKeyError(err: unknown): boolean {
  return (
    typeof err === "object" &&
    err !== null &&
    "code" in err &&
    (err as { code: unknown }).code === "23503"
  );
}
