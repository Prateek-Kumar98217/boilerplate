/**
 * ApiError — the only error class that route handlers should throw.
 * The handler wrapper catches this and converts it to a structured ApiErrorResponse.
 */
import type { ApiErrorCode } from "./types";

export class ApiError extends Error {
  public readonly code: ApiErrorCode;
  public readonly statusCode: number;
  public readonly fieldErrors?: Partial<Record<string, string[]>>;

  constructor(
    message: string,
    code: ApiErrorCode = "INTERNAL_ERROR",
    fieldErrors?: Partial<Record<string, string[]>>,
  ) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.fieldErrors = fieldErrors;

    // Import lazily to avoid circular dependency.
    const { ERROR_STATUS_MAP } = require("./types") as typeof import("./types");
    this.statusCode = ERROR_STATUS_MAP[code];
  }
}

/**
 * Pre-built factory methods for the most common error scenarios.
 */
export const ApiErrors = {
  badRequest: (msg = "Bad request.") => new ApiError(msg, "BAD_REQUEST"),

  validation: (
    msg = "Validation failed.",
    fieldErrors?: Partial<Record<string, string[]>>,
  ) => new ApiError(msg, "VALIDATION_ERROR", fieldErrors),

  unauthorized: (msg = "Authentication required.") =>
    new ApiError(msg, "UNAUTHORIZED"),

  forbidden: (msg = "You do not have permission to access this resource.") =>
    new ApiError(msg, "FORBIDDEN"),

  notFound: (resource = "Resource") =>
    new ApiError(`${resource} not found.`, "NOT_FOUND"),

  methodNotAllowed: (allowed: string[]) =>
    new ApiError(
      `Method not allowed. Allowed: ${allowed.join(", ")}.`,
      "METHOD_NOT_ALLOWED",
    ),

  conflict: (msg = "A conflict occurred.") => new ApiError(msg, "CONFLICT"),

  rateLimited: (msg = "Too many requests. Please try again later.") =>
    new ApiError(msg, "RATE_LIMITED"),

  internal: (msg = "An unexpected error occurred.") =>
    new ApiError(msg, "INTERNAL_ERROR"),
};
