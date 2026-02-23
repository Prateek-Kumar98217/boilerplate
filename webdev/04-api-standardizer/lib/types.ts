/**
 * Standardized API response types.
 *
 * Every API route handler should return ApiResponse<T>.
 * Clients can always rely on the { success, data | error } shape.
 */

// ─── Meta ─────────────────────────────────────────────────────────────────────

export type PaginationMeta = {
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
};

export type ResponseMeta = {
  requestId: string;
  timestamp: string;
  /** Present on list endpoints. */
  pagination?: PaginationMeta;
};

// ─── Response shapes ─────────────────────────────────────────────────────────

export type ApiSuccessResponse<T> = {
  success: true;
  data: T;
  message?: string;
  meta: ResponseMeta;
};

export type ApiErrorResponse = {
  success: false;
  error: {
    code: ApiErrorCode;
    message: string;
    /** Per-field validation errors, present on 422 responses. */
    fieldErrors?: Partial<Record<string, string[]>>;
  };
  meta: ResponseMeta;
};

export type ApiResponse<T> = ApiSuccessResponse<T> | ApiErrorResponse;

// ─── Error codes ──────────────────────────────────────────────────────────────

export type ApiErrorCode =
  | "BAD_REQUEST"
  | "VALIDATION_ERROR"
  | "UNAUTHORIZED"
  | "FORBIDDEN"
  | "NOT_FOUND"
  | "METHOD_NOT_ALLOWED"
  | "CONFLICT"
  | "UNPROCESSABLE_ENTITY"
  | "RATE_LIMITED"
  | "INTERNAL_ERROR";

/**
 * Maps ApiErrorCode to HTTP status codes.
 */
export const ERROR_STATUS_MAP: Record<ApiErrorCode, number> = {
  BAD_REQUEST: 400,
  VALIDATION_ERROR: 422,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  METHOD_NOT_ALLOWED: 405,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  RATE_LIMITED: 429,
  INTERNAL_ERROR: 500,
};
