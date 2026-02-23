/**
 * Response builder helpers.
 *
 * These build the standardized ApiResponse JSON and wrap it in a Next.js Response.
 */
import { NextResponse } from "next/server";
import { nanoid } from "nanoid"; // or use crypto.randomUUID()
import type {
  ApiSuccessResponse,
  ApiErrorResponse,
  ApiErrorCode,
  ApiResponse,
  PaginationMeta,
  ResponseMeta,
} from "./types";
import { ERROR_STATUS_MAP } from "./types";

// ─── Meta factory ─────────────────────────────────────────────────────────────

function buildMeta(pagination?: PaginationMeta): ResponseMeta {
  return {
    requestId: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    ...(pagination ? { pagination } : {}),
  };
}

// ─── Success helpers ──────────────────────────────────────────────────────────

/**
 * Returns a 200 JSON response with the standardized success shape.
 */
export function ok<T>(
  data: T,
  options: {
    message?: string;
    status?: number;
    pagination?: PaginationMeta;
    headers?: HeadersInit;
  } = {},
): NextResponse<ApiSuccessResponse<T>> {
  const body: ApiSuccessResponse<T> = {
    success: true,
    data,
    ...(options.message ? { message: options.message } : {}),
    meta: buildMeta(options.pagination),
  };

  return NextResponse.json(body, {
    status: options.status ?? 200,
    headers: options.headers,
  });
}

/**
 * Returns a 201 Created response. Use after successful resource creation.
 */
export function created<T>(
  data: T,
  options: { message?: string; headers?: HeadersInit } = {},
): NextResponse<ApiSuccessResponse<T>> {
  return ok(data, { ...options, status: 201 });
}

/**
 * Returns a 204 No Content response. Body must be empty — do not pass data.
 */
export function noContent(): Response {
  return new Response(null, { status: 204 });
}

// ─── Error helpers ────────────────────────────────────────────────────────────

/**
 * Returns a JSON error response with the standardized error shape.
 */
export function apiError(
  code: ApiErrorCode,
  message: string,
  options: {
    fieldErrors?: Partial<Record<string, string[]>>;
    headers?: HeadersInit;
  } = {},
): NextResponse<ApiErrorResponse> {
  const body: ApiErrorResponse = {
    success: false,
    error: {
      code,
      message,
      ...(options.fieldErrors ? { fieldErrors: options.fieldErrors } : {}),
    },
    meta: buildMeta(),
  };

  return NextResponse.json(body, {
    status: ERROR_STATUS_MAP[code],
    headers: options.headers,
  });
}

// ─── Type guards ──────────────────────────────────────────────────────────────

export function isApiSuccess<T>(
  response: ApiResponse<T>,
): response is ApiSuccessResponse<T> {
  return response.success === true;
}

export function isApiError(
  response: ApiResponse<unknown>,
): response is ApiErrorResponse {
  return response.success === false;
}
