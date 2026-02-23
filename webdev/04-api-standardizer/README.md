# 04 — API Response Standardizer

A set of helpers that enforce a consistent response shape across all Next.js App Router API routes.

Every response — success or error — always has the same `{ success, data | error, meta }` structure so clients never have to guess what shape they'll receive.

---

## Files

| File                    | Purpose                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| `lib/types.ts`          | `ApiResponse<T>`, `ApiSuccessResponse<T>`, `ApiErrorResponse`, `ApiErrorCode`, status map |
| `lib/api-error.ts`      | `ApiError` class + `ApiErrors` factory                                                    |
| `lib/response.ts`       | `ok()`, `created()`, `noContent()`, `apiError()`, type guards                             |
| `lib/handler.ts`        | `withApiHandler()` wrapper, `parseBody()`, `parseSearchParams()`                          |
| `lib/index.ts`          | Barrel export                                                                             |
| `example/route.ts`      | `GET /api/posts` + `POST /api/posts`                                                      |
| `example/[id]/route.ts` | `GET`, `PATCH`, `DELETE /api/posts/[id]`                                                  |

---

## Response Shape

```ts
// Success
{
  "success": true,
  "data": { ... },
  "message": "Created successfully.",   // optional
  "meta": {
    "requestId": "3f2a1b...",
    "timestamp": "2026-02-22T10:00:00Z",
    "pagination": { ... }               // list endpoints only
  }
}

// Error
{
  "success": false,
  "error": {
    "code": "NOT_FOUND",
    "message": "Post not found.",
    "fieldErrors": { "email": ["Invalid email."] }  // 422 only
  },
  "meta": { "requestId": "...", "timestamp": "..." }
}
```

---

## Flow

```
HTTP Request
    │
    ▼
withApiHandler(handler, { requireAuth?: true })
    │
    ├─ requireAuth → auth()  →  401 if no session
    ├─ call handler(req, ctx)
    │     │
    │     ├─ parseBody(req, Schema)         throws ApiError on invalid JSON / Zod fail
    │     ├─ parseSearchParams(req, Schema) throws ApiError on invalid params
    │     └─ throw ApiErrors.notFound(...)  for intentional errors
    │
    ├─ catch ApiError     → apiError(err.code, err.message)
    ├─ catch ZodError     → apiError("VALIDATION_ERROR", ...)
    └─ catch unknown      → apiError("INTERNAL_ERROR", ...) + console.error
    │
    ▼
NextResponse<ApiResponse<T>>
```

---

## Usage

```ts
// app/api/posts/route.ts
import { withApiHandler, parseBody, ok, created, ApiErrors } from "@/lib/api";
import { z } from "zod";

const BodySchema = z.object({ title: z.string().min(3) });

export const GET = withApiHandler(async (req) => {
  const posts = await db.query.posts.findMany();
  return ok(posts);
});

export const POST = withApiHandler(
  async (req, ctx) => {
    const body = await parseBody(req, BodySchema);
    const post = await db
      .insert(posts)
      .values({ ...body, authorId: ctx.userId })
      .returning();
    if (!post[0]) throw ApiErrors.internal();
    return created(post[0], { message: "Post created." });
  },
  { requireAuth: true },
);
```

---

## HTTP Status Code Mapping

| `ApiErrorCode`         | HTTP Status |
| ---------------------- | ----------- |
| `BAD_REQUEST`          | 400         |
| `UNAUTHORIZED`         | 401         |
| `FORBIDDEN`            | 403         |
| `NOT_FOUND`            | 404         |
| `METHOD_NOT_ALLOWED`   | 405         |
| `CONFLICT`             | 409         |
| `VALIDATION_ERROR`     | 422         |
| `UNPROCESSABLE_ENTITY` | 422         |
| `RATE_LIMITED`         | 429         |
| `INTERNAL_ERROR`       | 500         |

---

## What Can Go Wrong

| Scenario                            | How it's handled                                                            |
| ----------------------------------- | --------------------------------------------------------------------------- |
| Handler throws `ApiError`           | Converted to correct status + shape by `withApiHandler`                     |
| Handler throws `ZodError` directly  | Caught and converted to 422 VALIDATION_ERROR                                |
| `req.json()` fails (malformed body) | `parseBody` throws `ApiErrors.badRequest`                                   |
| Search params fail validation       | `parseSearchParams` throws `ApiErrors.validation`                           |
| Unhandled throw                     | Logged server-side, returns 500 INTERNAL_ERROR — implementation never leaks |
| Missing auth                        | `withApiHandler({ requireAuth: true })` returns 401 before handler runs     |
| `noContent()` used with data        | TypeScript error — `noContent()` takes no arguments                         |

---

## Installation

```bash
npm install next zod next-auth@beta
```
