# 01 — Safe Server Actions

A typed, composable pattern for Next.js Server Actions that enforces a strict pipeline:
**validate input → check auth → perform DB op → handle errors → return result**

Two implementations are provided:

| Directory   | Stack                                   |
| ----------- | --------------------------------------- |
| `drizzle/`  | PostgreSQL + Drizzle ORM + Next Auth v5 |
| `supabase/` | Supabase (PostgREST + Auth)             |

---

## Files

### `drizzle/`

| File                           | Purpose                                             |
| ------------------------------ | --------------------------------------------------- |
| `types.ts`                     | `ActionResult<T>`, `ActionContext`, error codes     |
| `action-error.ts`              | `ActionError` class + `createActionError` factory   |
| `create-safe-action.ts`        | **Core factory** — wraps a Zod schema + handler     |
| `db.ts`                        | Singleton Drizzle + node-postgres client            |
| `schema.ts`                    | Example `posts` table schema                        |
| `example-actions.ts`           | `createPost`, `updatePost`, `deletePost`, `getPost` |
| `example-client-component.tsx` | Calling an action from a Client Component           |

### `supabase/`

| File                    | Purpose                                                  |
| ----------------------- | -------------------------------------------------------- |
| `types.ts`              | Same shape as Drizzle variant                            |
| `action-error.ts`       | `ActionError` + `classifySupabaseError`                  |
| `supabase-server.ts`    | Per-request Supabase server client factory               |
| `database.types.ts`     | Generated-types stub (replace with `supabase gen types`) |
| `create-safe-action.ts` | **Core factory** — injects Supabase client into context  |
| `example-actions.ts`    | CRUD actions using Supabase client                       |

---

## Flow

```
Client Component
    │
    ▼  (calls action with FormData / plain object)
createSafeAction(schema, handler, options)
    │
    ├─ 1. schema.safeParse(input)          → VALIDATION_ERROR if invalid
    ├─ 2. auth() / supabase.auth.getUser() → UNAUTHORIZED if no session
    ├─ 3. handler(validatedData, context)  → throws ActionError on expected failures
    └─ 4. catch(err)                       → normalizes to ActionResult<never>
    │
    ▼
ActionResult<TData>
  { success: true,  data: TData }
  { success: false, error: string, fieldErrors?, code: ActionErrorCode }
```

---

## Usage

```ts
// actions/posts.ts
"use server";
import { createSafeAction, createActionError } from "./create-safe-action";
import { eq, and } from "drizzle-orm";
import { z } from "zod";
import { revalidatePath } from "next/cache";
import { db } from "./db";
import { posts } from "./schema";

const Schema = z.object({
  title: z.string().min(3),
  content: z.string().min(10),
});

export const createPost = createSafeAction(Schema, async (input, ctx) => {
  const [post] = await db
    .insert(posts)
    .values({ ...input, authorId: ctx.userId })
    .returning();
  if (!post) throw createActionError.internal();
  revalidatePath("/posts");
  return post;
});
```

```tsx
// CreatePostForm.tsx (Client Component)
"use client";
import { useActionState } from "react";
import { createPost } from "./actions/posts";

const [state, action, isPending] = useActionState(
  (_prev, formData) =>
    createPost({
      title: formData.get("title"),
      content: formData.get("content"),
    }),
  { success: false, error: "", code: "INTERNAL_ERROR" },
);
```

---

## What Can Go Wrong

| Scenario                                              | How it's handled                                                     |
| ----------------------------------------------------- | -------------------------------------------------------------------- |
| Invalid input (wrong type, missing field)             | `schema.safeParse` returns field-level errors                        |
| No session / expired JWT                              | `auth()` returns null → `UNAUTHORIZED` result                        |
| Unique constraint violation (email exists)            | Postgres error code `23505` detected → `CONFLICT`                    |
| Foreign key violation                                 | Postgres error code `23503` detected → `NOT_FOUND`                   |
| Supabase RLS policy blocks the operation              | PostgREST code `42501` → `FORBIDDEN`                                 |
| Handler throws arbitrary error                        | Caught, logged server-side, returns generic `INTERNAL_ERROR`         |
| `getSession()` used instead of `getUser()` (Supabase) | **Always use `getUser()`** — `getSession()` reads unverified cookies |

---

## Installation

```bash
# Drizzle variant
npm install drizzle-orm pg @auth/drizzle-adapter next-auth@beta zod

# Supabase variant
npm install @supabase/ssr @supabase/supabase-js zod
```

Required environment variables:

```env
# Drizzle
DATABASE_URL=postgresql://user:password@localhost:5432/db
AUTH_SECRET=your-secret
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...

# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=...
```
