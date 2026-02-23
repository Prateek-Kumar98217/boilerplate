# 03 — CRUD

Generic, typed CRUD operations for Drizzle ORM + Next.js Server Actions.

Covers four entities (User, Post, Comment, Tag) with full schemas, relations,
paginated listing, auth ownership enforcement, and a reusable base that
eliminates repetitive try/catch boilerplate.

---

## Files

| File                  | Purpose                                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| `db/schema.ts`        | Drizzle schemas: `users`, `posts`, `comments`, `tags`, `post_tags`                                        |
| `types/index.ts`      | `CrudResult<T>`, `PaginatedResult<T>`, `PaginationInput`                                                  |
| `actions/base.ts`     | `CrudError`, `requireAuth`, `runAction`, `normalizeCrudError`, `PaginationSchema`, `buildPaginatedResult` |
| `actions/posts.ts`    | `listPosts`, `getPostById`, `getPostBySlug`, `createPost`, `updatePost`, `deletePost`                     |
| `actions/comments.ts` | `listComments`, `createComment`, `updateComment`, `deleteComment`                                         |

---

## Flow

```
Client / Server Component
    │
    ▼  calls action with raw input
action (e.g. createPost)
    │
    ├─ 1. validateInput(schema, input)     → CrudResult error if invalid
    ├─ 2. runAction(async () => {
    │       ├─ 3. requireAuth()            → CrudError(UNAUTHORIZED) if no session
    │       ├─ 4. DB query via Drizzle
    │       └─ 5. throw CrudError(...)    → expected failures (NOT_FOUND, etc.)
    │    })
    └─ normalizeCrudError(err)             → converts Postgres / unknown errors
    │
    ▼
CrudResult<TData>
  { success: true,  data: TData }
  { success: false, error: string, code: CrudErrorCode }
```

---

## Usage

```ts
// Server Action call (from Server Component or Client Component)
const result = await createPost({
  title: "Hello",
  content: "World content here",
});

if (!result.success) {
  console.error(result.error, result.code);
  return;
}

console.log(result.data.id); // typed as Post
```

```ts
// Paginated list
const result = await listPosts({ page: 1, pageSize: 10, search: "typescript" });
if (result.success) {
  const { items, total, hasNextPage } = result.data;
}
```

---

## Schema Design

```
users ──< posts ──< comments
               │
              >< (M:M via post_tags) >< tags
```

- All PKs are UUIDs (`defaultRandom()`).
- `ON DELETE CASCADE` on foreign keys — deleting a post also deletes its comments and tag associations.
- `updatedAt` uses `.$onUpdate(() => new Date())` for automatic timestamp management.
- Indexes on all foreign keys for query performance.

---

## What Can Go Wrong

| Scenario                                     | How it's handled                                                       |
| -------------------------------------------- | ---------------------------------------------------------------------- |
| Input fails Zod validation                   | `validateInput` returns early with `fieldErrors`                       |
| Unauthenticated request                      | `requireAuth()` throws `CrudError(UNAUTHORIZED)`                       |
| Resource not found                           | Handler throws `CrudError(NOT_FOUND)`                                  |
| User tries to modify another user's resource | Ownership check with `AND authorId = userId` query                     |
| Deleting a comment with replies              | Soft-deletes the content to `"[deleted]"` to preserve thread structure |
| Unique constraint (duplicate slug)           | Postgres `23505` → `CONFLICT`                                          |
| Foreign key violation                        | Postgres `23503` → `NOT_FOUND`                                         |
| Empty update payload                         | Checked explicitly → `VALIDATION_ERROR`                                |
| Concurrent deletes                           | Last `DELETE` is a no-op since the row is already gone; no crash       |

---

## Adding a New Entity

1. Add the table to `db/schema.ts`.
2. Create `actions/your-entity.ts`.
3. Use `validateInput`, `requireAuth`, `runAction`, `CrudError` from `base.ts`.
4. Export actions as `"use server"` functions.

---

## Installation

```bash
npm install drizzle-orm pg zod next-auth@beta
npm install -D drizzle-kit @types/pg
```

Run migrations:

```bash
npx drizzle-kit generate
npx drizzle-kit migrate
```
