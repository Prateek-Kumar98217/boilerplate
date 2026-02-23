# Web Dev Boilerplate

TypeScript boilerplate for recurring patterns in React / Next.js development.
Every pattern is production-ready: no `any`, full error handling, and typed edge cases covered.

## Stack

- **Framework:** Next.js (App Router)
- **Language:** TypeScript (strict, no `any`)
- **Database ORM:** Drizzle ORM
- **Database:** PostgreSQL
- **BaaS alternative:** Supabase
- **Auth:** Next Auth v5 / Clerk
- **Validation:** Zod
- **Forms:** React Hook Form
- **Styles:** Tailwind CSS
- **State:** Zustand
- **Payments:** Stripe / Razorpay
- **Vector DB:** ChromaDB / Pinecone

---

## Patterns

| #   | Directory                 | Pattern                         | Key files                                              |
| --- | ------------------------- | ------------------------------- | ------------------------------------------------------ |
| 1   | `01-safe-server-actions/` | Safe server actions             | `create-safe-action.ts`                                |
| 2   | `02-zod-hook-form/`       | Generic Zod + RHF form          | `GenericForm.tsx`, `FormField.tsx`                     |
| 3   | `03-crud/`                | Full CRUD with pagination       | `actions/base.ts`, `actions/posts.ts`                  |
| 4   | `04-api-standardizer/`    | Standardized API responses      | `lib/handler.ts`, `lib/response.ts`                    |
| 5   | `05-auth-scaffold/`       | Auth setup (Next Auth + Clerk)  | `middleware.ts`, `ProtectedRoute.tsx`, `useAuth.ts`    |
| 6   | `06-subscription-guard/`  | Subscription middleware + toast | `middleware.ts`, `UpgradeToast.tsx`, `session-sync.ts` |
| 7   | `07-payment-processor/`   | Stripe + Razorpay abstraction   | `stripe.ts`, `razorpay.ts`, `index.ts`                 |
| 8   | `08-modular-store/`       | Modular Zustand store + persist | `store/index.ts`, `hooks/useTaskStore.ts`              |
| 9   | `09-rag/`                 | RAG: embed + store + retrieve   | `upsert-document.ts`, `providers/chroma.ts`            |

Each directory contains:

- **Implementation files** — copy into your project
- **`README.md`** — flow diagrams, usage examples, and a "What Can Go Wrong" table
- **Example files** — end-to-end usage demonstrations

---

## Quick Start

Each pattern is self-contained. Copy the directory you need into your Next.js project,
install the listed dependencies from that pattern's README, and adapt the imports.

Common path alias used throughout: `@/` → your `src/` or project root.
