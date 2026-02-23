# 05 — Auth Scaffold

Complete authentication setup for Next.js App Router.

Two implementations are provided — choose one:

| Directory    | Stack                                    |
| ------------ | ---------------------------------------- |
| `next-auth/` | Next Auth v5 (Auth.js) + Drizzle adapter |
| `clerk/`     | Clerk (`@clerk/nextjs`)                  |

---

## Next Auth v5

### Files

| File                                 | Purpose                                                  |
| ------------------------------------ | -------------------------------------------------------- |
| `auth.config.ts`                     | Providers, callbacks, pages config — **Edge-safe**       |
| `auth.ts`                            | Main `NextAuth()` instance with Drizzle adapter          |
| `middleware.ts`                      | Edge middleware — imports from `auth.config.ts` only     |
| `components/AuthSessionProvider.tsx` | Root `<SessionProvider>` wrapper for Client Components   |
| `components/ProtectedRoute.tsx`      | Server Component route guard + `requireSession()` helper |
| `pages/sign-in.tsx`                  | Server Component sign-in page with OAuth + Credentials   |
| `hooks/useAuth.ts`                   | `useAuth()` — auth state + signIn/signOut helpers        |
| `hooks/useUser.ts`                   | `useUser()` — typed session user                         |

### Setup

**1. Place files:**

```
app/
  api/auth/[...nextauth]/route.ts   ← export { handlers as GET, handlers as POST } from "@/auth"
auth.ts                             ← project root
auth.config.ts                      ← project root
middleware.ts                       ← project root
```

**2. Root layout:**

```tsx
// app/layout.tsx
import { AuthSessionProvider } from "@/components/AuthSessionProvider";
import { auth } from "@/auth";

export default async function RootLayout({ children }) {
  const session = await auth();
  return (
    <html>
      <body>
        <AuthSessionProvider session={session}>{children}</AuthSessionProvider>
      </body>
    </html>
  );
}
```

**3. Protect a page (Server Component):**

```tsx
import { ProtectedRoute } from "@/components/ProtectedRoute";

export default function DashboardLayout({ children }) {
  return <ProtectedRoute>{children}</ProtectedRoute>;
}
```

**4. Use auth state (Client Component):**

```tsx
"use client";
import { useAuth } from "@/hooks/useAuth";
import { useUser } from "@/hooks/useUser";

const { isAuthenticated, signOut } = useAuth();
const { user } = useUser();
```

### Flow

```
Request → middleware (Edge)
    │
    ├─ authorized() callback checks session from JWT
    ├─ public route? → pass through
    ├─ auth page + signed in? → redirect to /dashboard
    └─ protected route + no session? → redirect to /auth/sign-in

Server Component
    ├─ auth()           → full session (Node.js runtime)
    └─ ProtectedRoute → redirect if no session

Client Component
    ├─ useAuth()  → { isAuthenticated, status, signIn, signOut }
    └─ useUser()  → { user: { id, name, email, image } | null }
```

---

## Clerk

### Files

| File                               | Purpose                                                     |
| ---------------------------------- | ----------------------------------------------------------- |
| `middleware.ts`                    | `clerkMiddleware` with public/admin route matchers          |
| `components/ClerkAuthProvider.tsx` | Root `<ClerkProvider>` wrapper                              |
| `components/ProtectedRoute.tsx`    | Server Component guard + `requireClerkSession()`            |
| `server-helpers.ts`                | `requireClerkSession`, `getCurrentClerkUser`, `getUserRole` |
| `hooks/useAuth.ts`                 | `useAuth()` — status, userId, getToken                      |
| `hooks/useUser.ts`                 | `useUser()` — normalized `AppUser` shape                    |

### Setup

**1. Root layout:**

```tsx
// app/layout.tsx
import { ClerkAuthProvider } from "@/components/ClerkAuthProvider";

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        <ClerkAuthProvider>{children}</ClerkAuthProvider>
      </body>
    </html>
  );
}
```

**2. Protect a page (Server Component):**

```tsx
import { ProtectedRoute } from "@/components/ProtectedRoute";

export default function DashboardLayout({ children }) {
  return <ProtectedRoute>{children}</ProtectedRoute>;
}
```

**3. Server Action:**

```ts
"use server";
import { requireClerkSession } from "@/clerk/server-helpers";

export async function createPost(input: unknown) {
  const { userId } = await requireClerkSession();
  // ...
}
```

**4. Client Component:**

```tsx
"use client";
import { useAuth } from "@/hooks/useAuth";
import { useUser } from "@/hooks/useUser";

const { isAuthenticated, userId, getToken } = useAuth();
const { user } = useUser(); // user.email, user.imageUrl, etc.
```

---

## Comparison

| Feature          | Next Auth v5                | Clerk                            |
| ---------------- | --------------------------- | -------------------------------- |
| Self-hosted      | Yes                         | No (SaaS)                        |
| Own DB / adapter | Yes (Drizzle, Prisma, etc.) | Optional (sync via webhooks)     |
| Built-in UI      | No (bring your own)         | Yes (`<SignIn>`, `<UserButton>`) |
| MFA / Passkeys   | Manual                      | Built-in                         |
| Organizations    | Manual                      | Built-in                         |
| Webhooks         | Manual                      | Built-in                         |
| Edge-compatible  | Yes (via auth.config.ts)    | Yes                              |
| Free tier        | Open source                 | Generous free tier               |

---

## What Can Go Wrong

| Scenario                                       | How it's handled                                                                         |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `auth.ts` imported in middleware               | Edge runtime error — always import `auth.config.ts` in middleware                        |
| `getSession()` used server-side (Supabase)     | **Never safe** — use `getUser()` which re-validates the JWT with Supabase's servers      |
| Credentials provider with plain-text passwords | Mentioned in code comments — always use bcrypt/argon2                                    |
| JWT token missing `id` (Next Auth)             | Type augmentation enforces `session.user.id` as `string`; augmented in `auth.ts`         |
| Static files hit middleware                    | Regex matchers exclude `_next/static`, `_next/image`, and file extensions                |
| Role not in session claims (Clerk)             | `getUserRole()` returns `null`; guards treat `null` as unauthorized                      |
| OAuth account linked to different provider     | `authorized` callback handles redirect; sign-in page shows `OAuthAccountNotLinked` error |

---

## Installation

```bash
# Next Auth v5
npm install next-auth@beta @auth/drizzle-adapter

# Clerk
npm install @clerk/nextjs
```

Required env vars:

```env
# Next Auth
AUTH_SECRET=your-32-char-secret
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...

# Clerk
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_...
CLERK_SECRET_KEY=sk_...
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/auth/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/auth/sign-up
NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/dashboard
NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/dashboard
```
