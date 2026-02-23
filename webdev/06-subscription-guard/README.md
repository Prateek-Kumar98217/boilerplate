# 06 · Subscription Guard

Middleware-first, Edge-safe subscription enforcement with toast redirects.
Compatible with **Next Auth v5** and **Clerk**, and normalises status strings
from both **Stripe** and **Razorpay**.

---

## File map

```
06-subscription-guard/
├── types.ts                  Plan/status types, plan hierarchy, access helpers
├── config.ts                 Route rules: which plans unlock which paths
├── middleware.ts             Edge-safe JWT → subscription check → redirect
├── session-sync.ts           Inject subscription into Next Auth / Clerk session
├── components/
│   ├── UpgradeToast.tsx      Client Component – fires sonner toast on mount
│   └── SubscriptionGate.tsx  Server Component – hard or soft gate
└── hooks/
    └── useSubscription.ts    Client hook – booleans, canAccess(), trialDays
```

---

## How it works

### 1. Session enrichment (`session-sync.ts`)

Call `injectSubscriptionIntoToken()` inside the Next Auth `jwt` callback, or
`syncSubscriptionToClerkMetadata()` after a subscription change. This ensures
the token always carries:

```ts
{ plan: "pro", status: "active", trialEndsAt?: string }
```

### 2. Middleware check (`middleware.ts`)

`subscriptionMiddleware(req)` runs on every request matched by the exported
Next.js `config.matcher`. It:

1. Reads the `authjs.session-token` (Next Auth) or `__session` (Clerk) cookie.
2. Decodes the JWT payload with **`atob()`** — no Node.js crypto, Edge-safe.
3. Normalises Razorpay-specific status strings (`"halted"` → `"past_due"`) and
   Stripe statuses (`"trialing"` → `"trialing"`).
4. Checks the decoded plan against `SUBSCRIPTION_RULES` from `config.ts`.
5. If access is denied, redirects to `/upgrade?reason=<UpgradeReason>&from=<path>`.

### 3. Toast (`components/UpgradeToast.tsx`)

The upgrade page renders `<UpgradeToast />`. On mount it reads the `reason` and
`from` search params and fires a **sonner** toast with a `Back` action link.
The toast has `id: "subscription-guard"` to prevent duplicates on fast nav.

### 4. Server Component gate (`components/SubscriptionGate.tsx`)

```tsx
<SubscriptionGate requiredPlan="pro">
  <ProFeature />
</SubscriptionGate>
```

Set `hardRedirect` to send the user to `/upgrade` instead of rendering
`fallback`.

---

## Installation

```bash
npm install sonner next-auth@beta
# or
npm install sonner @clerk/nextjs
```

---

## Configuration

Edit `config.ts`:

```ts
export const SUBSCRIPTION_RULES: RouteGuardRule[] = [
  { pathPrefix: "/dashboard/analytics", requiredPlan: "pro" },
  { pathPrefix: "/dashboard/ai", requiredPlan: "enterprise" },
];
```

---

## What can go wrong

| Issue                                    | Cause                                           | Fix                                                                           |
| ---------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------- |
| Middleware sees no subscription          | Token doesn't include plan/status fields        | Call `injectSubscriptionIntoToken()` in the jwt callback                      |
| Status always `"inactive"` with Razorpay | Razorpay returns `"halted"` not `"inactive"`    | Already handled in `normaliseStatus()` in middleware.ts                       |
| Toast fires twice                        | Multiple navigations hit `/upgrade`             | Toast has a stable `id`; ensure `<Toaster />` is rendered once                |
| Edge runtime error                       | Importing something that uses Node.js crypto    | Keep middleware.ts import chain Edge-clean; don't import `auth.ts`            |
| Plan hierarchy wrong                     | `meetsMinimumPlan("pro", "enterprise")` → true? | `PLAN_HIERARCHY` is `["free","pro","enterprise"]`; higher index = higher tier |
