/**
 * Subscription-aware Next.js middleware.
 *
 * Reads subscription plan and status from the session JWT, then
 * redirects under-privileged users to /upgrade (or a custom path).
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  How the toast message works                                    │
 * │                                                                 │
 * │  Middleware cannot directly trigger a React toast, so it        │
 * │  appends a `reason` query parameter to the redirect URL.        │
 * │  The /upgrade page (or a layout) reads this param and renders   │
 * │  the <UpgradeToast> Client Component, which fires the toast     │
 * │  on mount.                                                      │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * Compatible with:
 *  - Next Auth v5: subscription fields stored via jwt() callback
 *  - Clerk:        subscription fields read from sessionClaims.metadata
 *
 * File placement: middleware.ts (project root, runs before auth middleware)
 * If using alongside Next Auth, compose them — see bottom of this file.
 */
import { NextRequest, NextResponse } from "next/server";
import { SUBSCRIPTION_RULES, ALWAYS_ALLOWED_PREFIXES } from "./config";
import { meetsMinimumPlan, isAccessGranted } from "./types";
import type { SessionSubscription, SubscriptionPlan } from "./types";

// ─── Session reading ──────────────────────────────────────────────────────────

/**
 * Extracts subscription data from the session token.
 *
 * Next Auth stores the JWT in an httpOnly cookie. We decode the
 * payload without verifying the signature here (Edge-safe, fast).
 * Signature verification happened when the session was established.
 *
 * For Clerk, the JWT is in the `__session` cookie and sessionClaims
 * are accessible via auth().  At middleware level, we read the raw
 * token the same way.
 */
function extractSubscriptionFromToken(
  request: NextRequest,
): SessionSubscription | null {
  // Next Auth v5 session token cookie name.
  const tokenCookieName =
    process.env.NODE_ENV === "production"
      ? "__Secure-authjs.session-token"
      : "authjs.session-token";

  // Clerk uses __session.
  const clerkTokenCookieName = "__session";

  const rawToken =
    request.cookies.get(tokenCookieName)?.value ??
    request.cookies.get(clerkTokenCookieName)?.value;

  if (!rawToken) return null;

  try {
    // JWT is base64url encoded: header.payload.signature
    const parts = rawToken.split(".");
    if (parts.length !== 3 || !parts[1]) return null;

    // Decode payload (base64url → JSON).
    const payloadBase64 = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    const padding = "=".repeat((4 - (payloadBase64.length % 4)) % 4);
    const decoded = atob(payloadBase64 + padding);
    const payload = JSON.parse(decoded) as Record<string, unknown>;

    // Next Auth stores custom claims at the top level of the JWT.
    // Clerk stores them under `metadata` or a custom key.
    const plan =
      (payload["subscription_plan"] as SubscriptionPlan | undefined) ??
      ((payload["metadata"] as Record<string, unknown> | undefined)?.[
        "subscription_plan"
      ] as SubscriptionPlan | undefined) ??
      "free";

    const status =
      (payload["subscription_status"] as string | undefined) ??
      ((payload["metadata"] as Record<string, unknown> | undefined)?.[
        "subscription_status"
      ] as string | undefined) ??
      "inactive";

    const trialEndsAt =
      (payload["subscription_trial_ends_at"] as string | undefined) ??
      ((payload["metadata"] as Record<string, unknown> | undefined)?.[
        "subscription_trial_ends_at"
      ] as string | undefined);

    const customerId =
      (payload["stripe_customer_id"] as string | undefined) ??
      (payload["razorpay_customer_id"] as string | undefined) ??
      ((payload["metadata"] as Record<string, unknown> | undefined)?.[
        "customerId"
      ] as string | undefined);

    // Normalise Razorpay-specific statuses → our unified type.
    const normalisedStatus = normaliseStatus(status);

    return {
      plan: isValidPlan(plan) ? plan : "free",
      status: normalisedStatus,
      trialEndsAt,
      customerId,
    };
  } catch {
    // Malformed token — treat as unauthenticated / free.
    return null;
  }
}

// ─── Status normalisation ─────────────────────────────────────────────────────

function normaliseStatus(raw: string): SessionSubscription["status"] {
  switch (raw) {
    case "active":
    case "trialing":
      return raw;
    case "authenticated": // Razorpay pre-payment state
    case "created":
      return "inactive";
    case "halted": // Razorpay failed payment
    case "past_due":
    case "unpaid":
    case "incomplete":
      return "past_due";
    case "cancelled": // Razorpay
    case "canceled": // Stripe
    case "incomplete_expired":
    case "expired":
    case "completed": // Razorpay one-time plan completed
      return "canceled";
    default:
      return "inactive";
  }
}

function isValidPlan(value: unknown): value is SubscriptionPlan {
  return value === "free" || value === "pro" || value === "enterprise";
}

// ─── Redirect reasons ─────────────────────────────────────────────────────────

export type UpgradeReason =
  | "plan_required" // user's plan is below required tier
  | "subscription_expired"
  | "subscription_past_due"
  | "subscription_inactive";

function resolveReason(
  subscription: SessionSubscription | null,
  requiredPlan: SubscriptionPlan,
): UpgradeReason {
  if (!subscription || subscription.status === "inactive") {
    return "subscription_inactive";
  }
  if (subscription.status === "canceled") {
    return "subscription_expired";
  }
  if (subscription.status === "past_due" || subscription.status === "unpaid") {
    return "subscription_past_due";
  }
  if (!meetsMinimumPlan(subscription.plan, requiredPlan)) {
    return "plan_required";
  }
  return "plan_required"; // fallback
}

// ─── Middleware ───────────────────────────────────────────────────────────────

export function subscriptionMiddleware(
  request: NextRequest,
): NextResponse | null {
  const { pathname } = request.nextUrl;

  // 1. Skip always-allowed paths.
  for (const prefix of ALWAYS_ALLOWED_PREFIXES) {
    if (pathname.startsWith(prefix)) return null;
  }

  // 2. Find the first matching rule.
  const rule = SUBSCRIPTION_RULES.find((r) =>
    pathname.startsWith(r.pathPrefix),
  );
  if (!rule) return null; // No rule matches — allow through.

  // 3. Read subscription from session JWT.
  const subscription = extractSubscriptionFromToken(request);

  // 4. Check access: plan AND status must both pass.
  const hasValidStatus =
    subscription !== null && isAccessGranted(subscription.status);
  const hasRequiredPlan =
    subscription !== null &&
    meetsMinimumPlan(subscription.plan, rule.requiredPlan);

  if (hasValidStatus && hasRequiredPlan) {
    return null; // Access granted — continue.
  }

  // 5. Build redirect URL with reason param (used by UpgradeToast).
  const reason = resolveReason(subscription, rule.requiredPlan);
  const redirectPath = rule.redirectTo ?? "/upgrade";
  const redirectUrl = new URL(redirectPath, request.nextUrl.origin);
  redirectUrl.searchParams.set("reason", reason);
  redirectUrl.searchParams.set("from", pathname); // so /upgrade can show "you tried to access X"

  return NextResponse.redirect(redirectUrl);
}

// ─── Standalone export ────────────────────────────────────────────────────────

/**
 * Use this as your middleware when you have NO other middleware to compose with.
 *
 *   // middleware.ts
 *   export { middleware, config } from "@/06-subscription-guard/middleware";
 */
export function middleware(request: NextRequest): NextResponse {
  const subscriptionRedirect = subscriptionMiddleware(request);
  if (subscriptionRedirect) return subscriptionRedirect;
  return NextResponse.next();
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon\\.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico)$).*)",
  ],
};

// ─── Composition helper (Next Auth + Subscription) ────────────────────────────

/**
 * If you already use NextAuth middleware, compose them:
 *
 *   // middleware.ts
 *   import NextAuth from "next-auth";
 *   import { authConfig } from "./auth.config";
 *   import { subscriptionMiddleware } from "@/06-subscription-guard/middleware";
 *
 *   const { auth } = NextAuth(authConfig);
 *
 *   export default auth((req) => {
 *     // Auth passed — now check subscription.
 *     const redirect = subscriptionMiddleware(req);
 *     if (redirect) return redirect;
 *   });
 *
 *   export const config = { matcher: [...] };
 */
