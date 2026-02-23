/**
 * Session sync helpers — store subscription data in Next Auth JWT / Clerk publicMetadata.
 *
 * ┌────────────────────────────────────────────────────────────────────────────┐
 * │  Data flow                                                                 │
 * │                                                                            │
 * │  Stripe/Razorpay webhook → /api/webhooks/[provider]                       │
 * │      │                                                                     │
 * │      ▼                                                                     │
 * │  updateUserSubscription()  — writes to your DB                            │
 * │      │                                                                     │
 * │      ▼                                                                     │
 * │  Next Auth jwt() callback  — reads from DB on each token refresh          │
 * │  Clerk updateUserMetadata() — pushes to Clerk's publicMetadata            │
 * │      │                                                                     │
 * │      ▼                                                                     │
 * │  subscriptionMiddleware()  — reads from token cookie                      │
 * └────────────────────────────────────────────────────────────────────────────┘
 */

import type { SubscriptionPlan, SubscriptionStatus } from "./types";

// ─── DB shape (adapt to your schema) ─────────────────────────────────────────

export type SubscriptionRecord = {
  userId: string;
  plan: SubscriptionPlan;
  status: SubscriptionStatus;
  trialEndsAt: Date | null;
  /** Stripe customer ID OR Razorpay customer ID */
  customerId: string | null;
  /** Stripe subscription ID OR Razorpay subscription ID */
  subscriptionId: string | null;
  /** ISO string of the current period end */
  currentPeriodEnd: Date | null;
  provider: "stripe" | "razorpay";
};

// ─── Next Auth: jwt() callback integration ────────────────────────────────────

/**
 * Paste this inside your `callbacks.jwt` in auth.config.ts.
 *
 * It fetches the subscription record from your DB on *every* token refresh,
 * guaranteeing the middleware always has up-to-date plan/status data.
 *
 * For performance, cache the DB result for the duration of the token's TTL.
 */
export async function injectSubscriptionIntoToken(
  token: Record<string, unknown>,
  // Pass your db.query.subscriptions.findFirst here to avoid a direct import.
  findSubscription: (userId: string) => Promise<SubscriptionRecord | null>,
): Promise<Record<string, unknown>> {
  const userId = token["id"] as string | undefined;
  if (!userId) return token;

  const record = await findSubscription(userId);

  return {
    ...token,
    subscription_plan: record?.plan ?? "free",
    subscription_status: record?.status ?? "inactive",
    subscription_trial_ends_at: record?.trialEndsAt?.toISOString() ?? null,
    stripe_customer_id:
      record?.provider === "stripe" ? record.customerId : null,
    razorpay_customer_id:
      record?.provider === "razorpay" ? record.customerId : null,
  };
}

/**
 * Example integration in auth.config.ts:
 *
 *   import { injectSubscriptionIntoToken } from "@/06-subscription-guard/session-sync";
 *   import { db } from "@/db";
 *   import { subscriptions } from "@/db/schema";
 *   import { eq } from "drizzle-orm";
 *
 *   callbacks: {
 *     async jwt({ token, user }) {
 *       if (user) token.id = user.id;               // initial sign-in
 *       return injectSubscriptionIntoToken(
 *         token as Record<string, unknown>,
 *         (uid) => db.query.subscriptions.findFirst({ where: eq(subscriptions.userId, uid) })
 *       );
 *     },
 *     async session({ session, token }) {
 *       session.user.id = token.id as string;
 *       return session;
 *     }
 *   }
 */

// ─── Clerk: publicMetadata update (called from webhook handler) ───────────────

/**
 * Updates Clerk user's publicMetadata with subscription data.
 * Call this from your Stripe/Razorpay webhook handler after confirmed payment.
 *
 * Requires: npm install @clerk/backend
 */
export async function syncSubscriptionToClerkMetadata(params: {
  clerkUserId: string;
  plan: SubscriptionPlan;
  status: SubscriptionStatus;
  trialEndsAt?: Date | null;
  customerId?: string | null;
}): Promise<void> {
  // Lazy import to avoid pulling @clerk/backend into non-Clerk projects.
  const { createClerkClient } = await import("@clerk/backend");

  const clerk = createClerkClient({
    secretKey: process.env.CLERK_SECRET_KEY,
  });

  await clerk.users.updateUserMetadata(params.clerkUserId, {
    publicMetadata: {
      subscription_plan: params.plan,
      subscription_status: params.status,
      subscription_trial_ends_at: params.trialEndsAt?.toISOString() ?? null,
      customerId: params.customerId ?? null,
    },
  });
}
