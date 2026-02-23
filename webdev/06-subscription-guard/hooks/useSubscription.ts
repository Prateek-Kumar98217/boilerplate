/**
 * useSubscription — Client Component hook for subscription state.
 *
 * Reads subscription data from the next-auth/react session or Clerk
 * sessionClaims and exposes a clean, typed API.
 *
 * Includes plan comparison helpers so components can gate UI without
 * duplicating logic.
 */
"use client";

import { useMemo } from "react";
import { meetsMinimumPlan, isAccessGranted, PLAN_HIERARCHY } from "../types";
import type { SubscriptionPlan, SubscriptionStatus } from "../types";

// ─── Input ────────────────────────────────────────────────────────────────────

type RawSubscription = {
  plan: SubscriptionPlan;
  status: SubscriptionStatus;
  trialEndsAt?: string | null;
  customerId?: string | null;
};

// ─── Return type ──────────────────────────────────────────────────────────────

export type UseSubscriptionReturn = {
  plan: SubscriptionPlan;
  status: SubscriptionStatus;
  isActive: boolean;
  isTrialing: boolean;
  isPastDue: boolean;
  isCanceled: boolean;
  isFree: boolean;
  isPro: boolean;
  isEnterprise: boolean;
  trialEndsAt: Date | null;
  /** True if the user meets `plan` as a minimum requirement. */
  canAccess: (requiredPlan: SubscriptionPlan) => boolean;
  /** True if the user has exactly this plan. */
  hasPlan: (plan: SubscriptionPlan) => boolean;
  /** Days remaining in trial, or null if not trialing. */
  trialDaysRemaining: number | null;
};

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useSubscription(raw: RawSubscription): UseSubscriptionReturn {
  return useMemo(() => {
    const { plan, status, trialEndsAt, customerId: _customerId } = raw;

    const trialDate = trialEndsAt ? new Date(trialEndsAt) : null;

    const trialDaysRemaining =
      trialDate && status === "trialing"
        ? Math.max(
            0,
            Math.ceil(
              (trialDate.getTime() - Date.now()) / (1000 * 60 * 60 * 24),
            ),
          )
        : null;

    return {
      plan,
      status,
      isActive: status === "active",
      isTrialing: status === "trialing",
      isPastDue: status === "past_due",
      isCanceled: status === "canceled",
      isFree: plan === "free",
      isPro: plan === "pro",
      isEnterprise: plan === "enterprise",
      trialEndsAt: trialDate,
      canAccess: (requiredPlan: SubscriptionPlan) =>
        isAccessGranted(status) && meetsMinimumPlan(plan, requiredPlan),
      hasPlan: (p: SubscriptionPlan) => plan === p,
      trialDaysRemaining,
    };
  }, [raw]);
}

// ─── Next Auth variant ────────────────────────────────────────────────────────

/**
 * useSubscriptionFromNextAuth — reads subscription data from the next-auth session.
 *
 * Requires:
 *  1. `useSession` from next-auth/react (must be inside <SessionProvider>)
 *  2. Subscription fields stored in the JWT token via the `jwt` callback.
 *
 * Usage:
 *
 *   const { isActive, canAccess } = useSubscriptionFromNextAuth();
 *   if (!canAccess("pro")) return <UpgradeBanner />;
 *
 * To enable, uncomment this function and install next-auth:
 *   npm install next-auth@beta
 */

// import { useSession } from "next-auth/react";
//
// export function useSubscriptionFromNextAuth(): UseSubscriptionReturn & { isLoading: boolean } {
//   const { data: session, status } = useSession();
//
//   const raw: RawSubscription = {
//     plan: (session?.user as { subscription_plan?: SubscriptionPlan })?.subscription_plan ?? "free",
//     status: (session?.user as { subscription_status?: SubscriptionStatus })?.subscription_status ?? "inactive",
//     trialEndsAt: (session?.user as { subscription_trial_ends_at?: string })?.subscription_trial_ends_at,
//   };
//
//   const subscription = useSubscription(raw);
//   return { ...subscription, isLoading: status === "loading" };
// }

// ─── Clerk variant ────────────────────────────────────────────────────────────

/**
 * useSubscriptionFromClerk — reads subscription data from Clerk session claims.
 *
 * Requires:
 *  1. `useAuth` from @clerk/nextjs
 *  2. Subscription fields stored in Clerk user publicMetadata.
 *
 * To enable, uncomment this function and install @clerk/nextjs:
 *   npm install @clerk/nextjs
 */

// import { useAuth } from "@clerk/nextjs";
//
// export function useSubscriptionFromClerk(): UseSubscriptionReturn & { isLoading: boolean } {
//   const { isLoaded, isSignedIn, sessionClaims } = useAuth();
//   const meta = sessionClaims?.metadata as Record<string, string> | undefined;
//
//   const raw: RawSubscription = {
//     plan: (meta?.["subscription_plan"] as SubscriptionPlan | undefined) ?? "free",
//     status: (meta?.["subscription_status"] as SubscriptionStatus | undefined) ?? "inactive",
//     trialEndsAt: meta?.["subscription_trial_ends_at"],
//   };
//
//   const subscription = useSubscription(raw);
//   return { ...subscription, isLoading: !isLoaded };
// }

export { PLAN_HIERARCHY };
