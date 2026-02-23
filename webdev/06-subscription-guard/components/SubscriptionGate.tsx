/**
 * SubscriptionGate — Server Component wrapper for subscription-gated UI.
 *
 * Unlike the middleware (which guards full routes), SubscriptionGate
 * conditionally renders content within a page — e.g. to show a locked
 * feature behind a paywall callout rather than a full redirect.
 *
 * Usage:
 *
 *   <SubscriptionGate requiredPlan="pro" fallback={<UpgradeBanner />}>
 *     <AnalyticsDashboard />
 *   </SubscriptionGate>
 *
 * For hard redirects, use the middleware or SubscriptionGate with `hardRedirect`.
 */
import { redirect } from "next/navigation";
import type { ReactNode } from "react";
import { meetsMinimumPlan, isAccessGranted } from "../types";
import type { SubscriptionPlan } from "../types";

// ─── Session reading helpers ──────────────────────────────────────────────────
// These import from the auth layer. Swap for your actual session helper.
// Next Auth:  import { auth } from "@/auth"
// Clerk:      import { auth } from "@clerk/nextjs/server"

type SessionProvider = "next-auth" | "clerk";

// We keep this generic so the file compiles without a concrete auth import.
// Replace the stub below with your real session function.
async function getSubscriptionFromSession() {
  // ── Next Auth ──────────────────────────────────────────────────────────────
  // const session = await auth();
  // if (!session?.user) return null;
  // return {
  //   plan: (session.user as { subscription_plan?: SubscriptionPlan }).subscription_plan ?? "free",
  //   status: (session.user as { subscription_status?: string }).subscription_status ?? "inactive",
  // };

  // ── Clerk ──────────────────────────────────────────────────────────────────
  // const { sessionClaims } = await auth();
  // const meta = sessionClaims?.metadata as Record<string, string> | undefined;
  // return {
  //   plan: (meta?.["subscription_plan"] as SubscriptionPlan | undefined) ?? "free",
  //   status: meta?.["subscription_status"] ?? "inactive",
  // };

  // Stub — replace with your auth integration.
  return { plan: "free" as SubscriptionPlan, status: "active" as const };
}

// ─── Component ────────────────────────────────────────────────────────────────

type SubscriptionGateProps = {
  children: ReactNode;
  requiredPlan: SubscriptionPlan;
  /**
   * Rendered when the user does not meet the plan requirement.
   * If not provided and `hardRedirect` is false, renders nothing.
   */
  fallback?: ReactNode;
  /**
   * When true, redirects to /upgrade instead of rendering fallback.
   * Default: false.
   */
  hardRedirect?: boolean;
  redirectTo?: string;
};

export async function SubscriptionGate({
  children,
  requiredPlan,
  fallback = null,
  hardRedirect = false,
  redirectTo = "/upgrade",
}: SubscriptionGateProps) {
  const subscription = await getSubscriptionFromSession();

  const hasAccess =
    subscription !== null &&
    isAccessGranted(
      subscription.status as
        | "active"
        | "trialing"
        | "past_due"
        | "canceled"
        | "unpaid"
        | "inactive",
    ) &&
    meetsMinimumPlan(subscription.plan, requiredPlan);

  if (!hasAccess) {
    if (hardRedirect) {
      redirect(`${redirectTo}?reason=plan_required`);
    }
    return <>{fallback}</>;
  }

  return <>{children}</>;
}
