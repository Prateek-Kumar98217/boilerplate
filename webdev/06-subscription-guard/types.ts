/**
 * Subscription types.
 *
 * SubscriptionPlan is a strict union — add tiers here as your product grows.
 * SubscriptionStatus mirrors Stripe's subscription status values (Razorpay uses
 * "active" / "halted" / "cancelled" — normalised to the same type below).
 */

// ─── Plans ────────────────────────────────────────────────────────────────────

export type SubscriptionPlan = "free" | "pro" | "enterprise";

/**
 * Ordered plan hierarchy — used to check whether a user meets a minimum plan.
 * Higher index = higher tier.
 */
export const PLAN_HIERARCHY: SubscriptionPlan[] = ["free", "pro", "enterprise"];

/** Returns true if `userPlan` satisfies the `requiredPlan` minimum. */
export function meetsMinimumPlan(
  userPlan: SubscriptionPlan,
  requiredPlan: SubscriptionPlan,
): boolean {
  return (
    PLAN_HIERARCHY.indexOf(userPlan) >= PLAN_HIERARCHY.indexOf(requiredPlan)
  );
}

// ─── Status ───────────────────────────────────────────────────────────────────

/**
 * Normalised subscription status — covers both Stripe and Razorpay states.
 *
 * Stripe:   active | trialing | past_due | canceled | unpaid | incomplete | incomplete_expired
 * Razorpay: created | authenticated | active | pending | halted | cancelled | completed | expired
 *
 * All statuses that grant access map to "active" | "trialing".
 * All statuses that should block access map to "past_due" | "canceled" | "unpaid".
 */
export type SubscriptionStatus =
  | "active"
  | "trialing"
  | "past_due"
  | "canceled"
  | "unpaid"
  | "inactive"; // default when no subscription record exists

/** Returns true if the status permits access to paid features. */
export function isAccessGranted(status: SubscriptionStatus): boolean {
  return status === "active" || status === "trialing";
}

// ─── Session shape ────────────────────────────────────────────────────────────

/**
 * The subscription fields that must be stored in the session JWT.
 *
 * For Next Auth: add these via the `jwt` callback in auth.config.ts.
 * For Clerk:     store them in `publicMetadata` and read via sessionClaims.
 */
export type SessionSubscription = {
  plan: SubscriptionPlan;
  status: SubscriptionStatus;
  /** ISO date string — present when trialing, needed for UI countdown. */
  trialEndsAt?: string;
  /** Your internal customer/subscription ID for portal links. */
  customerId?: string;
};

// ─── Route guard config ───────────────────────────────────────────────────────

export type RouteGuardRule = {
  /** Glob-style path prefix, e.g. "/admin" or "/dashboard/billing" */
  pathPrefix: string;
  /** The minimum plan required to access this route. */
  requiredPlan: SubscriptionPlan;
  /**
   * Redirect target when the user does not meet the plan requirement.
   * Defaults to "/upgrade". A `reason` search param is appended automatically.
   */
  redirectTo?: string;
};
