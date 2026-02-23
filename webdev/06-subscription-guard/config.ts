/**
 * Route guard configuration.
 *
 * Define which path prefixes require which subscription plan.
 * The middleware evaluates rules top-to-bottom; the first match wins.
 *
 * Add/remove rules to match your application's route structure.
 */
import type { RouteGuardRule } from "./types";

export const SUBSCRIPTION_RULES: RouteGuardRule[] = [
  {
    pathPrefix: "/admin",
    requiredPlan: "pro",
    redirectTo: "/upgrade",
  },
  {
    pathPrefix: "/dashboard/analytics",
    requiredPlan: "pro",
    redirectTo: "/upgrade",
  },
  {
    pathPrefix: "/dashboard/exports",
    requiredPlan: "enterprise",
    redirectTo: "/upgrade",
  },
  {
    pathPrefix: "/api/export",
    requiredPlan: "enterprise",
    redirectTo: "/upgrade",
  },
];

/**
 * Paths that the subscription middleware should never touch.
 * These are skipped before any rule evaluation.
 */
export const ALWAYS_ALLOWED_PREFIXES: string[] = [
  "/upgrade",
  "/auth",
  "/api/auth",
  "/api/webhooks",
  "/_next",
  "/favicon.ico",
];
