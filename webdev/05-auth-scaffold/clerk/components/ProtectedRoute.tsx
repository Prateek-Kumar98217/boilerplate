/**
 * ProtectedRoute — Server Component wrapper for Clerk-gated pages.
 *
 * Usage in a Server Component / Layout:
 *
 *   export default async function DashboardLayout({ children }) {
 *     return (
 *       <ProtectedRoute>
 *         {children}
 *       </ProtectedRoute>
 *     );
 *   }
 *
 * For Client Components, use Clerk's <SignedIn> / <SignedOut> components
 * or the useAuth hook instead.
 */
import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";
import type { ReactNode } from "react";

type ProtectedRouteProps = {
  children: ReactNode;
  redirectTo?: string;
  /**
   * Optional role check.
   * Compares against `sessionClaims.metadata.role` — set this in
   * a Clerk JWT template or session token customization.
   */
  requiredRole?: string;
  forbiddenRedirectTo?: string;
};

export async function ProtectedRoute({
  children,
  redirectTo = "/auth/sign-in",
  requiredRole,
  forbiddenRedirectTo = "/",
}: ProtectedRouteProps) {
  const { userId, sessionClaims } = await auth();

  if (!userId) {
    redirect(redirectTo);
  }

  if (requiredRole) {
    const role = (sessionClaims?.metadata as { role?: string } | undefined)
      ?.role;
    if (role !== requiredRole) {
      redirect(forbiddenRedirectTo);
    }
  }

  return <>{children}</>;
}

// ─── requireClerkAuth helper (for server actions) ─────────────────────────────

/**
 * Use in Server Actions or Route Handlers to assert authentication.
 * Returns { userId, sessionClaims } or redirects to sign-in.
 */
export async function requireClerkAuth() {
  const { userId, sessionClaims } = await auth();
  if (!userId) redirect("/auth/sign-in");
  return { userId, sessionClaims };
}
