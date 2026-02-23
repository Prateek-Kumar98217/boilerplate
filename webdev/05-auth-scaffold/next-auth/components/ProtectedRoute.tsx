/**
 * ProtectedRoute — Server Component wrapper for auth-gated pages.
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
 * For Client Components, use the useRequireAuth hook instead.
 */
import { redirect } from "next/navigation";
import { auth } from "../auth";
import type { ReactNode } from "react";

type ProtectedRouteProps = {
  children: ReactNode;
  /**
   * Where to redirect unauthenticated users.
   * Default: "/auth/sign-in"
   */
  redirectTo?: string;
  /**
   * Optional role check. If the user's role is not in this list,
   * they are redirected to `forbiddenRedirectTo`.
   * Requires `role` to be stored in session.user (extend NextAuth types).
   */
  allowedRoles?: string[];
  forbiddenRedirectTo?: string;
};

export async function ProtectedRoute({
  children,
  redirectTo = "/auth/sign-in",
  allowedRoles,
  forbiddenRedirectTo = "/",
}: ProtectedRouteProps) {
  const session = await auth();

  if (!session?.user?.id) {
    redirect(redirectTo);
  }

  // Role-based access control (requires role in session, see auth.ts type augmentation).
  if (allowedRoles && allowedRoles.length > 0) {
    const userRole = (session.user as { role?: string }).role;
    if (!userRole || !allowedRoles.includes(userRole)) {
      redirect(forbiddenRedirectTo);
    }
  }

  return <>{children}</>;
}

// ─── Server-side session assertion (for server actions / route handlers) ───────

/**
 * Asserts that the current request has a valid session.
 * Redirects to the sign-in page if not.
 * Use in Server Components or Server Actions when you need the session data.
 */
export async function requireSession(redirectTo = "/auth/sign-in") {
  const session = await auth();
  if (!session?.user?.id) redirect(redirectTo);
  return session;
}
