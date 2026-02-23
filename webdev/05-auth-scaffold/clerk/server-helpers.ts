/**
 * Clerk server-side helpers for Server Actions and Route Handlers.
 *
 * Provides type-safe wrappers around Clerk's server APIs.
 */
import { auth, currentUser } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";

// ─── Types ────────────────────────────────────────────────────────────────────

export type ClerkServerContext = {
  userId: string;
  orgId: string | null;
  orgRole: string | null;
  /** Raw session claims — use for custom JWT claims. */
  sessionClaims: Record<string, unknown>;
};

// ─── Auth helpers ─────────────────────────────────────────────────────────────

/**
 * Returns auth context if the user is signed in, or redirects.
 * Use in Server Components, Server Actions, and Route Handlers.
 */
export async function requireClerkSession(
  redirectTo = "/auth/sign-in",
): Promise<ClerkServerContext> {
  const { userId, orgId, orgRole, sessionClaims } = await auth();

  if (!userId) redirect(redirectTo);

  return {
    userId,
    orgId: orgId ?? null,
    orgRole: orgRole ?? null,
    sessionClaims: (sessionClaims as Record<string, unknown>) ?? {},
  };
}

/**
 * Returns the full Clerk User object for the current request.
 * More detailed than the session — includes email addresses, profile image, etc.
 * Makes an additional request to Clerk's API.
 */
export async function getCurrentClerkUser() {
  const user = await currentUser();
  if (!user) return null;

  return {
    id: user.id,
    firstName: user.firstName,
    lastName: user.lastName,
    fullName: `${user.firstName ?? ""} ${user.lastName ?? ""}`.trim() || null,
    email: user.emailAddresses[0]?.emailAddress ?? null,
    imageUrl: user.imageUrl,
    createdAt: new Date(user.createdAt).toISOString(),
    // Custom public metadata (set via Clerk dashboard or API).
    metadata: user.publicMetadata as Record<string, unknown>,
  };
}

/**
 * Returns the user's role from public metadata.
 * Requires the role to be set in Clerk's user public metadata.
 */
export async function getUserRole(): Promise<string | null> {
  const user = await currentUser();
  if (!user) return null;
  return (user.publicMetadata as { role?: string })?.role ?? null;
}
