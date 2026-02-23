/**
 * Clerk middleware — protects routes and controls access.
 *
 * File: middleware.ts  (place at your project root, alongside next.config.ts)
 *
 * Install: npm install @clerk/nextjs
 * Docs:    https://clerk.com/docs/quickstarts/nextjs
 *
 * Env vars (required):
 *   NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_...
 *   CLERK_SECRET_KEY=sk_...
 *
 * Env vars (optional — customize URLs):
 *   NEXT_PUBLIC_CLERK_SIGN_IN_URL=/auth/sign-in
 *   NEXT_PUBLIC_CLERK_SIGN_UP_URL=/auth/sign-up
 *   NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/dashboard
 *   NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/dashboard
 */
import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

// ─── Route matchers ───────────────────────────────────────────────────────────

/** All routes that don't require authentication. */
const isPublicRoute = createRouteMatcher([
  "/",
  "/auth/sign-in(.*)",
  "/auth/sign-up(.*)",
  "/api/auth(.*)", // If using Next Auth alongside — unlikely but possible
  "/api/webhooks(.*)", // Clerk webhooks must be public
  "/blog(.*)", // Example public content
  "/:path*/opengraph-image", // Open Graph images
]);

/** Routes that require a specific organization role. */
const isAdminRoute = createRouteMatcher(["/admin(.*)"]);

// ─── Middleware ───────────────────────────────────────────────────────────────

export default clerkMiddleware(async (auth, req) => {
  // ── Block unauthenticated access to protected routes ──────────────────
  if (!isPublicRoute(req)) {
    await auth.protect(); // Redirects to sign-in if not authenticated
  }

  // ── Role-based access for admin routes ────────────────────────────────
  if (isAdminRoute(req)) {
    const { sessionClaims } = await auth();

    // Adjust the role claim key to match your Clerk metadata structure.
    const role = (sessionClaims?.metadata as { role?: string } | undefined)
      ?.role;

    if (role !== "admin") {
      const url = req.nextUrl.clone();
      url.pathname = "/";
      return NextResponse.redirect(url);
    }
  }

  return NextResponse.next();
});

// ─── Route matcher ────────────────────────────────────────────────────────────

export const config = {
  matcher: [
    /*
     * Match all paths except static files and Next.js internals.
     * Clerk processes the auth state but the clerkMiddleware decides
     * whether to act on each route.
     */
    "/((?!_next/static|_next/image|favicon\\.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico)$).*)",
  ],
};
