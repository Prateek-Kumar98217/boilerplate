/**
 * Next.js middleware using NextAuth v5.
 *
 * File: middleware.ts  (place at your project root, alongside next.config.ts)
 *
 * How it works:
 *  - Auth.js processes the request and calls `authorized` in auth.config.ts.
 *  - The matcher controls which paths invoke the middleware.
 *  - Public paths are whitelisted so static assets and API routes are not blocked.
 *
 * Pattern:
 *  1. Import only from auth.config.ts — NOT from auth.ts — to keep the
 *     middleware bundle Edge-compatible (no Node.js APIs).
 *  2. Use matchers to avoid running middleware on static files / images.
 */
import NextAuth from "next-auth";
import { authConfig } from "./auth.config";

// ─── Middleware ───────────────────────────────────────────────────────────────

export const { auth: middleware } = NextAuth(authConfig);

// ─── Route matcher ────────────────────────────────────────────────────────────

export const config = {
  matcher: [
    /*
     * Match all paths EXCEPT:
     *  - _next/static  (static assets)
     *  - _next/image   (image optimization)
     *  - favicon.ico
     *  - Files with extensions (images, fonts, etc.)
     *  - /api/auth      (NextAuth's own endpoints)
     */
    "/((?!_next/static|_next/image|favicon\\.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico|css|js)$|api/auth).*)",
  ],
};
