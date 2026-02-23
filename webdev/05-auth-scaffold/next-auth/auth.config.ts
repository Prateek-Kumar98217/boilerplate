/**
 * NextAuth v5 (next-auth@beta) — core auth configuration.
 *
 * File: auth.config.ts
 * Kept separate from auth.ts so it can be imported in middleware
 * without pulling in heavy Node.js-only dependencies.
 *
 * Docs: https://authjs.dev/getting-started
 * Install: npm install next-auth@beta
 */
import type { NextAuthConfig } from "next-auth";
import Credentials from "next-auth/providers/credentials";
import GitHub from "next-auth/providers/github";
import Google from "next-auth/providers/google";
import { z } from "zod";

// ─── Validation schema for credentials login ──────────────────────────────────

const CredentialsSchema = z.object({
  email: z.string().email("Invalid email."),
  password: z.string().min(8, "Password must be at least 8 characters."),
});

// ─── Auth config  ─────────────────────────────────────────────────────────────

export const authConfig: NextAuthConfig = {
  // ── Providers ──────────────────────────────────────────────────────────────
  providers: [
    GitHub({
      clientId: process.env.GITHUB_CLIENT_ID!,
      clientSecret: process.env.GITHUB_CLIENT_SECRET!,
    }),
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
    Credentials({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        // Never trust raw `credentials` — always validate with Zod.
        const parsed = CredentialsSchema.safeParse(credentials);
        if (!parsed.success) return null;

        // Replace with your real DB lookup + password comparison.
        // NEVER store plain-text passwords. Use bcrypt/argon2.
        const { email, password } = parsed.data;

        // Example (swap with real implementation):
        // const user = await db.query.users.findFirst({ where: eq(users.email, email) });
        // if (!user) return null;
        // const isValid = await bcrypt.compare(password, user.passwordHash);
        // if (!isValid) return null;
        // return { id: user.id, email: user.email, name: user.name };

        // Placeholder — remove in production:
        if (email === "test@example.com" && password === "Password1") {
          return { id: "test-user-id", email, name: "Test User" };
        }

        return null;
      },
    }),
  ],

  // ── Custom pages ──────────────────────────────────────────────────────────
  pages: {
    signIn: "/auth/sign-in",
    signOut: "/auth/sign-out",
    error: "/auth/error",
    newUser: "/auth/new-user",
  },

  // ── Callbacks ─────────────────────────────────────────────────────────────
  callbacks: {
    /**
     * jwt — runs whenever a JWT is created or updated.
     * Persist extra data (role, userId) into the token so it's available
     * in session without an extra DB round-trip.
     */
    async jwt({ token, user, account }) {
      if (user) {
        token.id = user.id;
        // Optionally fetch role from DB here and store in token.
        // const dbUser = await db.query.users.findFirst(...);
        // token.role = dbUser?.role ?? "user";
      }
      return token;
    },

    /**
     * session — exposes safe data to the client.
     * NEVER put secrets or sensitive data in the session object.
     */
    async session({ session, token }) {
      if (token.id && typeof token.id === "string") {
        session.user.id = token.id;
      }
      return session;
    },

    /**
     * authorized — controls whether a request is allowed.
     * Used by middleware to protect routes without a full session lookup.
     */
    authorized({ auth, request }) {
      const isLoggedIn = !!auth?.user;
      const isOnAuthPage = request.nextUrl.pathname.startsWith("/auth");

      if (isOnAuthPage) {
        // Redirect logged-in users away from auth pages.
        if (isLoggedIn)
          return Response.redirect(new URL("/dashboard", request.nextUrl));
        return true;
      }

      return isLoggedIn;
    },
  },

  // ── Session strategy ──────────────────────────────────────────────────────
  session: { strategy: "jwt" },
};
