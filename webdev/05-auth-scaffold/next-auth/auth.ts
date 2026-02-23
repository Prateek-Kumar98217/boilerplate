/**
 * NextAuth v5 — main auth instance.
 *
 * File: auth.ts   (place at your project root or src/)
 * Import: import { auth, signIn, signOut, handlers } from "@/auth"
 *
 * This file can import Node.js-only adapters (Drizzle, Prisma, etc).
 * Do NOT import this in middleware — use auth.config.ts there instead.
 */
import NextAuth from "next-auth";
import { DrizzleAdapter } from "@auth/drizzle-adapter";
import { authConfig } from "./auth.config";
import { db } from "@/db"; // your Drizzle db instance

export const {
  handlers, // { GET, POST } — export from app/api/auth/[...nextauth]/route.ts
  auth, // session helper — server components, server actions, route handlers
  signIn, // programmatic sign-in
  signOut, // programmatic sign-out
} = NextAuth({
  ...authConfig,
  adapter: DrizzleAdapter(db),
});

// ─── Type augmentation ────────────────────────────────────────────────────────
// Extend the built-in types so session.user.id is always typed as string
// instead of string | undefined.

declare module "next-auth" {
  interface Session {
    user: {
      id: string;
      name?: string | null;
      email?: string | null;
      image?: string | null;
    };
  }
}
