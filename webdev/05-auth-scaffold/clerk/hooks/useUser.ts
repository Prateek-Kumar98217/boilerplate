/**
 * useUser — Clerk user data hook for Client Components.
 *
 * Returns the typed Clerk user object or null,
 * with a consistent shape matching the NextAuth variant.
 */
"use client";

import { useUser as useClerkUser } from "@clerk/nextjs";

// ─── Types ────────────────────────────────────────────────────────────────────

export type AppUser = {
  id: string;
  firstName: string | null;
  lastName: string | null;
  fullName: string | null;
  email: string | null;
  imageUrl: string | null;
  /** ISO timestamp string */
  createdAt: string | null;
};

export type UseUserReturn =
  | { user: AppUser; isLoading: false }
  | { user: null; isLoading: true }
  | { user: null; isLoading: false };

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useUser(): UseUserReturn {
  const { isLoaded, isSignedIn, user } = useClerkUser();

  if (!isLoaded) {
    return { user: null, isLoading: true };
  }

  if (!isSignedIn || !user) {
    return { user: null, isLoading: false };
  }

  // Normalize Clerk's user object to a stable app-specific shape.
  const primaryEmail = user.primaryEmailAddress?.emailAddress ?? null;

  return {
    user: {
      id: user.id,
      firstName: user.firstName ?? null,
      lastName: user.lastName ?? null,
      fullName: user.fullName ?? null,
      email: primaryEmail,
      imageUrl: user.imageUrl ?? null,
      createdAt: user.createdAt ? new Date(user.createdAt).toISOString() : null,
    },
    isLoading: false,
  };
}
