/**
 * useUser — Client Component hook for current user data.
 *
 * Returns the typed user from the session, or null when not authenticated.
 * Must be used inside <SessionProvider>.
 */
"use client";

import { useSession } from "next-auth/react";

// ─── Types ────────────────────────────────────────────────────────────────────

export type SessionUser = {
  id: string;
  name: string | null;
  email: string | null;
  image: string | null;
};

export type UseUserReturn =
  | { user: SessionUser; isLoading: false }
  | { user: null; isLoading: true }
  | { user: null; isLoading: false };

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useUser(): UseUserReturn {
  const { data: session, status } = useSession();

  if (status === "loading") {
    return { user: null, isLoading: true };
  }

  if (status === "unauthenticated" || !session?.user?.id) {
    return { user: null, isLoading: false };
  }

  return {
    user: {
      id: session.user.id,
      name: session.user.name ?? null,
      email: session.user.email ?? null,
      image: session.user.image ?? null,
    },
    isLoading: false,
  };
}
