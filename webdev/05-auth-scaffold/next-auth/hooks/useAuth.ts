/**
 * useAuth — Client Component hook for auth state.
 *
 * Wraps next-auth/react's useSession with a clean, typed API.
 * Provides loading state, isAuthenticated flag, and sign-in/out helpers.
 *
 * Must be used inside <SessionProvider> (wrap your root layout).
 */
"use client";

import { useSession, signIn, signOut } from "next-auth/react";
import { useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

export type AuthStatus = "loading" | "authenticated" | "unauthenticated";

export type UseAuthReturn = {
  status: AuthStatus;
  isLoading: boolean;
  isAuthenticated: boolean;
  isUnauthenticated: boolean;
  /** Redirect to the sign-in page (or a provider's OAuth flow). */
  signIn: (provider?: string, callbackUrl?: string) => Promise<void>;
  /** Sign out and redirect. */
  signOut: (callbackUrl?: string) => Promise<void>;
};

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useAuth(): UseAuthReturn {
  const { status } = useSession();

  const handleSignIn = useCallback(
    async (provider?: string, callbackUrl?: string) => {
      await signIn(provider, {
        callbackUrl: callbackUrl ?? window.location.href,
      });
    },
    [],
  );

  const handleSignOut = useCallback(async (callbackUrl?: string) => {
    await signOut({ callbackUrl: callbackUrl ?? "/" });
  }, []);

  return {
    status,
    isLoading: status === "loading",
    isAuthenticated: status === "authenticated",
    isUnauthenticated: status === "unauthenticated",
    signIn: handleSignIn,
    signOut: handleSignOut,
  };
}
