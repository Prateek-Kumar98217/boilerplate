/**
 * useAuth — Clerk auth state hook for Client Components.
 *
 * Wraps Clerk's useAuth with a clean, opinionated API consistent
 * with the Next Auth variant in this boilerplate.
 */
"use client";

import { useAuth as useClerkAuth } from "@clerk/nextjs";
import { useCallback } from "react";

export type AuthStatus = "loading" | "authenticated" | "unauthenticated";

export type UseAuthReturn = {
  status: AuthStatus;
  isLoading: boolean;
  isAuthenticated: boolean;
  isUnauthenticated: boolean;
  userId: string | null;
  /**
   * Get the current session token.
   * Clerk automatically refreshes it when needed.
   */
  getToken: () => Promise<string | null>;
};

export function useAuth(): UseAuthReturn {
  const { isLoaded, isSignedIn, userId, getToken } = useClerkAuth();

  const status: AuthStatus = !isLoaded
    ? "loading"
    : isSignedIn
      ? "authenticated"
      : "unauthenticated";

  const handleGetToken = useCallback(async () => {
    if (!isSignedIn) return null;
    return getToken();
  }, [isSignedIn, getToken]);

  return {
    status,
    isLoading: !isLoaded,
    isAuthenticated: isLoaded && !!isSignedIn,
    isUnauthenticated: isLoaded && !isSignedIn,
    userId: userId ?? null,
    getToken: handleGetToken,
  };
}
