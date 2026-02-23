/**
 * UpgradeToast — Client Component that reads the `reason` search param
 * and fires a toast notification on mount.
 *
 * Place this in your /upgrade page or its layout so it's always present
 * when the middleware redirects to /upgrade?reason=...
 *
 * Compatible with: sonner (recommended), react-hot-toast, or shadcn/ui toast.
 * This example uses `sonner` — swap the `toast` call for your library.
 *
 * Install sonner: npm install sonner
 * Add <Toaster /> to your root layout.
 */
"use client";

import { useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { toast } from "sonner";
import type { UpgradeReason } from "../middleware";

// ─── Human-readable messages ──────────────────────────────────────────────────

const REASON_MESSAGES: Record<UpgradeReason, string> = {
  plan_required: "This feature requires a higher plan. Upgrade to unlock it.",
  subscription_expired:
    "Your subscription has expired. Renew to regain access.",
  subscription_past_due:
    "Your payment is past due. Please update your billing details.",
  subscription_inactive:
    "You don't have an active subscription. Upgrade to get started.",
};

// ─── Component ────────────────────────────────────────────────────────────────

export function UpgradeToast() {
  const searchParams = useSearchParams();
  const reason = searchParams.get("reason") as UpgradeReason | null;
  const from = searchParams.get("from");

  useEffect(() => {
    if (!reason) return;

    const message =
      REASON_MESSAGES[reason] ??
      "You need to upgrade your plan to access this feature.";

    const description = from ? `You tried to access: ${from}` : undefined;

    toast.warning(message, {
      description,
      duration: 6000,
      id: "subscription-guard", // prevent duplicate toasts on re-render
    });
  }, [reason, from]);

  return null; // renders nothing — side-effect only
}
