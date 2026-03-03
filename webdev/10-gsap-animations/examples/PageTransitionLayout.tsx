/**
 * PageTransitionLayout — app-level route transition wrapper.
 *
 * Wraps the main content area and animates page transitions
 * when the route (pathname) changes.
 *
 * Designed for Next.js App Router but adaptable to any framework.
 *
 * ```tsx
 * // app/layout.tsx
 * import { PageTransitionLayout } from "@/lib/gsap/examples/PageTransitionLayout";
 *
 * export default function RootLayout({ children }: { children: React.ReactNode }) {
 *   return (
 *     <html lang="en">
 *       <body>
 *         <PageTransitionLayout variant="slide">
 *           {children}
 *         </PageTransitionLayout>
 *       </body>
 *     </html>
 *   );
 * }
 * ```
 */

"use client";

import React, { useRef, useCallback, useEffect, useState } from "react";
import { gsap } from "gsap";
import { EASINGS, resolveEase } from "../core/config";
import type { TransitionVariant } from "../types";

// ─── Transition preset map ────────────────────────────────────────────────────

const PRESETS: Record<
  TransitionVariant,
  { enter: gsap.TweenVars; exit: gsap.TweenVars }
> = {
  fade: {
    enter: { opacity: 0 },
    exit: { opacity: 0 },
  },
  slide: {
    enter: { x: 80, opacity: 0 },
    exit: { x: -40, opacity: 0 },
  },
  "slide-up": {
    enter: { y: 60, opacity: 0 },
    exit: { y: -30, opacity: 0 },
  },
  "slide-down": {
    enter: { y: -60, opacity: 0 },
    exit: { y: 30, opacity: 0 },
  },
  scale: {
    enter: { scale: 0.95, opacity: 0 },
    exit: { scale: 1.03, opacity: 0 },
  },
  curtain: {
    enter: { clipPath: "inset(0 0 100% 0)" },
    exit: { clipPath: "inset(100% 0 0 0)" },
  },
  wipe: {
    enter: { clipPath: "inset(0 100% 0 0)" },
    exit: { clipPath: "inset(0 0 0 100%)" },
  },
};

// ─── Component ─────────────────────────────────────────────────────────────────

export interface PageTransitionLayoutProps {
  /** Transition variant */
  variant?: TransitionVariant;
  /** Duration for each half (enter / exit) in seconds */
  duration?: number;
  /** Children — the routed page content */
  children: React.ReactNode;
}

/**
 * NOTE: In a real Next.js App Router setup, detecting route changes
 * requires either:
 *   1. next/navigation `usePathname()` hook
 *   2. A global event system
 *
 * This example shows the animation pattern. Integrate with your router's
 * lifecycle hooks.
 */
export function PageTransitionLayout({
  variant = "fade",
  duration = 0.5,
  children,
}: PageTransitionLayoutProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [displayChildren, setDisplayChildren] = useState(children);
  const preset = PRESETS[variant] ?? PRESETS.fade;

  // Animate enter on mount
  useEffect(() => {
    if (!containerRef.current) return;

    gsap.from(containerRef.current, {
      ...preset.enter,
      duration,
      ease: EASINGS.expo,
      clearProps: "all",
    });
  }, []);

  // When children change (route change), animate out → swap → animate in
  useEffect(() => {
    if (children === displayChildren) return;
    if (!containerRef.current) {
      setDisplayChildren(children);
      return;
    }

    const el = containerRef.current;

    // Exit animation
    gsap.to(el, {
      ...preset.exit,
      duration: duration * 0.7,
      ease: EASINGS.smooth,
      onComplete: () => {
        // Swap content
        setDisplayChildren(children);

        // Enter animation
        gsap.from(el, {
          ...preset.enter,
          duration,
          ease: EASINGS.expo,
          clearProps: "all",
        });

        // Scroll to top
        window.scrollTo({ top: 0 });
      },
    });
  }, [children]);

  return (
    <div
      ref={containerRef}
      style={{ minHeight: "100vh", position: "relative" }}
    >
      {displayChildren}
    </div>
  );
}
