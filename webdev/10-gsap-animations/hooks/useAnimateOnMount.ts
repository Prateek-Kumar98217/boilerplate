/**
 * useAnimateOnMount — run a preset animation when a component mounts.
 *
 * ```tsx
 * function Toast({ message }: { message: string }) {
 *   const ref = useAnimateOnMount<HTMLDivElement>("fade-up", {
 *     duration: 0.5,
 *     delay: 0.1,
 *   });
 *
 *   return <div ref={ref} className="toast">{message}</div>;
 * }
 * ```
 */

import { useRef, useEffect, useLayoutEffect } from "react";
import { gsap } from "gsap";
import {
  ANIMATION_CONFIG,
  VARIANT_FROM,
  resolveEase,
  prefersReducedMotion,
} from "../core/config";
import type { AnimationVariant } from "../types";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export interface UseAnimateOnMountOptions {
  /** Duration in seconds */
  duration?: number;
  /** Delay in seconds */
  delay?: number;
  /** Ease (preset key or GSAP string) */
  ease?: string;
  /** Callback when animation completes */
  onComplete?: () => void;
}

/**
 * @param variant - Animation variant name (e.g. "fade-up", "scale", "blur").
 * @param options - Duration, delay, ease overrides.
 * @returns       - Ref to attach to the animated element.
 */
export function useAnimateOnMount<T extends HTMLElement = HTMLDivElement>(
  variant: AnimationVariant = "fade-up",
  options: UseAnimateOnMountOptions = {},
): React.RefObject<T | null> {
  const ref = useRef<T | null>(null);

  useIsomorphicLayoutEffect(() => {
    if (!ref.current) return;
    if (prefersReducedMotion()) return;

    const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM.fade;

    const ctx = gsap.context(() => {
      gsap.from(ref.current!, {
        ...fromVars,
        duration: options.duration ?? ANIMATION_CONFIG.defaultDuration,
        delay: options.delay ?? 0,
        ease: resolveEase(options.ease),
        onComplete: options.onComplete,
      });
    });

    return () => ctx.revert();
  }, []);

  return ref;
}
