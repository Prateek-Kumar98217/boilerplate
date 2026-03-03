/**
 * AnimatePresence — mount / unmount animation wrapper.
 *
 * Animates children in on mount and out on unmount.
 * Similar concept to Framer Motion's AnimatePresence but using GSAP.
 *
 * ```tsx
 * {isVisible && (
 *   <AnimatePresence variant="fade-up" duration={0.5}>
 *     <Modal />
 *   </AnimatePresence>
 * )}
 * ```
 */

"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { gsap } from "gsap";
import {
  ANIMATION_CONFIG,
  VARIANT_FROM,
  resolveEase,
  prefersReducedMotion,
} from "../core/config";
import type { AnimationVariant } from "../types";

export interface AnimatePresenceProps {
  /** Animation variant */
  variant?: AnimationVariant;
  /** Duration for enter animation */
  enterDuration?: number;
  /** Duration for exit animation */
  exitDuration?: number;
  /** Ease for enter */
  enterEase?: string;
  /** Ease for exit */
  exitEase?: string;
  /** Delay before enter animation */
  delay?: number;
  /** Called when enter animation completes */
  onEnterComplete?: () => void;
  /** Called when exit animation completes */
  onExitComplete?: () => void;
  /** Custom className for wrapper div */
  className?: string;
  children: React.ReactNode;
}

export function AnimatePresence({
  variant = "fade-up",
  enterDuration,
  exitDuration,
  enterEase,
  exitEase,
  delay = 0,
  onEnterComplete,
  onExitComplete,
  className,
  children,
}: AnimatePresenceProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current || prefersReducedMotion()) return;

    const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM.fade;

    const ctx = gsap.context(() => {
      gsap.from(ref.current!, {
        ...fromVars,
        duration: enterDuration ?? ANIMATION_CONFIG.defaultDuration,
        delay,
        ease: resolveEase(enterEase),
        onComplete: onEnterComplete,
      });
    });

    return () => ctx.revert();
  }, []);

  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  );
}

// ─── With exit animation support ───────────────────────────────────────────────

export interface AnimatePresenceControlledProps extends AnimatePresenceProps {
  /** Controls visibility — animates out before unmounting */
  show: boolean;
}

/**
 * Controlled version that animates out before removing from DOM.
 *
 * ```tsx
 * <AnimatePresenceControlled show={isOpen} variant="scale">
 *   <Dropdown />
 * </AnimatePresenceControlled>
 * ```
 */
export function AnimatePresenceControlled({
  show,
  variant = "fade-up",
  enterDuration,
  exitDuration,
  enterEase,
  exitEase,
  delay = 0,
  onEnterComplete,
  onExitComplete,
  className,
  children,
}: AnimatePresenceControlledProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [shouldRender, setShouldRender] = useState(show);

  useEffect(() => {
    if (show) {
      setShouldRender(true);
    }
  }, [show]);

  useEffect(() => {
    if (!ref.current || prefersReducedMotion()) return;

    const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM.fade;

    if (show) {
      // Enter
      gsap.from(ref.current, {
        ...fromVars,
        duration: enterDuration ?? ANIMATION_CONFIG.defaultDuration,
        delay,
        ease: resolveEase(enterEase),
        onComplete: onEnterComplete,
      });
    } else if (shouldRender) {
      // Exit
      gsap.to(ref.current, {
        ...fromVars,
        duration: exitDuration ?? ANIMATION_CONFIG.defaultDuration * 0.7,
        ease: resolveEase(exitEase ?? "smooth"),
        onComplete: () => {
          setShouldRender(false);
          onExitComplete?.();
        },
      });
    }
  }, [show]);

  if (!shouldRender) return null;

  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  );
}
