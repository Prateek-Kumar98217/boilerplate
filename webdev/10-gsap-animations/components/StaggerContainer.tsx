/**
 * StaggerContainer — stagger-animate children on mount or scroll.
 *
 * Each direct child animates in with a stagger delay.
 *
 * ```tsx
 * <StaggerContainer variant="fade-up" stagger={0.08} trigger="scroll">
 *   <Card title="one" />
 *   <Card title="two" />
 *   <Card title="three" />
 * </StaggerContainer>
 * ```
 */

"use client";

import React, { useRef, useEffect, useLayoutEffect } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import {
  ANIMATION_CONFIG,
  VARIANT_FROM,
  resolveEase,
  prefersReducedMotion,
} from "../core/config";
import type { AnimationVariant, StaggerOrigin } from "../types";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export interface StaggerContainerProps {
  /** Animation variant */
  variant?: AnimationVariant;
  /** Stagger amount between children (seconds) */
  stagger?: number;
  /** Stagger origin */
  from?: StaggerOrigin;
  /** Duration per child */
  duration?: number;
  /** Ease */
  ease?: string;
  /** Trigger on mount or scroll */
  trigger?: "mount" | "scroll";
  /** ScrollTrigger start position */
  scrollStart?: string;
  /** Only animate once */
  once?: boolean;
  /** Delay before first child starts */
  delay?: number;
  /** Custom className */
  className?: string;
  children: React.ReactNode;
}

export function StaggerContainer({
  variant = "fade-up",
  stagger: staggerProp,
  from = "start",
  duration,
  ease,
  trigger = "mount",
  scrollStart,
  once = true,
  delay = 0,
  className,
  children,
}: StaggerContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useIsomorphicLayoutEffect(() => {
    if (!containerRef.current || prefersReducedMotion()) return;

    const targets = containerRef.current.children;
    if (targets.length === 0) return;

    const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM["fade-up"];
    const stagger = staggerProp ?? ANIMATION_CONFIG.defaultStagger;

    const ctx = gsap.context(() => {
      const tweenVars: gsap.TweenVars = {
        ...fromVars,
        duration: duration ?? ANIMATION_CONFIG.defaultDuration,
        delay,
        stagger: {
          each: stagger,
          from,
        },
        ease: resolveEase(ease),
      };

      if (trigger === "scroll") {
        tweenVars.scrollTrigger = {
          trigger: containerRef.current,
          start: scrollStart ?? ANIMATION_CONFIG.scrollStart,
          once,
        };
      }

      gsap.from(targets, tweenVars);
    }, containerRef.current);

    return () => ctx.revert();
  }, []);

  return (
    <div ref={containerRef} className={className}>
      {children}
    </div>
  );
}
