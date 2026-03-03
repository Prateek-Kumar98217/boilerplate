/**
 * ScrollReveal — scroll-triggered reveal wrapper.
 *
 * Wraps children and reveals them with a configurable animation
 * when scrolled into view.
 *
 * ```tsx
 * <ScrollReveal variant="fade-up" stagger={0.08} duration={0.6}>
 *   <Card />
 *   <Card />
 *   <Card />
 * </ScrollReveal>
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
import type { ScrollRevealProps } from "../types";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export function ScrollReveal({
  variant = "fade-up",
  duration,
  stagger,
  delay = 0,
  start,
  once = true,
  className,
  children,
}: ScrollRevealProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useIsomorphicLayoutEffect(() => {
    if (!containerRef.current || prefersReducedMotion()) return;

    const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM["fade-up"];
    const targets = containerRef.current.children;

    if (targets.length === 0) return;

    const ctx = gsap.context(() => {
      gsap.from(targets, {
        ...fromVars,
        duration: duration ?? ANIMATION_CONFIG.defaultDuration,
        delay,
        stagger:
          stagger ?? (targets.length > 1 ? ANIMATION_CONFIG.defaultStagger : 0),
        ease: resolveEase(),
        scrollTrigger: {
          trigger: containerRef.current,
          start: start ?? ANIMATION_CONFIG.scrollStart,
          once,
          markers: ANIMATION_CONFIG.markers,
        },
      });
    }, containerRef.current);

    return () => ctx.revert();
  }, []);

  return (
    <div ref={containerRef} className={className}>
      {children}
    </div>
  );
}
