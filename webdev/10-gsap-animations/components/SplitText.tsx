/**
 * SplitText — character / word / line split reveal component.
 *
 * Wraps text content and animates each piece individually.
 *
 * ```tsx
 * <SplitText type="chars" stagger={0.03} trigger="scroll">
 *   Every character slides in
 * </SplitText>
 *
 * <SplitText as="h1" type="words" variant="fade-up" stagger={0.06}>
 *   Word by word reveal on mount
 * </SplitText>
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
import type { SplitType, AnimationVariant } from "../types";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export interface SplitTextProps {
  /** HTML tag to render (default: "div") */
  as?: React.ElementType;
  /** Split mode */
  type?: SplitType;
  /** Animation variant for each split piece */
  variant?: AnimationVariant;
  /** Stagger between pieces */
  stagger?: number;
  /** Duration */
  duration?: number;
  /** Ease */
  ease?: string;
  /** Trigger: animate on mount or scroll */
  trigger?: "mount" | "scroll";
  /** ScrollTrigger start position */
  scrollStart?: string;
  /** Delay */
  delay?: number;
  /** ClassName for the wrapper */
  className?: string;
  children: string;
}

export function SplitText({
  as: Tag = "div",
  type = "words",
  variant = "fade-up",
  stagger: staggerProp,
  duration,
  ease,
  trigger = "mount",
  scrollStart,
  delay = 0,
  className,
  children,
}: SplitTextProps) {
  const ref = useRef<HTMLElement>(null);

  useIsomorphicLayoutEffect(() => {
    if (!ref.current || prefersReducedMotion()) return;

    const el = ref.current;
    const text = el.textContent ?? "";
    const originalHTML = el.innerHTML;

    // ── Split text into spans ──
    el.setAttribute("aria-label", text);
    el.innerHTML = "";

    const spans: HTMLSpanElement[] = [];
    const splitMode = type.includes("chars") ? "chars" : "words";
    const pieces = splitMode === "chars" ? text.split("") : text.split(/\s+/);

    pieces.forEach((piece, i) => {
      if (!piece) return;
      const span = document.createElement("span");
      span.style.display = "inline-block";
      span.textContent = piece === " " ? "\u00A0" : piece;
      span.setAttribute("aria-hidden", "true");
      span.classList.add(`split-${splitMode.slice(0, -1)}`);
      el.appendChild(span);
      spans.push(span);

      if (splitMode === "words" && i < pieces.length - 1) {
        el.appendChild(document.createTextNode("\u00A0"));
      }
    });

    // ── Animate ──
    const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM["fade-up"];
    const stagger = staggerProp ?? (splitMode === "chars" ? 0.03 : 0.06);

    const ctx = gsap.context(() => {
      const tweenVars: gsap.TweenVars = {
        ...fromVars,
        stagger,
        duration: duration ?? ANIMATION_CONFIG.defaultDuration * 0.8,
        delay,
        ease: resolveEase(ease),
      };

      if (trigger === "scroll") {
        tweenVars.scrollTrigger = {
          trigger: el,
          start: scrollStart ?? ANIMATION_CONFIG.scrollStart,
          once: true,
        };
      }

      gsap.from(spans, tweenVars);
    });

    return () => {
      ctx.revert();
      el.innerHTML = originalHTML;
    };
  }, []);

  return (
    <Tag ref={ref} className={className}>
      {children}
    </Tag>
  );
}
