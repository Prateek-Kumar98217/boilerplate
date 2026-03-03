/**
 * MarqueeText — infinite horizontal scrolling text component.
 *
 * Seamlessly loops text content in a continuous ticker.
 * Optionally pauses on hover.
 *
 * ```tsx
 * <MarqueeText speed={100} pauseOnHover gap={48}>
 *   <span>BREAKING NEWS — </span>
 *   <span>GSAP IS AWESOME — </span>
 *   <span>SCROLL FOR MORE — </span>
 * </MarqueeText>
 * ```
 */

"use client";

import React, { useRef, useEffect, useLayoutEffect } from "react";
import { gsap } from "gsap";
import { prefersReducedMotion } from "../core/config";
import type { MarqueeProps } from "../types";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export function MarqueeText({
  speed = 80,
  direction = "left",
  pauseOnHover = false,
  gap = 32,
  className,
  children,
}: MarqueeProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);

  useIsomorphicLayoutEffect(() => {
    if (!containerRef.current || !innerRef.current || prefersReducedMotion())
      return;

    const container = containerRef.current;
    const inner = innerRef.current;

    // Clone content to create seamless loop
    const clone = inner.cloneNode(true) as HTMLDivElement;
    clone.setAttribute("aria-hidden", "true");
    container.appendChild(clone);

    const totalWidth = inner.offsetWidth + gap;
    const duration = totalWidth / speed;
    const xDir = direction === "left" ? -totalWidth : totalWidth;

    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ repeat: -1 });

      tl.fromTo(
        container.children,
        { x: direction === "left" ? 0 : -totalWidth },
        {
          x: xDir,
          duration,
          ease: "none",
        },
      );

      // Pause on hover
      if (pauseOnHover) {
        const pause = () => tl.pause();
        const play = () => tl.play();
        container.addEventListener("mouseenter", pause);
        container.addEventListener("mouseleave", play);
      }
    }, container);

    return () => {
      ctx.revert();
      // Remove clone
      if (clone.parentNode) clone.remove();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        overflow: "hidden",
        whiteSpace: "nowrap",
        display: "flex",
        gap: `${gap}px`,
      }}
    >
      <div
        ref={innerRef}
        style={{ display: "flex", gap: `${gap}px`, flexShrink: 0 }}
      >
        {children}
      </div>
    </div>
  );
}
