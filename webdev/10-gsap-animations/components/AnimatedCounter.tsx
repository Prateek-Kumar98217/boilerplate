/**
 * AnimatedCounter — counting number animation component.
 *
 * Animates a number from one value to another with optional
 * formatting (separators, prefix, suffix, decimals).
 *
 * ```tsx
 * <AnimatedCounter from={0} to={12500} separator="," prefix="$" />
 * <AnimatedCounter to={99.9} decimals={1} suffix="%" duration={1.5} />
 * <AnimatedCounter to={5000} scrollTrigger />
 * ```
 */

"use client";

import React, { useRef, useEffect, useLayoutEffect, useState } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { ANIMATION_CONFIG, prefersReducedMotion } from "../core/config";
import type { AnimatedCounterProps } from "../types";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

function formatNumber(
  value: number,
  decimals: number,
  separator: string,
): string {
  const fixed = value.toFixed(decimals);
  if (!separator) return fixed;

  const [intPart, decPart] = fixed.split(".");
  const formatted = intPart.replace(/\B(?=(\d{3})+(?!\d))/g, separator);
  return decPart ? `${formatted}.${decPart}` : formatted;
}

export function AnimatedCounter({
  from = 0,
  to,
  duration = 2,
  decimals = 0,
  separator = "",
  prefix = "",
  suffix = "",
  scrollTrigger = false,
  className,
}: AnimatedCounterProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const [display, setDisplay] = useState(
    `${prefix}${formatNumber(from, decimals, separator)}${suffix}`,
  );

  useIsomorphicLayoutEffect(() => {
    if (!ref.current) return;

    if (prefersReducedMotion()) {
      setDisplay(`${prefix}${formatNumber(to, decimals, separator)}${suffix}`);
      return;
    }

    const obj = { value: from };

    const ctx = gsap.context(() => {
      gsap.to(obj, {
        value: to,
        duration,
        ease: "power1.out",
        scrollTrigger: scrollTrigger
          ? {
              trigger: ref.current,
              start: ANIMATION_CONFIG.scrollStart,
              once: true,
            }
          : undefined,
        onUpdate: () => {
          setDisplay(
            `${prefix}${formatNumber(obj.value, decimals, separator)}${suffix}`,
          );
        },
      });
    });

    return () => ctx.revert();
  }, [from, to]);

  return (
    <span ref={ref} className={className}>
      {display}
    </span>
  );
}
