/**
 * MagneticButton — magnetic hover button component.
 *
 * Wraps any child element and applies a magnetic cursor-attraction
 * effect on hover.
 *
 * ```tsx
 * <MagneticButton strength={0.4}>
 *   <button className="btn-primary">Hover me</button>
 * </MagneticButton>
 *
 * <MagneticButton strength={0.3} rotate maxRotation={15}>
 *   <a href="/about" className="nav-link">About</a>
 * </MagneticButton>
 * ```
 */

"use client";

import React, { useRef, useEffect } from "react";
import { applyMagnetic } from "../patterns/magnetic";
import { prefersReducedMotion } from "../core/config";
import type { MagneticOptions } from "../types";

export interface MagneticButtonProps extends MagneticOptions {
  /** Custom className for the wrapper */
  className?: string;
  /** Wrapper element tag (default: "div") */
  as?: React.ElementType;
  children: React.ReactNode;
}

export function MagneticButton({
  strength,
  ease,
  duration,
  rotate,
  maxRotation,
  className,
  as: Tag = "div",
  children,
}: MagneticButtonProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current || prefersReducedMotion()) return;

    const cleanup = applyMagnetic(ref.current, {
      strength,
      ease,
      duration,
      rotate,
      maxRotation,
    });

    return cleanup;
  }, [strength, ease, duration, rotate, maxRotation]);

  return (
    <Tag ref={ref} className={className} style={{ display: "inline-block" }}>
      {children}
    </Tag>
  );
}
