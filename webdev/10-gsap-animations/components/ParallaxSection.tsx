/**
 * ParallaxSection — parallax depth section component.
 *
 * Wraps child layers and applies depth-based parallax scrolling.
 * Each direct child can have a `data-depth` attribute to control
 * its parallax speed.
 *
 * ```tsx
 * <ParallaxSection className="hero">
 *   <img data-depth="0.2" src="/bg.jpg" className="absolute inset-0" />
 *   <div data-depth="0.5" className="relative z-10">
 *     <h1>Parallax Hero</h1>
 *   </div>
 *   <img data-depth="0.8" src="/fg.png" className="absolute bottom-0" />
 * </ParallaxSection>
 * ```
 */

"use client";

import React, { useRef, useEffect, useLayoutEffect } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { ANIMATION_CONFIG, prefersReducedMotion } from "../core/config";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export interface ParallaxSectionProps {
  /** Custom className for the section */
  className?: string;
  /** Default depth for children without data-depth (default: 0.5) */
  defaultDepth?: number;
  /** Parallax base distance in pixels (default: 200) */
  distance?: number;
  /** Section height — defaults to CSS, but useful for overriding */
  style?: React.CSSProperties;
  children: React.ReactNode;
}

export function ParallaxSection({
  className,
  defaultDepth = 0.5,
  distance = 200,
  style,
  children,
}: ParallaxSectionProps) {
  const sectionRef = useRef<HTMLElement>(null);

  useIsomorphicLayoutEffect(() => {
    if (!sectionRef.current || prefersReducedMotion()) return;

    const section = sectionRef.current;
    const layers = section.querySelectorAll<HTMLElement>("[data-depth]");

    const ctx = gsap.context(() => {
      layers.forEach((layer) => {
        const depth = parseFloat(layer.dataset.depth ?? String(defaultDepth));

        gsap.to(layer, {
          y: -(depth * distance),
          ease: "none",
          scrollTrigger: {
            trigger: section,
            start: "top bottom",
            end: "bottom top",
            scrub: true,
            markers: ANIMATION_CONFIG.markers,
          },
        });
      });
    }, section);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      className={className}
      style={{ position: "relative", overflow: "hidden", ...style }}
    >
      {children}
    </section>
  );
}
