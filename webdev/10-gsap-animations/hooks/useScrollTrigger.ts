/**
 * useScrollTrigger — declarative ScrollTrigger hook.
 *
 * Auto-creates a ScrollTrigger-driven animation scoped to a container
 * ref with full cleanup on unmount.
 *
 * ```tsx
 * function Section() {
 *   const ref = useRef<HTMLDivElement>(null);
 *
 *   useScrollTrigger(ref, {
 *     animation: (el) => gsap.from(el.querySelectorAll(".item"), {
 *       y: 60, opacity: 0, stagger: 0.1,
 *     }),
 *     start: "top 75%",
 *     once: true,
 *   });
 *
 *   return (
 *     <section ref={ref}>
 *       <div className="item">A</div>
 *       <div className="item">B</div>
 *     </section>
 *   );
 * }
 * ```
 */

import { useEffect, useLayoutEffect } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { ANIMATION_CONFIG, prefersReducedMotion } from "../core/config";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export interface UseScrollTriggerOptions {
  /** Build the animation — receives the trigger element */
  animation: (trigger: HTMLElement) => gsap.core.Animation | void;
  /** ScrollTrigger start position (default: "top 80%") */
  start?: string;
  /** ScrollTrigger end position */
  end?: string;
  /** Scrub value */
  scrub?: boolean | number;
  /** Pin the trigger element */
  pin?: boolean;
  /** Snap progress values */
  snap?: number | number[] | gsap.SnapVars;
  /** Only play once, then kill the trigger */
  once?: boolean;
  /** Show debug markers */
  markers?: boolean;
  /** Callbacks */
  onEnter?: () => void;
  onLeave?: () => void;
  onEnterBack?: () => void;
  onLeaveBack?: () => void;
}

export function useScrollTrigger(
  ref: React.RefObject<HTMLElement | null>,
  options: UseScrollTriggerOptions,
  deps: React.DependencyList = [],
): void {
  useIsomorphicLayoutEffect(() => {
    if (!ref.current) return;
    if (prefersReducedMotion()) return;

    const trigger = ref.current;

    const ctx = gsap.context(() => {
      const anim = options.animation(trigger);

      ScrollTrigger.create({
        trigger,
        animation: anim || undefined,
        start: options.start ?? ANIMATION_CONFIG.scrollStart,
        end: options.end ?? ANIMATION_CONFIG.scrollEnd,
        scrub: options.scrub ?? false,
        pin: options.pin ?? false,
        snap: options.snap,
        markers: options.markers ?? ANIMATION_CONFIG.markers,
        once: options.once ?? true,
        onEnter: options.onEnter,
        onLeave: options.onLeave,
        onEnterBack: options.onEnterBack,
        onLeaveBack: options.onLeaveBack,
      });
    }, trigger);

    return () => ctx.revert();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ref, ...deps]);
}
