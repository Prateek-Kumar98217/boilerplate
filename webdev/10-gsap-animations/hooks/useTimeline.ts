/**
 * useTimeline — managed GSAP timeline with proper React lifecycle.
 *
 * Returns a timeline ref that is auto-created / auto-killed.
 * Rebuild the timeline when dependencies change.
 *
 * ```tsx
 * function Card() {
 *   const container = useRef<HTMLDivElement>(null);
 *   const tl = useTimeline(container, (timeline) => {
 *     timeline
 *       .from(".card-img", { scale: 0.8, opacity: 0, duration: 0.5 })
 *       .from(".card-body", { y: 30, opacity: 0 }, "-=0.2");
 *   });
 *
 *   const handleHover = () => tl.current?.restart();
 *
 *   return <div ref={container} onMouseEnter={handleHover}>...</div>;
 * }
 * ```
 */

import { useRef, useEffect, useLayoutEffect } from "react";
import { gsap } from "gsap";
import { ANIMATION_CONFIG, prefersReducedMotion } from "../core/config";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export interface UseTimelineOptions {
  /** Whether the timeline starts paused (default: false) */
  paused?: boolean;
  /** Default tween vars for every child tween */
  defaults?: gsap.TweenVars;
  /** Repeat count (-1 = infinite) */
  repeat?: number;
  /** Yoyo on repeat */
  yoyo?: boolean;
}

export type TimelineCallback = (tl: gsap.core.Timeline) => void;

/**
 * @param scope     - Container ref for scoped selectors.
 * @param callback  - Receives a fresh timeline — add tweens here.
 * @param deps      - Dependency array. Default: `[]`.
 * @param options   - Timeline creation options.
 * @returns         - Ref to the current gsap.core.Timeline (use `.current`).
 */
export function useTimeline(
  scope: React.RefObject<HTMLElement | null>,
  callback: TimelineCallback,
  deps: React.DependencyList = [],
  options: UseTimelineOptions = {},
): React.RefObject<gsap.core.Timeline | null> {
  const tlRef = useRef<gsap.core.Timeline | null>(null);
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useIsomorphicLayoutEffect(() => {
    if (!scope.current) return;
    if (prefersReducedMotion()) return;

    const ctx = gsap.context(() => {
      const tl = gsap.timeline({
        paused: options.paused ?? false,
        repeat: options.repeat,
        yoyo: options.yoyo,
        defaults: {
          duration: ANIMATION_CONFIG.defaultDuration,
          ease: ANIMATION_CONFIG.defaultEase,
          ...options.defaults,
        },
      });

      callbackRef.current(tl);

      tlRef.current = tl;
    }, scope.current);

    return () => {
      ctx.revert();
      tlRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope, ...deps]);

  return tlRef;
}
