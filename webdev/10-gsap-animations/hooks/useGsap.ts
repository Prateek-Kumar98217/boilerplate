/**
 * useGsap — base GSAP hook with automatic cleanup.
 *
 * Wraps `gsap.context()` scoped to a container ref so that all GSAP
 * calls inside the callback are automatically reverted on unmount.
 *
 * ```tsx
 * function Hero() {
 *   const container = useRef<HTMLDivElement>(null);
 *
 *   useGsap(container, () => {
 *     gsap.from(".title", { y: 60, opacity: 0 });
 *   });
 *
 *   return <div ref={container}><h1 className="title">Hi</h1></div>;
 * }
 * ```
 */

import { useRef, useEffect, useLayoutEffect } from "react";
import { gsap } from "gsap";
import { prefersReducedMotion } from "../core/config";

/** Use useLayoutEffect on the client, useEffect on the server (SSR-safe). */
const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export type GsapCallback = (
  /** The gsap.Context instance */
  ctx: gsap.Context,
  /** Wrap event handlers with this to add them to the context for cleanup */
  contextSafe: <T extends (...args: any[]) => any>(fn: T) => T,
) => void;

/**
 * @param scope   - Ref to the container element — GSAP selectors inside
 *                  the callback are scoped to this element.
 * @param callback - Animation setup function. Receives `ctx` and `contextSafe`.
 * @param deps     - Dependency array (like useEffect). Default: `[]`.
 */
export function useGsap(
  scope: React.RefObject<HTMLElement | null>,
  callback: GsapCallback,
  deps: React.DependencyList = [],
): void {
  // Keep callback ref stable to avoid re-running on every render
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useIsomorphicLayoutEffect(() => {
    if (!scope.current) return;

    // Skip animations when user prefers reduced motion
    if (prefersReducedMotion()) return;

    const ctx = gsap.context((self) => {
      callbackRef.current(ctx, self.add.bind(self) as any);
    }, scope.current);

    return () => ctx.revert();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope, ...deps]);
}
