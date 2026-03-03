/**
 * Animation cleanup utilities.
 *
 * GSAP `gsap.context()` handles most cleanup automatically when used
 * with the `useGsap` hook. These helpers cover edge cases.
 */

import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

/**
 * Kill all GSAP tweens targeting the given element(s).
 */
export function killTweens(targets: gsap.TweenTarget): void {
  gsap.killTweensOf(targets);
}

/**
 * Kill all ScrollTrigger instances associated with a trigger element.
 */
export function killScrollTriggers(trigger: string | Element): void {
  ScrollTrigger.getAll()
    .filter((st) => st.trigger === trigger || st.vars.trigger === trigger)
    .forEach((st) => st.kill());
}

/**
 * Kill every ScrollTrigger in the page. Useful during route transitions.
 */
export function killAllScrollTriggers(): void {
  ScrollTrigger.getAll().forEach((st) => st.kill());
}

/**
 * Revert a gsap.context — restores inline styles set by GSAP.
 * Typically called inside a useEffect cleanup.
 */
export function revertContext(ctx: gsap.Context): void {
  ctx.revert();
}

/**
 * Refresh all ScrollTriggers.
 * Call after dynamic content changes that affect page height.
 */
export function refreshScrollTriggers(): void {
  ScrollTrigger.refresh();
}

/**
 * Clear all inline styles GSAP set on target elements.
 */
export function clearProps(targets: gsap.TweenTarget, props = "all"): void {
  gsap.set(targets, { clearProps: props });
}
