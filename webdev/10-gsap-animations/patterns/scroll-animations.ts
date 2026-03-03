/**
 * Scroll-driven animation patterns.
 *
 * Factory functions that return GSAP tweens / timelines wired to
 * ScrollTrigger. Use inside a `useGsap` callback or imperatively.
 *
 * ```ts
 * useGsap(container, () => {
 *   createScrollFadeIn(".card", { stagger: 0.1 });
 *   createScrollParallax(".bg-image", { speed: 0.3 });
 *   createPinSection(".hero", { endOffset: "+=200%" });
 * });
 * ```
 */

import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { ANIMATION_CONFIG, VARIANT_FROM, resolveEase } from "../core/config";
import type { AnimationVariant, ScrollTriggerOptions } from "../types";

// ─── Scroll fade-in ────────────────────────────────────────────────────────────

export interface ScrollFadeInOptions {
  variant?: AnimationVariant;
  duration?: number;
  stagger?: number;
  ease?: string;
  start?: string;
  end?: string;
  scrub?: boolean | number;
  once?: boolean;
  markers?: boolean;
}

/**
 * Animate elements into view as the user scrolls.
 */
export function createScrollFadeIn(
  targets: gsap.TweenTarget,
  options: ScrollFadeInOptions = {},
): gsap.core.Tween {
  const variant = options.variant ?? "fade-up";
  const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM["fade-up"];

  return gsap.from(targets, {
    ...fromVars,
    duration: options.duration ?? ANIMATION_CONFIG.defaultDuration,
    stagger: options.stagger ?? ANIMATION_CONFIG.defaultStagger,
    ease: resolveEase(options.ease),
    scrollTrigger: {
      trigger: targets as gsap.DOMTarget,
      start: options.start ?? ANIMATION_CONFIG.scrollStart,
      end: options.end,
      scrub: options.scrub ?? false,
      once: options.once ?? true,
      markers: options.markers ?? ANIMATION_CONFIG.markers,
    },
  });
}

// ─── Scroll progress bar ──────────────────────────────────────────────────────

/**
 * Animate a progress bar (e.g. reading indicator) that fills
 * based on scroll position.
 */
export function createScrollProgressBar(
  target: gsap.TweenTarget,
  options: { scaleAxis?: "x" | "y" } = {},
): gsap.core.Tween {
  const axis = options.scaleAxis ?? "x";
  const prop = axis === "x" ? "scaleX" : "scaleY";

  return gsap.to(target, {
    [prop]: 1,
    ease: "none",
    scrollTrigger: {
      trigger: document.documentElement,
      start: "top top",
      end: "bottom bottom",
      scrub: 0.3,
    },
  });
}

// ─── Pin section ───────────────────────────────────────────────────────────────

export interface PinSectionOptions {
  /** How far past the section to keep it pinned (default: "+=100%") */
  endOffset?: string;
  /** Inner animation to play while pinned */
  animation?: gsap.core.Animation;
  scrub?: boolean | number;
  snap?: number | number[];
  markers?: boolean;
}

/**
 * Pin a section while scrolling through it, optionally scrubbing
 * an animation during the pinned scroll distance.
 */
export function createPinSection(
  trigger: gsap.DOMTarget,
  options: PinSectionOptions = {},
): ScrollTrigger {
  return ScrollTrigger.create({
    trigger,
    start: "top top",
    end: options.endOffset ?? "+=100%",
    pin: true,
    scrub: options.scrub ?? 1,
    snap: options.snap,
    animation: options.animation,
    markers: options.markers ?? ANIMATION_CONFIG.markers,
  });
}

// ─── Horizontal scroll section ─────────────────────────────────────────────────

export interface HorizontalScrollOptions {
  /** The horizontal scrolling container */
  container: gsap.DOMTarget;
  /** Panel selector inside the container */
  panelSelector?: string;
  ease?: string;
  markers?: boolean;
}

/**
 * Transform a row of panels into a horizontal scroll section.
 * The section pins while the user scrolls through all panels.
 *
 * ```html
 * <section class="horizontal-wrapper">
 *   <div class="horizontal-container" style="display:flex">
 *     <div class="panel" />
 *     <div class="panel" />
 *     <div class="panel" />
 *   </div>
 * </section>
 * ```
 */
export function createHorizontalScroll(
  wrapper: gsap.DOMTarget,
  options: HorizontalScrollOptions,
): gsap.core.Tween {
  const container =
    typeof options.container === "string"
      ? document.querySelector(options.container)!
      : (options.container as HTMLElement);
  const panels = container.querySelectorAll(options.panelSelector ?? ".panel");

  return gsap.to(container, {
    x: () => -(container.scrollWidth - window.innerWidth),
    ease: options.ease ?? "none",
    scrollTrigger: {
      trigger: wrapper as gsap.DOMTarget,
      start: "top top",
      end: () => `+=${container.scrollWidth - window.innerWidth}`,
      pin: true,
      scrub: 1,
      snap: 1 / (panels.length - 1),
      markers: options.markers ?? ANIMATION_CONFIG.markers,
      invalidateOnRefresh: true,
    },
  });
}

// ─── Scroll-linked counter ────────────────────────────────────────────────────

export interface ScrollCounterOptions {
  from?: number;
  to: number;
  duration?: number;
  start?: string;
  once?: boolean;
}

/**
 * Count up a number as the element scrolls into view.
 */
export function createScrollCounter(
  target: gsap.DOMTarget,
  options: ScrollCounterOptions,
): gsap.core.Tween {
  const obj = { value: options.from ?? 0 };
  const el =
    typeof target === "string"
      ? document.querySelector(target)!
      : (target as HTMLElement);

  return gsap.to(obj, {
    value: options.to,
    duration: options.duration ?? 2,
    ease: "power1.out",
    scrollTrigger: {
      trigger: el,
      start: options.start ?? ANIMATION_CONFIG.scrollStart,
      once: options.once ?? true,
    },
    onUpdate: () => {
      el.textContent = Math.round(obj.value).toLocaleString();
    },
  });
}

// ─── Batch reveal ──────────────────────────────────────────────────────────────

export interface BatchRevealOptions {
  variant?: AnimationVariant;
  stagger?: number;
  duration?: number;
  ease?: string;
  start?: string;
  /** Number of elements per batch (default: all visible) */
  batchMax?: number;
  once?: boolean;
}

/**
 * Efficiently animate large lists by batching elements that enter
 * the viewport together.
 */
export function createBatchReveal(
  targets: string,
  options: BatchRevealOptions = {},
): void {
  const variant = options.variant ?? "fade-up";
  const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM["fade-up"];

  ScrollTrigger.batch(targets, {
    start: options.start ?? ANIMATION_CONFIG.scrollStart,
    once: options.once ?? true,
    batchMax: options.batchMax,
    onEnter: (batch) => {
      gsap.from(batch, {
        ...fromVars,
        duration: options.duration ?? ANIMATION_CONFIG.defaultDuration,
        stagger: options.stagger ?? 0.05,
        ease: resolveEase(options.ease),
        overwrite: true,
      });
    },
  });
}
