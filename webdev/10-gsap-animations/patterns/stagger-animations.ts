/**
 * Stagger animation patterns.
 *
 * Grid reveals, list cascades, and advanced stagger configurations
 * for animating groups of elements.
 *
 * ```ts
 * useGsap(container, () => {
 *   createStaggerReveal(".card", { from: "center", variant: "scale" });
 *   createGridReveal(".grid-item", { grid: [4, 3], axis: "x" });
 *   createCascade(".list-item", { direction: "down" });
 * });
 * ```
 */

import { gsap } from "gsap";
import { ANIMATION_CONFIG, VARIANT_FROM, resolveEase } from "../core/config";
import type { AnimationVariant, StaggerOrigin } from "../types";

// ─── Basic stagger reveal ──────────────────────────────────────────────────────

export interface StaggerRevealOptions {
  variant?: AnimationVariant;
  stagger?: number;
  from?: StaggerOrigin;
  duration?: number;
  ease?: string;
  scrollTrigger?: boolean | ScrollTrigger.Vars;
}

/**
 * Stagger-animate a set of elements from a named origin.
 */
export function createStaggerReveal(
  targets: gsap.TweenTarget,
  options: StaggerRevealOptions = {},
): gsap.core.Tween {
  const variant = options.variant ?? "fade-up";
  const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM["fade-up"];

  const staggerVars: gsap.StaggerVars = {
    each: options.stagger ?? ANIMATION_CONFIG.defaultStagger,
    from: options.from ?? "start",
  };

  const tweenVars: gsap.TweenVars = {
    ...fromVars,
    stagger: staggerVars,
    duration: options.duration ?? ANIMATION_CONFIG.defaultDuration,
    ease: resolveEase(options.ease),
  };

  if (options.scrollTrigger) {
    tweenVars.scrollTrigger =
      typeof options.scrollTrigger === "object"
        ? options.scrollTrigger
        : {
            trigger: targets as gsap.DOMTarget,
            start: ANIMATION_CONFIG.scrollStart,
            once: true,
          };
  }

  return gsap.from(targets, tweenVars);
}

// ─── Grid reveal ───────────────────────────────────────────────────────────────

export interface GridRevealOptions {
  /** Grid dimensions [cols, rows] or "auto" */
  grid?: [number, number] | "auto";
  /** Stagger axis for grid animations */
  axis?: "x" | "y" | null;
  /** Stagger amount per element */
  stagger?: number;
  /** Stagger origin */
  from?: StaggerOrigin | number;
  variant?: AnimationVariant;
  duration?: number;
  ease?: string;
  scrollTrigger?: boolean | ScrollTrigger.Vars;
}

/**
 * 2D grid stagger with directional control — items ripple from
 * a corner, center, or edges.
 */
export function createGridReveal(
  targets: gsap.TweenTarget,
  options: GridRevealOptions = {},
): gsap.core.Tween {
  const variant = options.variant ?? "scale";
  const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM.scale;

  const staggerVars: gsap.StaggerVars = {
    each: options.stagger ?? 0.06,
    grid: options.grid ?? "auto",
    axis: options.axis ?? null,
    from: options.from ?? "center",
  };

  const tweenVars: gsap.TweenVars = {
    ...fromVars,
    stagger: staggerVars,
    duration: options.duration ?? ANIMATION_CONFIG.defaultDuration,
    ease: resolveEase(options.ease ?? "back"),
  };

  if (options.scrollTrigger) {
    tweenVars.scrollTrigger =
      typeof options.scrollTrigger === "object"
        ? options.scrollTrigger
        : {
            trigger: targets as gsap.DOMTarget,
            start: ANIMATION_CONFIG.scrollStart,
            once: true,
          };
  }

  return gsap.from(targets, tweenVars);
}

// ─── Cascade (sequential list) ─────────────────────────────────────────────────

export interface CascadeOptions {
  /** Direction of cascade */
  direction?: "down" | "up" | "left" | "right";
  /** Distance in pixels */
  distance?: number;
  /** Stagger between items */
  stagger?: number;
  /** Duration of each item's animation */
  duration?: number;
  ease?: string;
  scrollTrigger?: boolean | ScrollTrigger.Vars;
}

/**
 * Sequential cascade — items slide in one after another, like
 * a waterfall list.
 */
export function createCascade(
  targets: gsap.TweenTarget,
  options: CascadeOptions = {},
): gsap.core.Tween {
  const direction = options.direction ?? "down";
  const distance = options.distance ?? 60;

  const axis = direction === "down" || direction === "up" ? "y" : "x";
  const value =
    direction === "down" || direction === "right" ? -distance : distance;

  return gsap.from(targets, {
    [axis]: value,
    opacity: 0,
    stagger: options.stagger ?? 0.1,
    duration: options.duration ?? 0.6,
    ease: resolveEase(options.ease ?? "smooth"),
    scrollTrigger:
      options.scrollTrigger === true
        ? {
            trigger: targets as gsap.DOMTarget,
            start: ANIMATION_CONFIG.scrollStart,
            once: true,
          }
        : typeof options.scrollTrigger === "object"
          ? options.scrollTrigger
          : undefined,
  });
}

// ─── Shuffle reveal ────────────────────────────────────────────────────────────

export interface ShuffleRevealOptions {
  variant?: AnimationVariant;
  duration?: number;
  ease?: string;
  scrollTrigger?: boolean | ScrollTrigger.Vars;
}

/**
 * Elements appear in random order — creates a playful, organic feel.
 */
export function createShuffleReveal(
  targets: gsap.TweenTarget,
  options: ShuffleRevealOptions = {},
): gsap.core.Tween {
  const variant = options.variant ?? "fade-up";
  const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM["fade-up"];

  const tweenVars: gsap.TweenVars = {
    ...fromVars,
    stagger: {
      each: 0.08,
      from: "random" as StaggerOrigin,
    },
    duration: options.duration ?? ANIMATION_CONFIG.defaultDuration,
    ease: resolveEase(options.ease),
  };

  if (options.scrollTrigger) {
    tweenVars.scrollTrigger =
      typeof options.scrollTrigger === "object"
        ? options.scrollTrigger
        : {
            trigger: targets as gsap.DOMTarget,
            start: ANIMATION_CONFIG.scrollStart,
            once: true,
          };
  }

  return gsap.from(targets, tweenVars);
}
