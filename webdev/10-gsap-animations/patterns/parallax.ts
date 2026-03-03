/**
 * Parallax scroll patterns.
 *
 * Depth-layered parallax, speed-based parallax, and directional
 * parallax for hero sections and immersive scroll experiences.
 *
 * ```ts
 * useGsap(container, () => {
 *   createParallax(".bg-layer", { speed: -0.3 });
 *   createDepthParallax([
 *     { target: ".bg",  depth: 0.2 },
 *     { target: ".mid", depth: 0.5 },
 *     { target: ".fg",  depth: 0.9 },
 *   ]);
 * });
 * ```
 */

import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { ANIMATION_CONFIG, resolveEase } from "../core/config";
import type { ParallaxLayer } from "../types";

// ─── Simple parallax ──────────────────────────────────────────────────────────

export interface SimpleParallaxOptions {
  /** Speed multiplier — negative reverses direction (default: 0.5) */
  speed?: number;
  /** Axis of motion (default: "vertical") */
  direction?: "vertical" | "horizontal";
  /** ScrollTrigger start */
  start?: string;
  /** ScrollTrigger end */
  end?: string;
  markers?: boolean;
}

/**
 * Apply a simple parallax shift to an element based on scroll position.
 */
export function createParallax(
  target: gsap.TweenTarget,
  options: SimpleParallaxOptions = {},
): gsap.core.Tween {
  const speed = options.speed ?? 0.5;
  const prop = options.direction === "horizontal" ? "x" : "y";

  return gsap.to(target, {
    [prop]: () => speed * 200,
    ease: "none",
    scrollTrigger: {
      trigger: target as gsap.DOMTarget,
      start: options.start ?? "top bottom",
      end: options.end ?? "bottom top",
      scrub: true,
      markers: options.markers ?? ANIMATION_CONFIG.markers,
    },
  });
}

// ─── Depth-layered parallax ────────────────────────────────────────────────────

export interface DepthParallaxOptions {
  /** Container element for ScrollTrigger */
  container?: gsap.DOMTarget;
  /** ScrollTrigger start */
  start?: string;
  /** ScrollTrigger end */
  end?: string;
  markers?: boolean;
}

/**
 * Multi-layer depth parallax — each layer moves at a different
 * speed based on its `depth` value (0 = static, 1 = full scroll speed).
 */
export function createDepthParallax(
  layers: ParallaxLayer[],
  options: DepthParallaxOptions = {},
): gsap.core.Tween[] {
  return layers.map((layer) => {
    const distance = layer.depth * 300;

    return gsap.to(layer.target, {
      y: -distance,
      ease: "none",
      scrollTrigger: {
        trigger: options.container ?? (layer.target as gsap.DOMTarget),
        start: options.start ?? "top bottom",
        end: options.end ?? "bottom top",
        scrub: true,
        markers: options.markers ?? ANIMATION_CONFIG.markers,
      },
    });
  });
}

// ─── Hero parallax (image + text) ──────────────────────────────────────────────

export interface HeroParallaxOptions {
  /** Image selector (moves slower) */
  imageTarget: gsap.TweenTarget;
  /** Text selector (moves faster or stays) */
  textTarget: gsap.TweenTarget;
  /** Section / container trigger */
  trigger: gsap.DOMTarget;
  /** Image parallax speed (default: 0.3) */
  imageSpeed?: number;
  /** Text parallax speed (default: 0.6) */
  textSpeed?: number;
  /** Whether to fade out text as it scrolls */
  fadeText?: boolean;
}

/**
 * Combined image + text parallax for hero sections.
 * Image drifts slowly, text scrolls away faster.
 */
export function createHeroParallax(
  options: HeroParallaxOptions,
): gsap.core.Timeline {
  const imageSpeed = options.imageSpeed ?? 0.3;
  const textSpeed = options.textSpeed ?? 0.6;

  const tl = gsap.timeline({
    scrollTrigger: {
      trigger: options.trigger,
      start: "top top",
      end: "bottom top",
      scrub: true,
    },
  });

  tl.to(options.imageTarget, { y: () => imageSpeed * 200, ease: "none" }, 0);

  const textVars: gsap.TweenVars = {
    y: () => textSpeed * -200,
    ease: "none",
  };
  if (options.fadeText) {
    textVars.opacity = 0;
  }
  tl.to(options.textTarget, textVars, 0);

  return tl;
}

// ─── Tilt on scroll ────────────────────────────────────────────────────────────

export interface ScrollTiltOptions {
  /** Max rotation in degrees (default: 8) */
  maxRotation?: number;
  /** Axis: "x", "y", or "both" */
  axis?: "x" | "y" | "both";
  ease?: string;
}

/**
 * Apply a slight 3D tilt to elements as they scroll through the viewport.
 */
export function createScrollTilt(
  target: gsap.TweenTarget,
  options: ScrollTiltOptions = {},
): gsap.core.Tween {
  const max = options.maxRotation ?? 8;
  const axis = options.axis ?? "x";

  const vars: gsap.TweenVars = {
    ease: "none",
    scrollTrigger: {
      trigger: target as gsap.DOMTarget,
      start: "top bottom",
      end: "bottom top",
      scrub: true,
    },
  };

  if (axis === "x" || axis === "both") vars.rotationX = max;
  if (axis === "y" || axis === "both") vars.rotationY = max;

  gsap.set(target, { transformPerspective: 800 });

  return gsap.fromTo(
    target,
    {
      rotationX: axis === "x" || axis === "both" ? -max : 0,
      rotationY: axis === "y" || axis === "both" ? -max : 0,
    },
    vars,
  );
}
