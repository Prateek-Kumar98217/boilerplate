/**
 * SVG animation patterns.
 *
 * Path drawing, dash-offset reveals, and morph placeholders.
 * MorphSVG requires GSAP Club — a CSS-only fallback is included.
 *
 * ```ts
 * useGsap(container, () => {
 *   createPathDraw(".logo path", { duration: 2 });
 *   createSVGStaggerPaths(".icon path", { stagger: 0.15 });
 * });
 * ```
 */

import { gsap } from "gsap";
import { ANIMATION_CONFIG, resolveEase } from "../core/config";

// ─── Path draw (stroke dashoffset) ─────────────────────────────────────────────

export interface PathDrawOptions {
  /** Duration in seconds */
  duration?: number;
  /** Delay in seconds */
  delay?: number;
  /** Ease */
  ease?: string;
  /** Direction: forward draws in, backward erases */
  direction?: "forward" | "backward";
  /** ScrollTrigger options */
  scrollTrigger?: boolean | ScrollTrigger.Vars;
  /** Fill after draw completes */
  fillAfter?: boolean;
  /** Fill color when fillAfter is true */
  fillColor?: string;
}

/**
 * Animate an SVG path's stroke from invisible to fully drawn.
 * Automatically calculates `getTotalLength()` for each path.
 */
export function createPathDraw(
  targets: gsap.TweenTarget,
  options: PathDrawOptions = {},
): gsap.core.Timeline {
  const elements =
    typeof targets === "string"
      ? Array.from(document.querySelectorAll<SVGPathElement>(targets))
      : Array.isArray(targets)
        ? targets
        : [targets];

  const tl = gsap.timeline({ delay: options.delay ?? 0 });

  (elements as SVGPathElement[]).forEach((path) => {
    const length = path.getTotalLength();

    // Set up initial state
    gsap.set(path, {
      strokeDasharray: length,
      strokeDashoffset: options.direction === "backward" ? 0 : length,
      fill: "transparent",
    });

    tl.to(
      path,
      {
        strokeDashoffset: options.direction === "backward" ? length : 0,
        duration: options.duration ?? 1.5,
        ease: resolveEase(options.ease ?? "smooth"),
        scrollTrigger: options.scrollTrigger
          ? typeof options.scrollTrigger === "object"
            ? options.scrollTrigger
            : {
                trigger: path,
                start: ANIMATION_CONFIG.scrollStart,
                once: true,
              }
          : undefined,
      },
      0,
    );

    // Optionally fill after drawing
    if (options.fillAfter) {
      tl.to(path, {
        fill: options.fillColor ?? "currentColor",
        duration: 0.4,
        ease: "power1.out",
      });
    }
  });

  return tl;
}

// ─── Stagger SVG paths ─────────────────────────────────────────────────────────

export interface SVGStaggerPathsOptions {
  /** Stagger between paths */
  stagger?: number;
  /** Duration per path */
  duration?: number;
  /** Ease */
  ease?: string;
  /** Fill after draw */
  fillAfter?: boolean;
  fillColor?: string;
  scrollTrigger?: boolean | ScrollTrigger.Vars;
}

/**
 * Draw multiple SVG paths sequentially with stagger.
 * Great for multi-path icons and illustrations.
 */
export function createSVGStaggerPaths(
  targets: string,
  options: SVGStaggerPathsOptions = {},
): gsap.core.Timeline {
  const paths = Array.from(document.querySelectorAll<SVGPathElement>(targets));
  const tl = gsap.timeline();

  paths.forEach((path) => {
    const length = path.getTotalLength();
    gsap.set(path, {
      strokeDasharray: length,
      strokeDashoffset: length,
      fill: "transparent",
    });
  });

  tl.to(paths, {
    strokeDashoffset: 0,
    duration: options.duration ?? 1,
    stagger: options.stagger ?? 0.15,
    ease: resolveEase(options.ease ?? "smooth"),
    scrollTrigger: options.scrollTrigger
      ? typeof options.scrollTrigger === "object"
        ? options.scrollTrigger
        : {
            trigger: paths[0]?.closest("svg") ?? paths[0],
            start: ANIMATION_CONFIG.scrollStart,
            once: true,
          }
      : undefined,
  });

  if (options.fillAfter) {
    tl.to(paths, {
      fill: options.fillColor ?? "currentColor",
      duration: 0.3,
      stagger: options.stagger ? (options.stagger as number) / 2 : 0.08,
    });
  }

  return tl;
}

// ─── SVG morph (requires GSAP Club MorphSVGPlugin) ────────────────────────────

export interface SVGMorphOptions {
  /** Target shape — CSS selector for the target <path> or an SVG path string */
  shape: string;
  /** Duration in seconds */
  duration?: number;
  /** Ease */
  ease?: string;
  /** Repeat (-1 = infinite) */
  repeat?: number;
  /** Yoyo */
  yoyo?: boolean;
}

/**
 * Morph one SVG shape into another.
 *
 * ⚠️ Requires GSAP Club's `MorphSVGPlugin`. Falls back to a
 *    scale + opacity crossfade if the plugin is not registered.
 */
export function createSVGMorph(
  source: gsap.TweenTarget,
  options: SVGMorphOptions,
): gsap.core.Tween {
  // Check if MorphSVGPlugin is available
  const hasMorphPlugin = !!(gsap as any).plugins?.morphSVG;

  if (hasMorphPlugin) {
    return gsap.to(source, {
      morphSVG: options.shape,
      duration: options.duration ?? 1,
      ease: resolveEase(options.ease ?? "smooth"),
      repeat: options.repeat ?? 0,
      yoyo: options.yoyo ?? false,
    });
  }

  // Fallback: simple crossfade
  console.warn(
    "[gsap-animations] MorphSVGPlugin not registered. Using fallback crossfade.",
  );
  return gsap.to(source, {
    scale: 0.9,
    opacity: 0.5,
    duration: options.duration ?? 1,
    ease: resolveEase(options.ease),
    repeat: options.repeat ?? 0,
    yoyo: options.yoyo ?? true,
  });
}

// ─── SVG line animation ────────────────────────────────────────────────────────

export interface SVGLineAnimationOptions {
  /** Duration in seconds */
  duration?: number;
  /** Ease */
  ease?: string;
  /** Repeat (-1 = infinite) */
  repeat?: number;
  /** Dash pattern (e.g. "10 5") */
  dashPattern?: string;
}

/**
 * Animated dashed line — the dashes flow along the path continuously.
 * Great for flow diagrams or progress indicators.
 */
export function createSVGLineAnimation(
  target: gsap.TweenTarget,
  options: SVGLineAnimationOptions = {},
): gsap.core.Tween {
  const dashPattern = options.dashPattern ?? "10 10";

  gsap.set(target, {
    strokeDasharray: dashPattern,
  });

  return gsap.to(target, {
    strokeDashoffset: -100,
    duration: options.duration ?? 2,
    ease: options.ease ?? "none",
    repeat: options.repeat ?? -1,
  });
}
