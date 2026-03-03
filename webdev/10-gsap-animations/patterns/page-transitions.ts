/**
 * Page / route transition patterns.
 *
 * Enter and exit animation factories for route-level transitions.
 * Designed for Next.js App Router or any SPA framework.
 *
 * ```ts
 * // In a layout wrapper
 * const { animateEnter, animateExit } = createPageTransition("slide");
 *
 * // On route change
 * await animateExit(containerEl);
 * router.push(nextRoute);
 * animateEnter(containerEl);
 * ```
 */

import { gsap } from "gsap";
import { ANIMATION_CONFIG, resolveEase } from "../core/config";
import type { TransitionVariant } from "../types";

// ─── Transition preset map ────────────────────────────────────────────────────

interface TransitionPreset {
  enter: gsap.TweenVars;
  exit: gsap.TweenVars;
}

const PRESETS: Record<TransitionVariant, TransitionPreset> = {
  fade: {
    enter: { opacity: 0 },
    exit: { opacity: 0 },
  },
  slide: {
    enter: { x: "100%", opacity: 0 },
    exit: { x: "-30%", opacity: 0 },
  },
  "slide-up": {
    enter: { y: "100%", opacity: 0 },
    exit: { y: "-30%", opacity: 0 },
  },
  "slide-down": {
    enter: { y: "-100%", opacity: 0 },
    exit: { y: "30%", opacity: 0 },
  },
  scale: {
    enter: { scale: 0.9, opacity: 0 },
    exit: { scale: 1.05, opacity: 0 },
  },
  curtain: {
    enter: { clipPath: "inset(0 0 100% 0)" },
    exit: { clipPath: "inset(100% 0 0 0)" },
  },
  wipe: {
    enter: { clipPath: "inset(0 100% 0 0)" },
    exit: { clipPath: "inset(0 0 0 100%)" },
  },
};

// ─── Factory ───────────────────────────────────────────────────────────────────

export interface PageTransitionConfig {
  duration?: number;
  enterEase?: string;
  exitEase?: string;
}

export interface PageTransitionAPI {
  /** Animate the incoming page */
  animateEnter: (target: gsap.TweenTarget) => gsap.core.Tween;
  /** Animate the outgoing page — returns a Promise that resolves on complete */
  animateExit: (target: gsap.TweenTarget) => Promise<void>;
  /** Combined exit → enter on same element */
  transition: (target: gsap.TweenTarget) => Promise<void>;
}

/**
 * Create enter / exit animation functions for a given transition variant.
 */
export function createPageTransition(
  variant: TransitionVariant = "fade",
  config: PageTransitionConfig = {},
): PageTransitionAPI {
  const preset = PRESETS[variant] ?? PRESETS.fade;
  const duration = config.duration ?? 0.5;
  const enterEase = resolveEase(config.enterEase ?? "expo");
  const exitEase = resolveEase(config.exitEase ?? "expo");

  const animateEnter = (target: gsap.TweenTarget): gsap.core.Tween => {
    return gsap.from(target, {
      ...preset.enter,
      duration,
      ease: enterEase,
      clearProps: "all",
    });
  };

  const animateExit = (target: gsap.TweenTarget): Promise<void> => {
    return new Promise((resolve) => {
      gsap.to(target, {
        ...preset.exit,
        duration,
        ease: exitEase,
        onComplete: resolve,
      });
    });
  };

  const transition = async (target: gsap.TweenTarget): Promise<void> => {
    await animateExit(target);
    animateEnter(target);
  };

  return { animateEnter, animateExit, transition };
}

// ─── Overlay transition ────────────────────────────────────────────────────────

export interface OverlayTransitionOptions {
  /** Overlay background color */
  color?: string;
  /** Duration for each half of the transition */
  duration?: number;
  /** Ease */
  ease?: string;
  /** Overlay z-index */
  zIndex?: number;
}

/**
 * Full-screen overlay transition — a colored layer sweeps across the
 * viewport, content changes behind it, then the layer exits.
 *
 * Creates and removes the overlay DOM element automatically.
 */
export function createOverlayTransition(
  options: OverlayTransitionOptions = {},
): {
  /** Call this to start the transition. Resolves mid-transition (overlay covers screen). */
  run: () => Promise<void>;
} {
  const color = options.color ?? "#000";
  const duration = options.duration ?? 0.5;
  const ease = resolveEase(options.ease ?? "expo");
  const zIndex = options.zIndex ?? 9999;

  const run = (): Promise<void> => {
    return new Promise((resolve) => {
      const overlay = document.createElement("div");
      Object.assign(overlay.style, {
        position: "fixed",
        inset: "0",
        backgroundColor: color,
        zIndex: String(zIndex),
        transformOrigin: "left",
        transform: "scaleX(0)",
      });
      document.body.appendChild(overlay);

      const tl = gsap.timeline();

      // Cover
      tl.to(overlay, {
        scaleX: 1,
        duration,
        ease,
        onComplete: () => resolve(), // content can change now
      });

      // Reveal
      tl.to(overlay, {
        scaleX: 0,
        transformOrigin: "right",
        duration,
        ease,
        onComplete: () => overlay.remove(),
      });
    });
  };

  return { run };
}
