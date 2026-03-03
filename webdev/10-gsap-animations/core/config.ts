/**
 * Default animation configuration and easing presets.
 *
 * Override globally by mutating ANIMATION_CONFIG, or per-call by passing
 * options to individual pattern functions / hooks.
 */

// ─── Easing presets ────────────────────────────────────────────────────────────

export const EASINGS = {
  smooth: "power2.out",
  smoothInOut: "power2.inOut",
  bounce: "bounce.out",
  elastic: "elastic.out(1, 0.5)",
  snap: "power4.out",
  expo: "expo.out",
  expoInOut: "expo.inOut",
  circ: "circ.out",
  back: "back.out(1.7)",
  backIn: "back.in(1.7)",
  steps: "steps(12)",
  none: "none",
} as const;

export type EasingKey = keyof typeof EASINGS;

// ─── Default config ────────────────────────────────────────────────────────────

export const ANIMATION_CONFIG = {
  /** Default tween duration in seconds */
  defaultDuration: 0.8,

  /** Default ease curve */
  defaultEase: EASINGS.smooth,

  /** Default stagger interval in seconds */
  defaultStagger: 0.08,

  /** Default scroll trigger start position */
  scrollStart: "top 80%",

  /** Default scroll trigger end position */
  scrollEnd: "bottom 20%",

  /** Whether ScrollTrigger markers are shown (dev only) */
  markers: false,

  /** Reduced-motion: skip animations for prefers-reduced-motion */
  respectReducedMotion: true,

  /** Default text split type */
  defaultSplitType: "words" as const,

  /** Breakpoints matching common Tailwind defaults */
  breakpoints: {
    sm: 640,
    md: 768,
    lg: 1024,
    xl: 1280,
    "2xl": 1536,
  },
} as const;

// ─── Variant offset map ────────────────────────────────────────────────────────

/**
 * Maps AnimationVariant → GSAP `from` vars.
 * Used by ScrollReveal, AnimatePresence, StaggerContainer, etc.
 */
export const VARIANT_FROM: Record<string, gsap.TweenVars> = {
  fade: { opacity: 0 },
  "fade-up": { opacity: 0, y: 60 },
  "fade-down": { opacity: 0, y: -60 },
  "fade-left": { opacity: 0, x: -60 },
  "fade-right": { opacity: 0, x: 60 },
  scale: { opacity: 0, scale: 0.85 },
  "scale-up": { opacity: 0, scale: 0.6, y: 40 },
  "scale-down": { opacity: 0, scale: 1.3, y: -40 },
  "slide-up": { y: 100 },
  "slide-down": { y: -100 },
  "slide-left": { x: -100 },
  "slide-right": { x: 100 },
  rotate: { opacity: 0, rotation: 15 },
  "flip-x": { opacity: 0, rotationX: 90 },
  "flip-y": { opacity: 0, rotationY: 90 },
  blur: { opacity: 0, filter: "blur(12px)" },
};

// ─── Reduced motion helper ─────────────────────────────────────────────────────

/**
 * Returns true when the user prefers reduced motion AND config says
 * we should respect that preference.
 */
export function prefersReducedMotion(): boolean {
  if (!ANIMATION_CONFIG.respectReducedMotion) return false;
  if (typeof window === "undefined") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Resolves an ease key: if user passes a key from EASINGS, return the
 * GSAP string; otherwise treat it as a raw GSAP ease string.
 */
export function resolveEase(ease?: string): string {
  if (!ease) return ANIMATION_CONFIG.defaultEase;
  return (EASINGS as Record<string, string>)[ease] ?? ease;
}
