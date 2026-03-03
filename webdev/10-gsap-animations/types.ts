/**
 * Shared types for GSAP animation patterns.
 *
 * Covers animation presets, scroll-trigger options, text-split modes,
 * magnetic effects, and component prop interfaces.
 */

// ─── Base animation ────────────────────────────────────────────────────────────

export type EasingPreset =
  | "smooth"
  | "bounce"
  | "elastic"
  | "snap"
  | "expo"
  | "circ"
  | "back"
  | "steps"
  | "none";

export type AnimationDirection = "up" | "down" | "left" | "right";

export type AnimationVariant =
  | "fade"
  | "fade-up"
  | "fade-down"
  | "fade-left"
  | "fade-right"
  | "scale"
  | "scale-up"
  | "scale-down"
  | "slide-up"
  | "slide-down"
  | "slide-left"
  | "slide-right"
  | "rotate"
  | "flip-x"
  | "flip-y"
  | "blur";

export interface AnimationOptions {
  /** Duration in seconds (default: from config) */
  duration?: number;
  /** Delay before animation starts in seconds */
  delay?: number;
  /** GSAP ease string or EasingPreset key */
  ease?: string | EasingPreset;
  /** Stagger between elements in seconds */
  stagger?: number | gsap.StaggerVars;
  /** Whether animation should repeat (-1 = infinite) */
  repeat?: number;
  /** Delay between repeats in seconds */
  repeatDelay?: number;
  /** Whether to yoyo on repeat */
  yoyo?: boolean;
  /** Callback when animation completes */
  onComplete?: () => void;
  /** Callback when animation starts */
  onStart?: () => void;
}

// ─── Scroll trigger ────────────────────────────────────────────────────────────

export type ScrubValue = boolean | number;

export interface ScrollTriggerOptions {
  /** ScrollTrigger trigger element / selector */
  trigger?: string | Element;
  /** Start position (default: "top 80%") */
  start?: string;
  /** End position */
  end?: string;
  /** Scrub value: true, false, or number (smooth scrub seconds) */
  scrub?: ScrubValue;
  /** Pin the trigger element */
  pin?: boolean | string | Element;
  /** Snap to progress values */
  snap?: number | number[] | gsap.SnapVars;
  /** Toggle CSS classes */
  toggleClass?: string | gsap.ToggleClassVars;
  /** Show debug markers */
  markers?: boolean;
  /** Callback when entering viewport */
  onEnter?: () => void;
  /** Callback when leaving viewport */
  onLeave?: () => void;
  /** Callback when scrolling back into viewport */
  onEnterBack?: () => void;
  /** Callback when scrolling back past viewport */
  onLeaveBack?: () => void;
}

// ─── Text animation ───────────────────────────────────────────────────────────

export type SplitType =
  | "chars"
  | "words"
  | "lines"
  | "chars,words"
  | "words,lines";

export interface TextAnimationOptions extends AnimationOptions {
  /** How to split the text */
  splitType?: SplitType;
  /** Animation variant for each split element */
  variant?: AnimationVariant;
  /** Whether triggered by scroll */
  trigger?: "mount" | "scroll";
  /** ScrollTrigger options when trigger = "scroll" */
  scrollTrigger?: ScrollTriggerOptions;
}

// ─── Parallax ──────────────────────────────────────────────────────────────────

export interface ParallaxOptions {
  /** Speed multiplier (negative = opposite direction, default: 0.5) */
  speed?: number;
  /** Direction of parallax motion */
  direction?: "vertical" | "horizontal";
  /** ScrollTrigger overrides */
  scrollTrigger?: ScrollTriggerOptions;
}

export interface ParallaxLayer {
  /** CSS selector or element */
  target: string | Element;
  /** Depth multiplier (0 = static, 1 = full speed) */
  depth: number;
}

// ─── Magnetic ──────────────────────────────────────────────────────────────────

export interface MagneticOptions {
  /** Strength of magnetic pull (0–1, default: 0.35) */
  strength?: number;
  /** Easing for return to original position */
  ease?: string;
  /** Duration of return animation in seconds */
  duration?: number;
  /** Whether to apply rotation based on cursor position */
  rotate?: boolean;
  /** Max rotation in degrees */
  maxRotation?: number;
}

// ─── Stagger ───────────────────────────────────────────────────────────────────

export type StaggerOrigin = "start" | "center" | "end" | "edges" | "random";

export interface StaggerOptions extends AnimationOptions {
  /** Stagger origin point */
  from?: StaggerOrigin;
  /** Grid dimensions for 2D stagger [cols, rows] */
  grid?: [number, number] | "auto";
  /** Axis for grid stagger */
  axis?: "x" | "y" | null;
  /** Animation variant for staggered items */
  variant?: AnimationVariant;
}

// ─── Page transitions ──────────────────────────────────────────────────────────

export type TransitionVariant =
  | "fade"
  | "slide"
  | "slide-up"
  | "slide-down"
  | "scale"
  | "curtain"
  | "wipe";

export interface PageTransitionOptions {
  /** Transition style */
  variant?: TransitionVariant;
  /** Duration in seconds */
  duration?: number;
  /** Ease for enter transition */
  enterEase?: string;
  /** Ease for exit transition */
  exitEase?: string;
}

// ─── SVG ───────────────────────────────────────────────────────────────────────

export interface SVGDrawOptions extends AnimationOptions {
  /** Draw direction: forward or backward */
  direction?: "forward" | "backward";
  /** Stroke dashoffset start value */
  drawFrom?: number;
  /** Stroke dashoffset end value */
  drawTo?: number;
}

export interface SVGMorphOptions extends AnimationOptions {
  /** Target shape (CSS selector or SVG path string) */
  shape?: string;
  /** SVG origin point */
  origin?: string;
  /** Morph precision */
  precision?: number;
}

// ─── FLIP ──────────────────────────────────────────────────────────────────────

export interface FlipOptions {
  /** Duration of FLIP animation */
  duration?: number;
  /** Ease for FLIP */
  ease?: string;
  /** Stagger between flipped elements */
  stagger?: number;
  /** Absolute positioning during animation */
  absolute?: boolean;
  /** Scale animation */
  scale?: boolean;
  /** Nested elements to also animate */
  nested?: boolean;
  /** Callback on complete */
  onComplete?: () => void;
}

// ─── Component props ───────────────────────────────────────────────────────────

export interface ScrollRevealProps {
  /** Animation variant */
  variant?: AnimationVariant;
  /** Duration in seconds */
  duration?: number;
  /** Stagger between children */
  stagger?: number;
  /** Delay before animation starts */
  delay?: number;
  /** ScrollTrigger start position */
  start?: string;
  /** Whether to only animate once */
  once?: boolean;
  /** Custom className */
  className?: string;
  children: React.ReactNode;
}

export interface AnimatedCounterProps {
  /** Starting number */
  from?: number;
  /** Target number */
  to: number;
  /** Duration in seconds */
  duration?: number;
  /** Decimal places */
  decimals?: number;
  /** Thousand separator */
  separator?: string;
  /** Prefix (e.g. "$") */
  prefix?: string;
  /** Suffix (e.g. "%") */
  suffix?: string;
  /** Trigger on scroll */
  scrollTrigger?: boolean;
  /** Custom className */
  className?: string;
}

export interface MarqueeProps {
  /** Scroll speed in pixels per second */
  speed?: number;
  /** Direction of scroll */
  direction?: "left" | "right";
  /** Pause on hover */
  pauseOnHover?: boolean;
  /** Gap between repeated items in pixels */
  gap?: number;
  /** Custom className */
  className?: string;
  children: React.ReactNode;
}
