/**
 * Barrel exports for 10-gsap-animations.
 *
 * Import from this file for convenience:
 *   import { useGsap, ScrollReveal, createScrollFadeIn } from "@/lib/gsap";
 */

// ─── Core ──────────────────────────────────────────────────────────────────────

export {
  EASINGS,
  ANIMATION_CONFIG,
  VARIANT_FROM,
  prefersReducedMotion,
  resolveEase,
} from "./core/config";

export type { EasingKey } from "./core/config";

export { createTimeline, TimelineBuilder } from "./core/timeline-factory";

export type { TimelineBuilderOptions } from "./core/timeline-factory";

export {
  killTweens,
  killScrollTriggers,
  killAllScrollTriggers,
  revertContext,
  refreshScrollTriggers,
  clearProps,
} from "./core/cleanup";

// ─── Types ─────────────────────────────────────────────────────────────────────

export type {
  EasingPreset,
  AnimationDirection,
  AnimationVariant,
  AnimationOptions,
  ScrubValue,
  ScrollTriggerOptions,
  SplitType,
  TextAnimationOptions,
  ParallaxOptions,
  ParallaxLayer,
  MagneticOptions,
  StaggerOrigin,
  StaggerOptions,
  TransitionVariant,
  PageTransitionOptions,
  SVGDrawOptions,
  SVGMorphOptions,
  FlipOptions,
  ScrollRevealProps,
  AnimatedCounterProps,
  MarqueeProps,
} from "./types";

// ─── Hooks ─────────────────────────────────────────────────────────────────────

export { useGsap } from "./hooks/useGsap";
export type { GsapCallback } from "./hooks/useGsap";

export { useTimeline } from "./hooks/useTimeline";
export type { UseTimelineOptions, TimelineCallback } from "./hooks/useTimeline";

export { useScrollTrigger } from "./hooks/useScrollTrigger";
export type { UseScrollTriggerOptions } from "./hooks/useScrollTrigger";

export { useAnimateOnMount } from "./hooks/useAnimateOnMount";
export type { UseAnimateOnMountOptions } from "./hooks/useAnimateOnMount";

export { useSplitText } from "./hooks/useSplitText";
export type { UseSplitTextOptions } from "./hooks/useSplitText";

// ─── Patterns ──────────────────────────────────────────────────────────────────

export {
  createScrollFadeIn,
  createScrollProgressBar,
  createPinSection,
  createHorizontalScroll,
  createScrollCounter,
  createBatchReveal,
} from "./patterns/scroll-animations";

export {
  createCharReveal,
  createWordReveal,
  createTypewriter,
  createTextScramble,
  createWaveText,
  createGradientReveal,
} from "./patterns/text-animations";

export {
  createStaggerReveal,
  createGridReveal,
  createCascade,
  createShuffleReveal,
} from "./patterns/stagger-animations";

export {
  createPageTransition,
  createOverlayTransition,
} from "./patterns/page-transitions";

export {
  createPathDraw,
  createSVGStaggerPaths,
  createSVGMorph,
  createSVGLineAnimation,
} from "./patterns/svg-animations";

export {
  createParallax,
  createDepthParallax,
  createHeroParallax,
  createScrollTilt,
} from "./patterns/parallax";

export {
  applyMagnetic,
  applyMagneticWithInner,
  createCursorFollower,
} from "./patterns/magnetic";

export {
  createFlipTransition,
  flipToggleLayout,
  flipReorderList,
  createSharedElementTransition,
} from "./patterns/flip-animations";

// ─── Components ────────────────────────────────────────────────────────────────

export {
  AnimatePresence,
  AnimatePresenceControlled,
} from "./components/AnimatePresence";
export { ScrollReveal } from "./components/ScrollReveal";
export { SplitText } from "./components/SplitText";
export { MagneticButton } from "./components/MagneticButton";
export { ParallaxSection } from "./components/ParallaxSection";
export { StaggerContainer } from "./components/StaggerContainer";
export { AnimatedCounter } from "./components/AnimatedCounter";
export { MarqueeText } from "./components/MarqueeText";
