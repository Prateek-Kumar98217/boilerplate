/**
 * Magnetic hover / cursor effect patterns.
 *
 * Elements are attracted toward the cursor when hovering within a
 * defined radius. Industry-standard for interactive buttons, CTAs,
 * and nav items.
 *
 * ```ts
 * // Imperative
 * const cleanup = applyMagnetic(document.querySelector(".btn")!, { strength: 0.4 });
 *
 * // React — see components/MagneticButton.tsx
 * ```
 */

import { gsap } from "gsap";
import { resolveEase } from "../core/config";
import type { MagneticOptions } from "../types";

// ─── Core magnetic function ────────────────────────────────────────────────────

/**
 * Apply a magnetic effect to an element. The element shifts toward the
 * cursor position when hovering, and returns to its origin on mouse leave.
 *
 * Returns a cleanup function that removes event listeners.
 */
export function applyMagnetic(
  el: HTMLElement,
  options: MagneticOptions = {},
): () => void {
  const strength = options.strength ?? 0.35;
  const ease = resolveEase(options.ease ?? "expo");
  const duration = options.duration ?? 0.6;
  const rotate = options.rotate ?? false;
  const maxRotation = options.maxRotation ?? 10;

  const handleMouseMove = (e: MouseEvent) => {
    const rect = el.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;

    const dx = e.clientX - cx;
    const dy = e.clientY - cy;

    const vars: gsap.TweenVars = {
      x: dx * strength,
      y: dy * strength,
      duration: 0.3,
      ease: "power2.out",
    };

    if (rotate) {
      vars.rotationX = (dy / rect.height) * -maxRotation;
      vars.rotationY = (dx / rect.width) * maxRotation;
    }

    gsap.to(el, vars);
  };

  const handleMouseLeave = () => {
    gsap.to(el, {
      x: 0,
      y: 0,
      rotationX: 0,
      rotationY: 0,
      duration,
      ease,
    });
  };

  el.addEventListener("mousemove", handleMouseMove);
  el.addEventListener("mouseleave", handleMouseLeave);

  // Optional: set transform perspective for 3D rotate
  if (rotate) {
    gsap.set(el, { transformPerspective: 600 });
  }

  return () => {
    el.removeEventListener("mousemove", handleMouseMove);
    el.removeEventListener("mouseleave", handleMouseLeave);
    gsap.set(el, { clearProps: "x,y,rotationX,rotationY" });
  };
}

// ─── Magnetic with inner element ───────────────────────────────────────────────

export interface MagneticInnerOptions extends MagneticOptions {
  /** CSS selector for the inner element that moves more than the outer */
  innerSelector: string;
  /** Inner element strength multiplier relative to outer (default: 1.5) */
  innerMultiplier?: number;
}

/**
 * Magnetic effect where an inner element (e.g. button text) moves
 * more than the container, creating a depth/lag effect.
 */
export function applyMagneticWithInner(
  el: HTMLElement,
  options: MagneticInnerOptions,
): () => void {
  const strength = options.strength ?? 0.35;
  const innerMultiplier = options.innerMultiplier ?? 1.5;
  const ease = resolveEase(options.ease ?? "expo");
  const duration = options.duration ?? 0.6;
  const inner = el.querySelector<HTMLElement>(options.innerSelector);

  if (!inner) {
    console.warn(
      `[magnetic] Inner element "${options.innerSelector}" not found.`,
    );
    return () => {};
  }

  const handleMouseMove = (e: MouseEvent) => {
    const rect = el.getBoundingClientRect();
    const dx = e.clientX - (rect.left + rect.width / 2);
    const dy = e.clientY - (rect.top + rect.height / 2);

    gsap.to(el, {
      x: dx * strength,
      y: dy * strength,
      duration: 0.3,
      ease: "power2.out",
    });

    gsap.to(inner, {
      x: dx * strength * innerMultiplier,
      y: dy * strength * innerMultiplier,
      duration: 0.3,
      ease: "power2.out",
    });
  };

  const handleMouseLeave = () => {
    gsap.to([el, inner], {
      x: 0,
      y: 0,
      duration,
      ease,
    });
  };

  el.addEventListener("mousemove", handleMouseMove);
  el.addEventListener("mouseleave", handleMouseLeave);

  return () => {
    el.removeEventListener("mousemove", handleMouseMove);
    el.removeEventListener("mouseleave", handleMouseLeave);
    gsap.set([el, inner], { clearProps: "x,y" });
  };
}

// ─── Cursor follower ───────────────────────────────────────────────────────────

export interface CursorFollowerOptions {
  /** Size of the follower in pixels */
  size?: number;
  /** Background color */
  color?: string;
  /** Follower lag (higher = slower, default: 0.15) */
  lag?: number;
  /** Whether to scale up on hovering interactive elements */
  scaleOnHover?: boolean;
  /** Selector for elements that trigger scale effect */
  hoverTargets?: string;
  /** Z-index */
  zIndex?: number;
}

/**
 * Create a custom cursor follower element.
 * Returns a cleanup function.
 */
export function createCursorFollower(
  options: CursorFollowerOptions = {},
): () => void {
  const size = options.size ?? 20;
  const color = options.color ?? "rgba(0, 0, 0, 0.15)";
  const lag = options.lag ?? 0.15;
  const zIndex = options.zIndex ?? 10000;

  // Create follower element
  const follower = document.createElement("div");
  Object.assign(follower.style, {
    position: "fixed",
    top: "0",
    left: "0",
    width: `${size}px`,
    height: `${size}px`,
    borderRadius: "50%",
    backgroundColor: color,
    pointerEvents: "none",
    zIndex: String(zIndex),
    transform: "translate(-50%, -50%)",
    mixBlendMode: "difference",
  });
  document.body.appendChild(follower);

  // Follow cursor with lag
  const pos = { x: 0, y: 0 };

  const handleMouseMove = (e: MouseEvent) => {
    pos.x = e.clientX;
    pos.y = e.clientY;
  };

  window.addEventListener("mousemove", handleMouseMove);

  // Smooth follow with GSAP ticker
  const tickHandler = () => {
    gsap.to(follower, {
      x: pos.x,
      y: pos.y,
      duration: lag,
      ease: "power2.out",
      overwrite: true,
    });
  };

  gsap.ticker.add(tickHandler);

  // Scale on hover targets
  let hoverCleanups: (() => void)[] = [];
  if (options.scaleOnHover && options.hoverTargets) {
    const targets = document.querySelectorAll(options.hoverTargets);
    targets.forEach((target) => {
      const enter = () => gsap.to(follower, { scale: 2.5, duration: 0.3 });
      const leave = () => gsap.to(follower, { scale: 1, duration: 0.3 });
      target.addEventListener("mouseenter", enter);
      target.addEventListener("mouseleave", leave);
      hoverCleanups.push(() => {
        target.removeEventListener("mouseenter", enter);
        target.removeEventListener("mouseleave", leave);
      });
    });
  }

  return () => {
    window.removeEventListener("mousemove", handleMouseMove);
    gsap.ticker.remove(tickHandler);
    hoverCleanups.forEach((fn) => fn());
    follower.remove();
  };
}
