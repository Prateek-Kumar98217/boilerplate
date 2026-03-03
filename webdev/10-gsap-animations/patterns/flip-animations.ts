/**
 * FLIP animation patterns.
 *
 * FLIP (First, Last, Invert, Play) enables smooth layout transitions
 * when elements change position, size, or parent in the DOM.
 *
 * ⚠️ Requires GSAP Club's `Flip` plugin. Falls back to a simple
 *    crossfade if the plugin is not registered.
 *
 * ```ts
 * import { Flip } from "gsap/Flip";
 * gsap.registerPlugin(Flip);
 *
 * // Record state, mutate DOM, animate
 * const state = Flip.getState(".card");
 * container.classList.toggle("grid-view");
 * Flip.from(state, { duration: 0.6, ease: "power2.out", stagger: 0.04 });
 * ```
 */

import { gsap } from "gsap";
import { ANIMATION_CONFIG, resolveEase } from "../core/config";
import type { FlipOptions } from "../types";

// ─── FLIP helpers ──────────────────────────────────────────────────────────────

/**
 * Check if the Flip plugin is available.
 */
function getFlipPlugin(): any | null {
  try {
    // Flip is typically at (gsap as any).Flip or imported directly
    return (gsap as any).plugins?.flip ?? (globalThis as any).Flip ?? null;
  } catch {
    return null;
  }
}

// ─── Layout transition ────────────────────────────────────────────────────────

/**
 * Animate a layout change using FLIP.
 *
 * Call this function, perform your DOM mutation, then the returned
 * `animate` function handles the transition.
 *
 * ```ts
 * const { getState, animate } = createFlipTransition(".card");
 * const state = getState();
 * toggleLayout(); // DOM mutation
 * animate(state, { duration: 0.5, stagger: 0.04 });
 * ```
 */
export function createFlipTransition(targets: string | Element[]) {
  const Flip = getFlipPlugin();

  if (!Flip) {
    console.warn(
      "[gsap-animations] Flip plugin not registered. Using fallback.",
    );
    return {
      getState: () => null,
      animate: (_state: any, _options?: FlipOptions) => {
        // Fallback: simple fade
        gsap.from(targets, {
          opacity: 0,
          duration: 0.4,
          stagger: 0.04,
        });
      },
    };
  }

  return {
    getState: () => Flip.getState(targets),
    animate: (state: any, options: FlipOptions = {}) => {
      Flip.from(state, {
        duration: options.duration ?? 0.6,
        ease: resolveEase(options.ease ?? "smooth"),
        stagger: options.stagger ?? 0.04,
        absolute: options.absolute ?? true,
        scale: options.scale ?? true,
        nested: options.nested ?? false,
        onComplete: options.onComplete,
      });
    },
  };
}

// ─── Toggle layout ─────────────────────────────────────────────────────────────

export interface FlipToggleOptions extends FlipOptions {
  /** CSS class to toggle on the container */
  toggleClass: string;
  /** Container element */
  container: HTMLElement;
}

/**
 * One-call FLIP layout toggle: records state, toggles a class, animates.
 *
 * ```ts
 * flipToggleLayout({
 *   container: gridEl,
 *   toggleClass: "list-view",
 *   duration: 0.5,
 * });
 * ```
 */
export function flipToggleLayout(
  targets: string | Element[],
  options: FlipToggleOptions,
): void {
  const Flip = getFlipPlugin();

  if (!Flip) {
    options.container.classList.toggle(options.toggleClass);
    gsap.from(targets, { opacity: 0, duration: 0.3 });
    return;
  }

  const state = Flip.getState(targets);
  options.container.classList.toggle(options.toggleClass);

  Flip.from(state, {
    duration: options.duration ?? 0.6,
    ease: resolveEase(options.ease ?? "smooth"),
    stagger: options.stagger ?? 0.04,
    absolute: options.absolute ?? true,
    scale: options.scale ?? true,
    onComplete: options.onComplete,
  });
}

// ─── Reorder list ──────────────────────────────────────────────────────────────

/**
 * Smoothly animate a list reorder (e.g. after drag-and-drop or sort change).
 *
 * ```ts
 * const items = document.querySelectorAll(".list-item");
 * const state = Flip.getState(items);
 * // ... reorder DOM ...
 * flipReorderList(items, state);
 * ```
 */
export function flipReorderList(
  targets: string | Element[] | NodeListOf<Element>,
  previousState: any,
  options: FlipOptions = {},
): void {
  const Flip = getFlipPlugin();

  if (!Flip) {
    gsap.from(targets, { opacity: 0.5, y: 10, stagger: 0.03, duration: 0.3 });
    return;
  }

  Flip.from(previousState, {
    targets,
    duration: options.duration ?? 0.5,
    ease: resolveEase(options.ease ?? "smooth"),
    stagger: options.stagger ?? 0.03,
    absolute: options.absolute ?? true,
    onComplete: options.onComplete,
  });
}

// ─── Shared element transition ─────────────────────────────────────────────────

export interface SharedElementOptions extends FlipOptions {
  /** Data attribute used to match source and destination elements */
  dataAttribute?: string;
}

/**
 * Shared-element (hero) transition — morph an element from one position
 * to another across layouts (similar to Android shared element transitions).
 *
 * Uses a `data-flip-id` attribute to match source and destination.
 *
 * ```html
 * <!-- List view -->
 * <img data-flip-id="product-1" src="thumb.jpg" />
 *
 * <!-- Detail view -->
 * <img data-flip-id="product-1" src="full.jpg" />
 * ```
 */
export function createSharedElementTransition(
  options: SharedElementOptions = {},
) {
  const Flip = getFlipPlugin();
  const attr = options.dataAttribute ?? "data-flip-id";

  return {
    capture: () => {
      if (!Flip) return null;
      const elements = document.querySelectorAll(`[${attr}]`);
      return Flip.getState(elements);
    },
    animate: (state: any) => {
      if (!Flip || !state) {
        gsap.from(`[${attr}]`, { opacity: 0, duration: 0.4 });
        return;
      }

      Flip.from(state, {
        duration: options.duration ?? 0.8,
        ease: resolveEase(options.ease ?? "expo"),
        stagger: options.stagger ?? 0,
        absolute: options.absolute ?? true,
        scale: options.scale ?? true,
        onComplete: options.onComplete,
      });
    },
  };
}
