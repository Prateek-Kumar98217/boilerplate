/**
 * Fluent Timeline builder.
 *
 * Makes complex multi-step timelines cleaner to compose:
 *
 * ```ts
 * const tl = createTimeline({ defaults: { duration: 0.6 } })
 *   .fadeIn(".hero-title")
 *   .slideIn(".hero-sub", "up", "<0.2")
 *   .stagger(".cards .card", { y: 40, opacity: 0 }, 0.08)
 *   .build();
 * ```
 */

import { gsap } from "gsap";
import { ANIMATION_CONFIG, VARIANT_FROM, resolveEase } from "./config";
import type { AnimationVariant } from "../types";

export interface TimelineBuilderOptions {
  defaults?: gsap.TweenVars;
  paused?: boolean;
  repeat?: number;
  yoyo?: boolean;
  onComplete?: () => void;
}

export class TimelineBuilder {
  private tl: gsap.core.Timeline;

  constructor(options: TimelineBuilderOptions = {}) {
    this.tl = gsap.timeline({
      paused: options.paused ?? false,
      repeat: options.repeat,
      yoyo: options.yoyo,
      onComplete: options.onComplete,
      defaults: {
        duration: ANIMATION_CONFIG.defaultDuration,
        ease: ANIMATION_CONFIG.defaultEase,
        ...options.defaults,
      },
    });
  }

  /** Return the raw gsap.core.Timeline */
  build(): gsap.core.Timeline {
    return this.tl;
  }

  /** Add a raw `from` tween */
  from(
    targets: gsap.TweenTarget,
    vars: gsap.TweenVars,
    position?: gsap.Position,
  ): this {
    this.tl.from(targets, vars, position);
    return this;
  }

  /** Add a raw `to` tween */
  to(
    targets: gsap.TweenTarget,
    vars: gsap.TweenVars,
    position?: gsap.Position,
  ): this {
    this.tl.to(targets, vars, position);
    return this;
  }

  /** Add a raw `fromTo` tween */
  fromTo(
    targets: gsap.TweenTarget,
    fromVars: gsap.TweenVars,
    toVars: gsap.TweenVars,
    position?: gsap.Position,
  ): this {
    this.tl.fromTo(targets, fromVars, toVars, position);
    return this;
  }

  /** Fade in from opacity 0 */
  fadeIn(targets: gsap.TweenTarget, position?: gsap.Position): this {
    this.tl.from(targets, { opacity: 0 }, position);
    return this;
  }

  /** Fade out to opacity 0 */
  fadeOut(targets: gsap.TweenTarget, position?: gsap.Position): this {
    this.tl.to(targets, { opacity: 0 }, position);
    return this;
  }

  /** Slide in from a direction */
  slideIn(
    targets: gsap.TweenTarget,
    direction: "up" | "down" | "left" | "right" = "up",
    position?: gsap.Position,
    distance = 80,
  ): this {
    const axis = direction === "up" || direction === "down" ? "y" : "x";
    const value =
      direction === "up" || direction === "left" ? distance : -distance;
    this.tl.from(targets, { [axis]: value, opacity: 0 }, position);
    return this;
  }

  /** Animate in using a named variant */
  variantIn(
    targets: gsap.TweenTarget,
    variant: AnimationVariant,
    position?: gsap.Position,
  ): this {
    const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM.fade;
    this.tl.from(targets, fromVars, position);
    return this;
  }

  /** Stagger animate a set of elements */
  stagger(
    targets: gsap.TweenTarget,
    fromVars: gsap.TweenVars,
    stagger?: number,
    position?: gsap.Position,
  ): this {
    this.tl.from(
      targets,
      {
        ...fromVars,
        stagger: stagger ?? ANIMATION_CONFIG.defaultStagger,
      },
      position,
    );
    return this;
  }

  /** Add a label at the current (or specified) position */
  label(name: string, position?: gsap.Position): this {
    this.tl.addLabel(name, position);
    return this;
  }

  /** Add a pause */
  addPause(position?: gsap.Position): this {
    this.tl.addPause(position);
    return this;
  }

  /** Set properties instantly */
  set(targets: gsap.TweenTarget, vars: gsap.TweenVars): this {
    this.tl.set(targets, vars);
    return this;
  }

  /** Call a function at a position */
  call(fn: () => void, position?: gsap.Position): this {
    this.tl.call(fn, undefined, position);
    return this;
  }
}

/**
 * Factory function — preferred API.
 *
 * ```ts
 * const tl = createTimeline().fadeIn(".title").slideIn(".sub", "up", "<0.2").build();
 * ```
 */
export function createTimeline(
  options?: TimelineBuilderOptions,
): TimelineBuilder {
  return new TimelineBuilder(options);
}
