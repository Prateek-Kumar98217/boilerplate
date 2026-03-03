/**
 * Text animation patterns.
 *
 * Character, word, and line-level reveal animations — typewriter,
 * scramble, wave, and gradient reveals.
 *
 * Uses a lightweight manual splitting approach. If you have GSAP Club,
 * replace with `SplitText` for best results.
 *
 * ```ts
 * useGsap(container, () => {
 *   createCharReveal(".headline");
 *   createTypewriter(".subtitle", { speed: 0.05 });
 *   createTextScramble(".tagline", { chars: "!<>-_\\/[]{}—=+*^?#" });
 * });
 * ```
 */

import { gsap } from "gsap";
import { ANIMATION_CONFIG, VARIANT_FROM, resolveEase } from "../core/config";
import type { AnimationVariant } from "../types";

// ─── Internal helpers ──────────────────────────────────────────────────────────

function splitIntoSpans(
  el: HTMLElement,
  type: "chars" | "words",
): HTMLSpanElement[] {
  const text = el.textContent ?? "";
  el.setAttribute("aria-label", text);
  el.innerHTML = "";

  const pieces = type === "chars" ? text.split("") : text.split(/\s+/);
  const spans: HTMLSpanElement[] = [];

  pieces.forEach((piece, i) => {
    if (!piece && type === "words") return;
    const span = document.createElement("span");
    span.style.display = "inline-block";
    span.textContent = piece === " " ? "\u00A0" : piece;
    span.setAttribute("aria-hidden", "true");
    span.classList.add(`split-${type.slice(0, -1)}`);
    el.appendChild(span);
    spans.push(span);

    // space between words
    if (type === "words" && i < pieces.length - 1) {
      el.appendChild(document.createTextNode("\u00A0"));
    }
  });

  return spans;
}

// ─── Character reveal ──────────────────────────────────────────────────────────

export interface CharRevealOptions {
  variant?: AnimationVariant;
  stagger?: number;
  duration?: number;
  ease?: string;
  scrollTrigger?: boolean | ScrollTrigger.Vars;
}

/**
 * Reveal text character-by-character.
 */
export function createCharReveal(
  target: string | HTMLElement,
  options: CharRevealOptions = {},
): gsap.core.Tween | null {
  const el =
    typeof target === "string"
      ? document.querySelector<HTMLElement>(target)
      : target;
  if (!el) return null;

  const chars = splitIntoSpans(el, "chars");
  const variant = options.variant ?? "fade-up";
  const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM["fade-up"];

  const tweenVars: gsap.TweenVars = {
    ...fromVars,
    stagger: options.stagger ?? 0.03,
    duration: options.duration ?? 0.5,
    ease: resolveEase(options.ease),
  };

  if (options.scrollTrigger) {
    tweenVars.scrollTrigger =
      typeof options.scrollTrigger === "object"
        ? options.scrollTrigger
        : { trigger: el, start: ANIMATION_CONFIG.scrollStart, once: true };
  }

  return gsap.from(chars, tweenVars);
}

// ─── Word reveal ───────────────────────────────────────────────────────────────

export interface WordRevealOptions {
  variant?: AnimationVariant;
  stagger?: number;
  duration?: number;
  ease?: string;
  scrollTrigger?: boolean | ScrollTrigger.Vars;
}

/**
 * Reveal text word-by-word.
 */
export function createWordReveal(
  target: string | HTMLElement,
  options: WordRevealOptions = {},
): gsap.core.Tween | null {
  const el =
    typeof target === "string"
      ? document.querySelector<HTMLElement>(target)
      : target;
  if (!el) return null;

  const words = splitIntoSpans(el, "words");
  const variant = options.variant ?? "fade-up";
  const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM["fade-up"];

  const tweenVars: gsap.TweenVars = {
    ...fromVars,
    stagger: options.stagger ?? 0.06,
    duration: options.duration ?? 0.6,
    ease: resolveEase(options.ease),
  };

  if (options.scrollTrigger) {
    tweenVars.scrollTrigger =
      typeof options.scrollTrigger === "object"
        ? options.scrollTrigger
        : { trigger: el, start: ANIMATION_CONFIG.scrollStart, once: true };
  }

  return gsap.from(words, tweenVars);
}

// ─── Typewriter ────────────────────────────────────────────────────────────────

export interface TypewriterOptions {
  /** Speed per character in seconds (default: 0.05) */
  speed?: number;
  /** Initial delay in seconds */
  delay?: number;
  /** Cursor character */
  cursor?: string;
  /** Whether to show a blinking cursor */
  showCursor?: boolean;
  /** Loop the typewriter */
  loop?: boolean;
  /** Callback on complete */
  onComplete?: () => void;
}

/**
 * Classic typewriter reveal effect with optional blinking cursor.
 */
export function createTypewriter(
  target: string | HTMLElement,
  options: TypewriterOptions = {},
): gsap.core.Timeline {
  const el =
    typeof target === "string"
      ? document.querySelector<HTMLElement>(target)!
      : target;

  const text = el.textContent ?? "";
  const speed = options.speed ?? 0.05;
  const cursor = options.cursor ?? "|";
  const showCursor = options.showCursor ?? true;

  el.textContent = "";

  const tl = gsap.timeline({
    delay: options.delay ?? 0,
    repeat: options.loop ? -1 : 0,
    onComplete: options.onComplete,
  });

  // Type each character
  text.split("").forEach((char, i) => {
    tl.call(
      () => {
        el.textContent = text.slice(0, i + 1) + (showCursor ? cursor : "");
      },
      undefined,
      i * speed,
    );
  });

  // Remove cursor after typing
  if (showCursor) {
    tl.call(() => {
      el.textContent = text;
    });
    // Blink cursor a few times then stop
    for (let i = 0; i < 4; i++) {
      tl.call(() => (el.textContent = text + cursor), undefined, `+=${0.4}`);
      tl.call(() => (el.textContent = text), undefined, `+=${0.4}`);
    }
  }

  return tl;
}

// ─── Text scramble ─────────────────────────────────────────────────────────────

export interface TextScrambleOptions {
  /** Characters to scramble with */
  chars?: string;
  /** Duration in seconds */
  duration?: number;
  /** Delay in seconds */
  delay?: number;
  /** Reveal direction: left-to-right or random */
  direction?: "ltr" | "random";
}

/**
 * Scramble / decode text effect — characters randomly cycle before
 * resolving to the final text.
 */
export function createTextScramble(
  target: string | HTMLElement,
  options: TextScrambleOptions = {},
): gsap.core.Timeline {
  const el =
    typeof target === "string"
      ? document.querySelector<HTMLElement>(target)!
      : target;

  const finalText = el.textContent ?? "";
  const chars =
    options.chars ??
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*";
  const duration = options.duration ?? 1.5;
  const stepsPerChar = 6;

  el.textContent = "";

  const tl = gsap.timeline({ delay: options.delay ?? 0 });

  const resolved = new Array(finalText.length).fill(false);
  const current = new Array(finalText.length).fill("");

  const totalSteps = finalText.length * stepsPerChar;
  const stepDuration = duration / totalSteps;

  for (let step = 0; step < totalSteps; step++) {
    tl.call(
      () => {
        // Resolve one character at the appropriate step
        const resolveIdx =
          options.direction === "random"
            ? Math.floor(Math.random() * finalText.length)
            : Math.floor(step / stepsPerChar);

        if (resolveIdx < finalText.length) {
          resolved[resolveIdx] =
            step >= resolveIdx * stepsPerChar + stepsPerChar - 1;
        }

        for (let i = 0; i < finalText.length; i++) {
          if (resolved[i]) {
            current[i] = finalText[i];
          } else {
            current[i] =
              finalText[i] === " "
                ? " "
                : chars[Math.floor(Math.random() * chars.length)];
          }
        }

        el.textContent = current.join("");
      },
      undefined,
      step * stepDuration,
    );
  }

  // Ensure final state
  tl.call(() => {
    el.textContent = finalText;
  });

  return tl;
}

// ─── Wave text ─────────────────────────────────────────────────────────────────

export interface WaveTextOptions {
  /** Wave height in pixels */
  height?: number;
  /** Duration of one wave cycle in seconds */
  duration?: number;
  /** Stagger between characters */
  stagger?: number;
  /** Number of repeats (-1 = infinite) */
  repeat?: number;
}

/**
 * Continuous wave animation on text characters — great for loading states
 * or playful headings.
 */
export function createWaveText(
  target: string | HTMLElement,
  options: WaveTextOptions = {},
): gsap.core.Tween | null {
  const el =
    typeof target === "string"
      ? document.querySelector<HTMLElement>(target)
      : target;
  if (!el) return null;

  const chars = splitIntoSpans(el, "chars");

  return gsap.to(chars, {
    y: -(options.height ?? 20),
    duration: options.duration ?? 0.6,
    stagger: {
      each: options.stagger ?? 0.05,
      repeat: options.repeat ?? -1,
      yoyo: true,
    },
    ease: "sine.inOut",
  });
}

// ─── Gradient text reveal ──────────────────────────────────────────────────────

export interface GradientRevealOptions {
  /** Duration in seconds */
  duration?: number;
  /** Delay in seconds */
  delay?: number;
}

/**
 * Reveal text by animating a clipping mask or background-position
 * of a gradient. Requires the element to have a gradient background
 * with `-webkit-background-clip: text`.
 */
export function createGradientReveal(
  target: gsap.DOMTarget,
  options: GradientRevealOptions = {},
): gsap.core.Tween {
  return gsap.from(target, {
    backgroundPosition: "200% center",
    duration: options.duration ?? 1.5,
    delay: options.delay ?? 0,
    ease: "power2.out",
  });
}
