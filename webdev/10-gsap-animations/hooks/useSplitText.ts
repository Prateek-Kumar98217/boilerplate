/**
 * useSplitText — text splitting animation hook.
 *
 * Uses a lightweight manual splitting approach (no GSAP Club plugin required).
 * For GSAP Club's SplitText plugin, swap in the commented section.
 *
 * ```tsx
 * function Headline() {
 *   const ref = useSplitText<HTMLHeadingElement>({
 *     type: "chars",
 *     stagger: 0.03,
 *     variant: "fade-up",
 *     trigger: "scroll",
 *   });
 *
 *   return <h1 ref={ref}>Every character animates in.</h1>;
 * }
 * ```
 */

import { useRef, useEffect, useLayoutEffect } from "react";
import { gsap } from "gsap";
import {
  ANIMATION_CONFIG,
  VARIANT_FROM,
  resolveEase,
  prefersReducedMotion,
} from "../core/config";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import type { SplitType, AnimationVariant } from "../types";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export interface UseSplitTextOptions {
  /** Split granularity */
  type?: SplitType;
  /** Stagger interval in seconds */
  stagger?: number;
  /** Animation variant for each piece */
  variant?: AnimationVariant;
  /** When to trigger: on mount or on scroll */
  trigger?: "mount" | "scroll";
  /** Duration in seconds */
  duration?: number;
  /** Ease */
  ease?: string;
  /** ScrollTrigger start (only with trigger = "scroll") */
  scrollStart?: string;
}

// ─── Lightweight manual text splitter ──────────────────────────────────────────

function splitElement(
  el: HTMLElement,
  type: SplitType,
): {
  chars: HTMLSpanElement[];
  words: HTMLSpanElement[];
  lines: HTMLSpanElement[];
} {
  const chars: HTMLSpanElement[] = [];
  const words: HTMLSpanElement[] = [];
  const lines: HTMLSpanElement[] = [];

  const text = el.textContent ?? "";
  el.innerHTML = "";
  el.setAttribute("aria-label", text);

  const wordStrings = text.split(/\s+/);

  wordStrings.forEach((word, wi) => {
    const wordSpan = document.createElement("span");
    wordSpan.style.display = "inline-block";
    wordSpan.classList.add("split-word");

    if (type.includes("chars")) {
      word.split("").forEach((char) => {
        const charSpan = document.createElement("span");
        charSpan.style.display = "inline-block";
        charSpan.classList.add("split-char");
        charSpan.textContent = char;
        charSpan.setAttribute("aria-hidden", "true");
        wordSpan.appendChild(charSpan);
        chars.push(charSpan);
      });
    } else {
      wordSpan.textContent = word;
      wordSpan.setAttribute("aria-hidden", "true");
    }

    words.push(wordSpan);
    el.appendChild(wordSpan);

    // Add space between words
    if (wi < wordStrings.length - 1) {
      const space = document.createTextNode("\u00A0");
      el.appendChild(space);
    }
  });

  // Lines: we approximate by checking offsetTop changes
  if (type.includes("lines")) {
    let currentTop: number | null = null;
    let currentLine: HTMLSpanElement | null = null;

    Array.from(el.children).forEach((child) => {
      const top = (child as HTMLElement).offsetTop;
      if (top !== currentTop) {
        currentLine = document.createElement("span");
        currentLine.style.display = "block";
        currentLine.classList.add("split-line");
        lines.push(currentLine);
        currentTop = top;
      }
    });
  }

  return { chars, words, lines };
}

// ─── Hook ──────────────────────────────────────────────────────────────────────

export function useSplitText<T extends HTMLElement = HTMLHeadingElement>(
  options: UseSplitTextOptions = {},
): React.RefObject<T | null> {
  const ref = useRef<T | null>(null);
  const {
    type = "words",
    stagger = 0.04,
    variant = "fade-up",
    trigger = "mount",
    duration = ANIMATION_CONFIG.defaultDuration,
    ease,
    scrollStart = ANIMATION_CONFIG.scrollStart,
  } = options;

  useIsomorphicLayoutEffect(() => {
    if (!ref.current) return;
    if (prefersReducedMotion()) return;

    const el = ref.current;
    const originalHTML = el.innerHTML;

    const { chars, words } = splitElement(el, type);
    const targets = type.includes("chars") ? chars : words;

    if (targets.length === 0) return;

    const fromVars = VARIANT_FROM[variant] ?? VARIANT_FROM["fade-up"];

    const ctx = gsap.context(() => {
      const tweenVars: gsap.TweenVars = {
        ...fromVars,
        stagger,
        duration,
        ease: resolveEase(ease),
      };

      if (trigger === "scroll") {
        tweenVars.scrollTrigger = {
          trigger: el,
          start: scrollStart,
          once: true,
        };
      }

      gsap.from(targets, tweenVars);
    });

    return () => {
      ctx.revert();
      // Restore original HTML to avoid DOM pollution
      el.innerHTML = originalHTML;
    };
  }, []);

  return ref;
}
