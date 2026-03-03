/**
 * GSAP plugin registration.
 *
 * Import this file ONCE in your app entry point (e.g. layout.tsx or _app.tsx)
 * to register all required GSAP plugins globally.
 *
 * Usage:
 *   import "@/lib/gsap/core/register-plugins";
 *
 * Only uncomment the plugins you have installed / licensed.
 */

import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

// ─── Free plugins (included with gsap npm package) ─────────────────────────────

gsap.registerPlugin(ScrollTrigger);

// Uncomment as needed:
// import { Draggable }     from "gsap/Draggable";
// import { MotionPathPlugin } from "gsap/MotionPathPlugin";
// import { TextPlugin }    from "gsap/TextPlugin";
// import { Observer }      from "gsap/Observer";
// gsap.registerPlugin(Draggable, MotionPathPlugin, TextPlugin, Observer);

// ─── Club / Business plugins (requires GSAP license) ──────────────────────────

// import { ScrollSmoother } from "gsap/ScrollSmoother";
// import { SplitText }      from "gsap/SplitText";
// import { Flip }           from "gsap/Flip";
// import { MorphSVGPlugin } from "gsap/MorphSVGPlugin";
// import { DrawSVGPlugin }  from "gsap/DrawSVGPlugin";
// gsap.registerPlugin(ScrollSmoother, SplitText, Flip, MorphSVGPlugin, DrawSVGPlugin);

// ─── Global defaults ──────────────────────────────────────────────────────────

gsap.defaults({
  ease: "power2.out",
  duration: 0.8,
});

// Ensure ScrollTrigger recalculates on resize / font load
if (typeof window !== "undefined") {
  window.addEventListener("load", () => ScrollTrigger.refresh());
}

export { gsap, ScrollTrigger };
