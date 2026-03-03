/**
 * ScrollShowcase — full-page scroll-driven showcase.
 *
 * Demonstrates: pinned sections, horizontal scroll, scroll-linked
 * progress bar, parallax layers, and scroll-triggered counters.
 *
 * Drop this into a page to see multiple scroll animation patterns
 * working together.
 *
 * ```tsx
 * export default function Page() {
 *   return <ScrollShowcase />;
 * }
 * ```
 */

"use client";

import React, { useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGsap } from "../hooks/useGsap";
import { EASINGS } from "../core/config";

export function ScrollShowcase() {
  const containerRef = useRef<HTMLDivElement>(null);

  useGsap(containerRef, () => {
    // ── 1. Progress bar at top ──
    gsap.to(".scroll-progress", {
      scaleX: 1,
      ease: "none",
      scrollTrigger: {
        trigger: containerRef.current,
        start: "top top",
        end: "bottom bottom",
        scrub: 0.3,
      },
    });

    // ── 2. First section: fade-in heading + parallax background ──
    gsap.from(".showcase-title", {
      y: 100,
      opacity: 0,
      duration: 1,
      ease: EASINGS.expo,
      scrollTrigger: {
        trigger: ".section-1",
        start: "top 80%",
        once: true,
      },
    });

    gsap.to(".section-1-bg", {
      y: -100,
      ease: "none",
      scrollTrigger: {
        trigger: ".section-1",
        start: "top bottom",
        end: "bottom top",
        scrub: true,
      },
    });

    // ── 3. Pinned section with step-through ──
    const steps = gsap.utils.toArray<HTMLElement>(".pin-step");
    const pinTl = gsap.timeline({
      scrollTrigger: {
        trigger: ".section-pin",
        start: "top top",
        end: `+=${steps.length * 100}%`,
        pin: true,
        scrub: 1,
        snap: 1 / (steps.length - 1),
      },
    });

    steps.forEach((step, i) => {
      if (i > 0) {
        pinTl.fromTo(
          step,
          { opacity: 0, y: 60 },
          { opacity: 1, y: 0, duration: 1 },
        );
      }
      if (i < steps.length - 1) {
        pinTl.to(step, { opacity: 0, y: -60, duration: 1 }, "+=0.5");
      }
    });

    // ── 4. Horizontal scroll section ──
    const hContainer = containerRef.current!.querySelector<HTMLElement>(
      ".h-scroll-container",
    );
    if (hContainer) {
      gsap.to(hContainer, {
        x: () => -(hContainer.scrollWidth - window.innerWidth),
        ease: "none",
        scrollTrigger: {
          trigger: ".section-horizontal",
          start: "top top",
          end: () => `+=${hContainer.scrollWidth - window.innerWidth}`,
          pin: true,
          scrub: 1,
          invalidateOnRefresh: true,
        },
      });
    }

    // ── 5. Counter section ──
    const counters = gsap.utils.toArray<HTMLElement>(".counter-value");
    counters.forEach((counter) => {
      const target = parseInt(counter.dataset.target ?? "0", 10);
      const obj = { value: 0 };
      gsap.to(obj, {
        value: target,
        duration: 2,
        ease: "power1.out",
        scrollTrigger: {
          trigger: counter,
          start: "top 80%",
          once: true,
        },
        onUpdate: () => {
          counter.textContent = Math.round(obj.value).toLocaleString();
        },
      });
    });

    // ── 6. Staggered cards at bottom ──
    ScrollTrigger.batch(".showcase-card", {
      start: "top 85%",
      once: true,
      onEnter: (batch) => {
        gsap.from(batch, {
          y: 50,
          opacity: 0,
          scale: 0.95,
          stagger: 0.08,
          duration: 0.7,
          ease: EASINGS.back,
        });
      },
    });
  });

  const sectionStyle: React.CSSProperties = {
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "column",
    position: "relative",
    padding: 48,
  };

  return (
    <div ref={containerRef}>
      {/* Progress bar */}
      <div
        className="scroll-progress"
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          height: 3,
          background: "#6366f1",
          transformOrigin: "left",
          transform: "scaleX(0)",
          zIndex: 1000,
        }}
      />

      {/* Section 1: Parallax hero */}
      <section
        className="section-1"
        style={{ ...sectionStyle, overflow: "hidden" }}
      >
        <div
          className="section-1-bg"
          style={{
            position: "absolute",
            inset: 0,
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            zIndex: 0,
          }}
        />
        <h1
          className="showcase-title"
          style={{
            fontSize: "clamp(3rem, 8vw, 6rem)",
            fontWeight: 800,
            color: "#fff",
            zIndex: 1,
            textAlign: "center",
          }}
        >
          Scroll Showcase
        </h1>
        <p
          style={{
            color: "rgba(255,255,255,0.8)",
            fontSize: "1.2rem",
            zIndex: 1,
            marginTop: 16,
          }}
        >
          Scroll down to explore animation patterns
        </p>
      </section>

      {/* Section 2: Pinned steps */}
      <section
        className="section-pin"
        style={{ ...sectionStyle, background: "#fafafa" }}
      >
        {["Step One — Discover", "Step Two — Build", "Step Three — Ship"].map(
          (text, i) => (
            <div
              key={i}
              className="pin-step"
              style={{
                position: "absolute",
                fontSize: "clamp(2rem, 5vw, 3.5rem)",
                fontWeight: 700,
                textAlign: "center",
                maxWidth: 600,
              }}
            >
              {text}
            </div>
          ),
        )}
      </section>

      {/* Section 3: Horizontal scroll */}
      <section className="section-horizontal" style={{ overflow: "hidden" }}>
        <div
          className="h-scroll-container"
          style={{ display: "flex", width: "max-content" }}
        >
          {["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7"].map(
            (color, i) => (
              <div
                key={i}
                style={{
                  width: "100vw",
                  height: "100vh",
                  background: color,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "3rem",
                  fontWeight: 700,
                  color: "#fff",
                }}
              >
                Panel {i + 1}
              </div>
            ),
          )}
        </div>
      </section>

      {/* Section 4: Counters */}
      <section style={{ ...sectionStyle, background: "#111", color: "#fff" }}>
        <h2 style={{ fontSize: "2rem", fontWeight: 700, marginBottom: 48 }}>
          By the Numbers
        </h2>
        <div style={{ display: "flex", gap: 64, textAlign: "center" }}>
          {[
            { label: "Users", target: 50000 },
            { label: "Projects", target: 12400 },
            { label: "Uptime %", target: 99 },
          ].map(({ label, target }) => (
            <div key={label}>
              <div
                className="counter-value"
                data-target={target}
                style={{ fontSize: "3rem", fontWeight: 800 }}
              >
                0
              </div>
              <div style={{ color: "#999", marginTop: 8 }}>{label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Section 5: Card grid */}
      <section style={{ ...sectionStyle, background: "#fafafa" }}>
        <h2 style={{ fontSize: "2rem", fontWeight: 700, marginBottom: 40 }}>
          Features
        </h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: 24,
            maxWidth: 900,
          }}
        >
          {Array.from({ length: 6 }).map((_, i) => (
            <div
              key={i}
              className="showcase-card"
              style={{
                background: "#fff",
                borderRadius: 12,
                padding: 32,
                boxShadow: "0 2px 12px rgba(0,0,0,0.06)",
              }}
            >
              <div
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: 12,
                  background: "#6366f1",
                  marginBottom: 16,
                }}
              />
              <h3 style={{ fontWeight: 600, fontSize: "1.1rem" }}>
                Feature {i + 1}
              </h3>
              <p
                style={{
                  color: "#666",
                  fontSize: "0.9rem",
                  marginTop: 8,
                  lineHeight: 1.5,
                }}
              >
                A brief description of this amazing feature and what it does.
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
