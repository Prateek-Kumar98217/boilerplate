/**
 * HeroSection — full animated hero with text + image animation.
 *
 * Demonstrates: timeline composition, text splitting, parallax image,
 * staggered elements, and scroll-linked fade-out.
 *
 * ```tsx
 * <HeroSection
 *   title="Build something amazing"
 *   subtitle="GSAP-powered animations for the modern web"
 *   ctaText="Get Started"
 *   ctaHref="/docs"
 *   imageSrc="/hero.jpg"
 * />
 * ```
 */

"use client";

import React, { useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGsap } from "../hooks/useGsap";
import { MagneticButton } from "../components/MagneticButton";
import { EASINGS } from "../core/config";

export interface HeroSectionProps {
  title: string;
  subtitle?: string;
  ctaText?: string;
  ctaHref?: string;
  imageSrc?: string;
  imageAlt?: string;
}

export function HeroSection({
  title,
  subtitle,
  ctaText = "Get Started",
  ctaHref = "#",
  imageSrc,
  imageAlt = "Hero image",
}: HeroSectionProps) {
  const containerRef = useRef<HTMLElement>(null);

  useGsap(containerRef, () => {
    const tl = gsap.timeline({ defaults: { ease: EASINGS.expo } });

    // ── Animate heading word-by-word ──
    const titleEl = containerRef.current!.querySelector(".hero-title");
    if (titleEl) {
      const words = (titleEl.textContent ?? "").split(/\s+/);
      titleEl.innerHTML = words
        .map(
          (w) =>
            `<span class="hero-word" style="display:inline-block;overflow:hidden">` +
            `<span class="hero-word-inner" style="display:inline-block">${w}</span>` +
            `</span>`,
        )
        .join("\u00A0");

      tl.from(".hero-word-inner", {
        y: "110%",
        rotateX: -20,
        opacity: 0,
        duration: 1,
        stagger: 0.08,
      });
    }

    // ── Subtitle fade-in ──
    tl.from(".hero-subtitle", { y: 30, opacity: 0, duration: 0.8 }, "-=0.5");

    // ── CTA button slide up ──
    tl.from(".hero-cta", { y: 40, opacity: 0, duration: 0.7 }, "-=0.4");

    // ── Image reveal with clip-path ──
    tl.from(
      ".hero-image",
      {
        clipPath: "inset(100% 0 0 0)",
        scale: 1.15,
        duration: 1.2,
      },
      "-=0.8",
    );

    // ── Scroll-linked fade-out ──
    gsap.to(containerRef.current, {
      opacity: 0,
      y: -60,
      ease: "none",
      scrollTrigger: {
        trigger: containerRef.current,
        start: "top top",
        end: "bottom top",
        scrub: true,
      },
    });
  });

  return (
    <section
      ref={containerRef}
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          maxWidth: 1200,
          padding: "0 24px",
          display: "flex",
          gap: 64,
          alignItems: "center",
        }}
      >
        {/* Text side */}
        <div style={{ flex: 1 }}>
          <h1
            className="hero-title"
            style={{
              fontSize: "clamp(2.5rem, 6vw, 5rem)",
              fontWeight: 700,
              lineHeight: 1.1,
              margin: 0,
            }}
          >
            {title}
          </h1>

          {subtitle && (
            <p
              className="hero-subtitle"
              style={{
                fontSize: "1.25rem",
                color: "#666",
                marginTop: 24,
                maxWidth: 500,
              }}
            >
              {subtitle}
            </p>
          )}

          <div style={{ marginTop: 40 }}>
            <MagneticButton strength={0.3}>
              <a
                href={ctaHref}
                className="hero-cta"
                style={{
                  display: "inline-block",
                  padding: "16px 36px",
                  background: "#000",
                  color: "#fff",
                  borderRadius: 8,
                  fontSize: "1rem",
                  fontWeight: 600,
                  textDecoration: "none",
                }}
              >
                {ctaText}
              </a>
            </MagneticButton>
          </div>
        </div>

        {/* Image side */}
        {imageSrc && (
          <div style={{ flex: 1, overflow: "hidden", borderRadius: 16 }}>
            <img
              className="hero-image"
              src={imageSrc}
              alt={imageAlt}
              style={{ width: "100%", height: "auto", display: "block" }}
            />
          </div>
        )}
      </div>
    </section>
  );
}
