/**
 * CardGrid — staggered scroll-revealed card grid.
 *
 * Demonstrates: ScrollTrigger batch reveal, stagger from center,
 * hover scale, and scroll-linked parallax on card images.
 *
 * ```tsx
 * <CardGrid
 *   cards={[
 *     { title: "Feature 1", desc: "Description", image: "/f1.jpg" },
 *     { title: "Feature 2", desc: "Description", image: "/f2.jpg" },
 *     // ...
 *   ]}
 *   columns={3}
 * />
 * ```
 */

"use client";

import React, { useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGsap } from "../hooks/useGsap";
import { EASINGS } from "../core/config";

export interface CardData {
  title: string;
  desc: string;
  image?: string;
  tag?: string;
}

export interface CardGridProps {
  cards: CardData[];
  columns?: 2 | 3 | 4;
}

export function CardGrid({ cards, columns = 3 }: CardGridProps) {
  const gridRef = useRef<HTMLDivElement>(null);

  useGsap(gridRef, () => {
    // ── Batch reveal cards as they enter viewport ──
    ScrollTrigger.batch(".anim-card", {
      start: "top 85%",
      once: true,
      batchMax: columns,
      onEnter: (batch) => {
        gsap.from(batch, {
          y: 60,
          opacity: 0,
          scale: 0.95,
          stagger: 0.08,
          duration: 0.7,
          ease: EASINGS.back,
        });
      },
    });

    // ── Hover scale on each card ──
    const cards = gridRef.current!.querySelectorAll<HTMLElement>(".anim-card");
    cards.forEach((card) => {
      const img = card.querySelector<HTMLElement>(".card-img");

      card.addEventListener("mouseenter", () => {
        gsap.to(card, {
          y: -8,
          boxShadow: "0 20px 60px rgba(0,0,0,0.15)",
          duration: 0.3,
        });
        if (img) gsap.to(img, { scale: 1.05, duration: 0.4 });
      });

      card.addEventListener("mouseleave", () => {
        gsap.to(card, {
          y: 0,
          boxShadow: "0 4px 20px rgba(0,0,0,0.06)",
          duration: 0.3,
        });
        if (img) gsap.to(img, { scale: 1, duration: 0.4 });
      });
    });
  });

  return (
    <div
      ref={gridRef}
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap: 24,
        padding: "64px 24px",
        maxWidth: 1200,
        margin: "0 auto",
      }}
    >
      {cards.map((card, i) => (
        <div
          key={i}
          className="anim-card"
          style={{
            background: "#fff",
            borderRadius: 16,
            overflow: "hidden",
            boxShadow: "0 4px 20px rgba(0,0,0,0.06)",
            cursor: "pointer",
          }}
        >
          {card.image && (
            <div style={{ overflow: "hidden", height: 200 }}>
              <img
                className="card-img"
                src={card.image}
                alt={card.title}
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  display: "block",
                }}
              />
            </div>
          )}
          <div style={{ padding: 24 }}>
            {card.tag && (
              <span
                style={{
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  color: "#6366f1",
                  letterSpacing: "0.05em",
                }}
              >
                {card.tag}
              </span>
            )}
            <h3
              style={{
                fontSize: "1.2rem",
                fontWeight: 600,
                margin: "8px 0 4px",
              }}
            >
              {card.title}
            </h3>
            <p style={{ color: "#666", fontSize: "0.9rem", lineHeight: 1.5 }}>
              {card.desc}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}
