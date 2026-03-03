/**
 * NavigationMenu — animated nav with dropdown reveals.
 *
 * Demonstrates: timeline-driven dropdown, stagger menu items,
 * contextSafe event handlers, hover-triggered animations.
 *
 * ```tsx
 * <NavigationMenu
 *   brand="Acme"
 *   links={[
 *     { label: "Products", children: [
 *       { label: "Analytics", href: "/products/analytics" },
 *       { label: "Automation", href: "/products/automation" },
 *     ]},
 *     { label: "Pricing", href: "/pricing" },
 *     { label: "Blog", href: "/blog" },
 *   ]}
 * />
 * ```
 */

"use client";

import React, { useRef, useState, useCallback } from "react";
import { gsap } from "gsap";
import { useGsap } from "../hooks/useGsap";
import { EASINGS } from "../core/config";

export interface NavLink {
  label: string;
  href?: string;
  children?: { label: string; href: string }[];
}

export interface NavigationMenuProps {
  brand: string;
  links: NavLink[];
}

export function NavigationMenu({ brand, links }: NavigationMenuProps) {
  const navRef = useRef<HTMLElement>(null);
  const [activeDropdown, setActiveDropdown] = useState<string | null>(null);

  useGsap(navRef, () => {
    // ── Entrance animation ──
    const tl = gsap.timeline({ defaults: { ease: EASINGS.smooth } });

    tl.from(".nav-brand", { x: -30, opacity: 0, duration: 0.6 });
    tl.from(
      ".nav-link",
      {
        y: -20,
        opacity: 0,
        stagger: 0.06,
        duration: 0.5,
      },
      "-=0.3",
    );
  });

  const openDropdown = useCallback((label: string) => {
    setActiveDropdown(label);

    // Animate dropdown in on next tick
    requestAnimationFrame(() => {
      const dropdown = navRef.current?.querySelector(
        `[data-dropdown="${label}"]`,
      );
      if (!dropdown) return;

      gsap.fromTo(
        dropdown,
        { opacity: 0, y: -10, scaleY: 0.9 },
        {
          opacity: 1,
          y: 0,
          scaleY: 1,
          duration: 0.3,
          ease: EASINGS.back,
          transformOrigin: "top center",
        },
      );

      // Stagger dropdown items
      gsap.from(dropdown.querySelectorAll(".dropdown-item"), {
        y: -8,
        opacity: 0,
        stagger: 0.04,
        duration: 0.25,
        delay: 0.1,
      });
    });
  }, []);

  const closeDropdown = useCallback(() => {
    if (!activeDropdown) return;

    const dropdown = navRef.current?.querySelector(
      `[data-dropdown="${activeDropdown}"]`,
    );
    if (dropdown) {
      gsap.to(dropdown, {
        opacity: 0,
        y: -6,
        duration: 0.2,
        ease: EASINGS.smooth,
        onComplete: () => setActiveDropdown(null),
      });
    } else {
      setActiveDropdown(null);
    }
  }, [activeDropdown]);

  return (
    <nav
      ref={navRef}
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "16px 32px",
        background: "#fff",
        boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}
    >
      <a
        className="nav-brand"
        href="/"
        style={{
          fontSize: "1.25rem",
          fontWeight: 700,
          textDecoration: "none",
          color: "#000",
        }}
      >
        {brand}
      </a>

      <div style={{ display: "flex", gap: 32, alignItems: "center" }}>
        {links.map((link) => (
          <div
            key={link.label}
            style={{ position: "relative" }}
            onMouseEnter={() => link.children && openDropdown(link.label)}
            onMouseLeave={() => link.children && closeDropdown()}
          >
            <a
              href={link.href ?? "#"}
              className="nav-link"
              style={{
                textDecoration: "none",
                color: "#333",
                fontSize: "0.95rem",
                fontWeight: 500,
                padding: "8px 4px",
                display: "inline-block",
              }}
            >
              {link.label}
              {link.children && " ▾"}
            </a>

            {/* Dropdown */}
            {link.children && activeDropdown === link.label && (
              <div
                data-dropdown={link.label}
                style={{
                  position: "absolute",
                  top: "100%",
                  left: -16,
                  background: "#fff",
                  borderRadius: 12,
                  boxShadow: "0 12px 40px rgba(0,0,0,0.12)",
                  padding: "12px 0",
                  minWidth: 200,
                  zIndex: 200,
                }}
              >
                {link.children.map((child) => (
                  <a
                    key={child.label}
                    href={child.href}
                    className="dropdown-item"
                    style={{
                      display: "block",
                      padding: "10px 20px",
                      textDecoration: "none",
                      color: "#333",
                      fontSize: "0.9rem",
                      transition: "background 0.15s",
                    }}
                    onMouseEnter={(e) =>
                      (e.currentTarget.style.background = "#f5f5f5")
                    }
                    onMouseLeave={(e) =>
                      (e.currentTarget.style.background = "transparent")
                    }
                  >
                    {child.label}
                  </a>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </nav>
  );
}
