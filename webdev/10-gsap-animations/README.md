````markdown
# 10 · GSAP Animations

Production-ready GSAP animation patterns for React / Next.js projects.
Includes reusable **hooks**, composable **patterns**, drop-in **components**,
and full-page **examples** covering scroll, text, stagger, parallax,
page-transitions, SVG morphing, magnetic effects, and more.

All animations auto-cleanup on unmount — no memory leaks.

---

## File map

```
10-gsap-animations/
├── types.ts                             Shared animation types & interfaces
├── index.ts                             Barrel exports
│
├── core/
│   ├── config.ts                        Default durations, easings, breakpoints
│   ├── register-plugins.ts              GSAP plugin registration (ScrollTrigger, Flip, etc.)
│   ├── timeline-factory.ts              fluent Timeline builder utility
│   └── cleanup.ts                       Kill / revert helpers
│
├── hooks/
│   ├── useGsap.ts                       Base GSAP hook with auto-cleanup via gsap.context()
│   ├── useTimeline.ts                   Timeline creation with dependency tracking
│   ├── useScrollTrigger.ts              ScrollTrigger hook (pin, scrub, snap)
│   ├── useAnimateOnMount.ts             Animate-in on mount with configurable preset
│   └── useSplitText.ts                  SplitText-powered text reveal hook
│
├── patterns/
│   ├── scroll-animations.ts             Scroll-driven fades, parallax, pin sections
│   ├── text-animations.ts               Char / word / line reveals, typewriter, scramble
│   ├── stagger-animations.ts            Grid, list, cascade stagger presets
│   ├── page-transitions.ts              Route-level enter / exit transitions
│   ├── svg-animations.ts                Path draw, morph, dashoffset
│   ├── parallax.ts                      Depth-layered parallax scroll
│   ├── magnetic.ts                      Cursor-attracted magnetic hover
│   └── flip-animations.ts              FLIP plugin layout animations
│
├── components/
│   ├── AnimatePresence.tsx               Mount / unmount animation wrapper
│   ├── ScrollReveal.tsx                  Scroll-triggered reveal (fade, slide, scale)
│   ├── SplitText.tsx                     Character / word / line split reveal
│   ├── MagneticButton.tsx               Magnetic hover button
│   ├── ParallaxSection.tsx              Parallax depth section
│   ├── StaggerContainer.tsx             Animate children with stagger
│   ├── AnimatedCounter.tsx              Counting number animation
│   └── MarqueeText.tsx                  Infinite horizontal scrolling text
│
└── examples/
    ├── HeroSection.tsx                   Full hero with text + image animation
    ├── NavigationMenu.tsx                Animated nav with dropdown reveals
    ├── CardGrid.tsx                      Staggered scroll-revealed card grid
    ├── ScrollShowcase.tsx                Full-page scroll-driven showcase
    └── PageTransitionLayout.tsx          App-level route transition wrapper
```

---

## Architecture

### Hook-first approach

Every animation is driven by a React hook that wraps `gsap.context()`,
guaranteeing cleanup when the component unmounts:

```tsx
import { useGsap } from "@/lib/gsap";

function Hero() {
  const container = useRef<HTMLDivElement>(null);

  useGsap(container, (ctx, contextSafe) => {
    gsap.from(".hero-title", { y: 80, opacity: 0, duration: 1 });
    gsap.from(".hero-sub", { y: 40, opacity: 0, delay: 0.3 });
  });

  return (
    <div ref={container}>
      <h1 className="hero-title">Hello</h1>
      <p className="hero-sub">World</p>
    </div>
  );
}
```

### Pattern functions

Standalone factory functions that return a GSAP tween / timeline.
Can be used inside hooks or imperatively:

```ts
import { createScrollFadeIn } from "@/lib/gsap/patterns/scroll-animations";

// Inside a useGsap callback
createScrollFadeIn(".card", { stagger: 0.1, scrub: true });
```

### Drop-in components

Pre-built React components for common effects:

```tsx
<ScrollReveal variant="fade-up" stagger={0.08}>
  <Card />
  <Card />
  <Card />
</ScrollReveal>
```

---

## Installation

```bash
npm install gsap
# GSAP 3.12+ includes ScrollTrigger, Flip, SplitText (Club) etc.
# Register the plugins you need in core/register-plugins.ts
```

### Plugin registration (run once, e.g. in layout.tsx)

```ts
import "@/lib/gsap/core/register-plugins";
```

---

## Quick recipes

### Scroll-triggered fade-in

```tsx
<ScrollReveal variant="fade-up" duration={0.8}>
  <p>I appear on scroll</p>
</ScrollReveal>
```

### Stagger grid items

```tsx
<StaggerContainer stagger={0.06} variant="fade-up">
  {items.map((item) => (
    <Card key={item.id} {...item} />
  ))}
</StaggerContainer>
```

### Magnetic button

```tsx
<MagneticButton strength={0.4}>
  <button className="btn-primary">Hover me</button>
</MagneticButton>
```

### Split text reveal

```tsx
<SplitText type="chars" stagger={0.03} trigger="scroll">
  Every character slides in individually
</SplitText>
```

### Number counter

```tsx
<AnimatedCounter from={0} to={12500} duration={2} separator="," />
```

### Page transitions

```tsx
// In layout.tsx
<PageTransitionLayout variant="slide">{children}</PageTransitionLayout>
```

---

## Customisation

All defaults live in `core/config.ts`. Override per-call or globally:

```ts
// Global override
import { ANIMATION_CONFIG } from "@/lib/gsap/core/config";
ANIMATION_CONFIG.defaultDuration = 0.6;

// Per-call override
gsap.to(el, { duration: 1.2, ease: EASINGS.smooth });
```
````
