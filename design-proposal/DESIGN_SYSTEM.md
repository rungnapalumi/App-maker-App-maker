# AIHiveHQ Homepage Design System

## Overview

This design system defines the visual language and technical foundation for the AIHiveHQ homepage. It draws inspiration from Linear's refined dark surfaces and Vercel's clean spatial hierarchy while establishing a unique AIHiveHQ identity.

---

## Color Tokens

### Primary Palette
```
Navy Blacks (Base)
--navy-950: #0a0e1a    // Deepest background
--navy-900: #0f1629    // Primary surface
--navy-800: #1a2238    // Elevated surface
--navy-700: #252f47    // Interactive surface

Charcoal Blacks (Alternative)
--charcoal-900: #0d1117 // GitHub-inspired
--charcoal-800: #161b22
--charcoal-700: #1f2937
```

### Accent Colors
```
Electric Blue
--blue-500: #3b82f6    // Primary accent
--blue-600: #2563eb    // Hover/active state

Violet
--violet-500: #8b5cf6  // Secondary accent
--violet-600: #7c3aed  // Gradient component
```

### Neutral Grays
```
--slate-50: #f8fafc    // Highest contrast text
--slate-100: #f1f5f9
--slate-200: #e2e8f0   // Primary text
--slate-300: #cbd5e1   // Secondary text
--slate-400: #94a3b8   // Tertiary text
--slate-500: #64748b   // Muted text
--slate-600: #475569
--slate-700: #334155
```

### Semantic Colors
```
Success: #10b981 (Demo badge)
Warning: #f59e0b
Info: #8b5cf6 (Coming Soon badge)
Muted: #64748b (Future Vision badge)
```

### Gradients
```css
--gradient-primary: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%)
--gradient-surface: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(124, 58, 237, 0.08) 100%)
--gradient-hover: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)
```

---

## Typography Scale

### Font Family
```
Primary: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif
```

### Type Scale
```
--text-xs: 0.75rem    // 12px - Badges, tiny labels
--text-sm: 0.875rem   // 14px - Body small, captions
--text-base: 1rem     // 16px - Body text
--text-lg: 1.125rem   // 18px - Large body
--text-xl: 1.25rem    // 20px - Card titles
--text-2xl: 1.5rem    // 24px - Section subtitles
--text-3xl: 1.875rem  // 30px - Mobile headings
--text-4xl: 2.25rem   // 36px - Desktop headings
--text-5xl: 3rem      // 48px - Large headings
--text-6xl: 3.75rem   // 60px - Hero title
```

### Font Weights
```
Regular: 400  // Body text
Medium: 500   // Navigation, links
Semibold: 600 // Card titles, buttons
Bold: 700     // Headings
```

### Line Heights
```
Tight: 1.1    // Large headings
Normal: 1.6   // Body text
Relaxed: 1.8  // Long-form content
```

---

## Spacing Scale

Based on 0.25rem (4px) increments:

```
--space-1: 0.25rem   // 4px
--space-2: 0.5rem    // 8px
--space-3: 0.75rem   // 12px
--space-4: 1rem      // 16px
--space-5: 1.25rem   // 20px
--space-6: 1.5rem    // 24px
--space-8: 2rem      // 32px
--space-10: 2.5rem   // 40px
--space-12: 3rem     // 48px
--space-16: 4rem     // 64px
--space-20: 5rem     // 80px
--space-24: 6rem     // 96px
```

### Application
- Component padding: space-4 to space-8
- Section padding: space-16 to space-24
- Card gaps: space-6
- Text margins: space-3 to space-6

---

## Border Radii

```
--radius-sm: 0.375rem  // 6px - Badges
--radius-md: 0.5rem    // 8px - Buttons, inputs
--radius-lg: 0.75rem   // 12px - Cards, containers
--radius-xl: 1rem      // 16px - Large cards
--radius-2xl: 1.5rem   // 24px - Feature sections
```

---

## Shadows

```css
--shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05)
--shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 
             0 2px 4px -1px rgba(0, 0, 0, 0.06)
--shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 
             0 4px 6px -2px rgba(0, 0, 0, 0.05)
--shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 
             0 10px 10px -5px rgba(0, 0, 0, 0.04)
--shadow-glow: 0 0 20px rgba(37, 99, 235, 0.3)
```

### Application
- Cards: shadow-md (default), shadow-xl (hover)
- Buttons: shadow-lg (primary), none (secondary)
- Modals: shadow-xl
- Accent glow: shadow-glow (hover states)

---

## Component Library

### Navigation Bar

**Desktop**
- Height: 72px (with padding)
- Background: rgba(10, 14, 26, 0.8) + backdrop-blur(12px)
- Border: 1px solid rgba(255, 255, 255, 0.05)
- Logo: text-xl, gradient-primary
- Links: text-sm, slate-300, hover slate-50
- CTA: bordered box, blue-500 accent

**Mobile**
- Hamburger icon: 3 lines, 24px width, 2px height
- Slide-down menu: navy-900 background
- Links: text-base, full-width touch targets

### Buttons

**Primary Button**
```
Background: gradient-primary
Color: white
Padding: space-4 space-8
Border-radius: radius-lg
Font: text-base, weight 600
Shadow: shadow-lg
Hover: gradient-hover, translateY(-2px), shadow-xl + shadow-glow
```

**Secondary Button**
```
Background: rgba(255, 255, 255, 0.05)
Border: 1px solid rgba(255, 255, 255, 0.1)
Color: slate-200
Padding: space-4 space-8
Border-radius: radius-lg
Font: text-base, weight 600
Hover: background rgba(255, 255, 255, 0.1), border rgba(255, 255, 255, 0.2)
```

**Disabled State**
- Opacity: 0.5
- Cursor: not-allowed
- No hover effects

### Status Badges

**Demo Badge**
```
Background: rgba(16, 185, 129, 0.15)
Border: 1px solid rgba(16, 185, 129, 0.3)
Color: #10b981
Text: uppercase, text-xs, weight 600, letter-spacing 0.05em
Padding: space-1 space-3
Border-radius: radius-sm
```

**Coming Soon Badge**
```
Background: rgba(139, 92, 246, 0.15)
Border: 1px solid rgba(139, 92, 246, 0.3)
Color: #8b5cf6
(Same text styling as Demo)
```

**Future Vision Badge**
```
Background: rgba(100, 116, 139, 0.15)
Border: 1px solid rgba(100, 116, 139, 0.3)
Color: slate-400
(Same text styling as Demo)
```

### Agent Card

**Structure**
```
Width: minmax(320px, 1fr) in grid
Background: gradient-surface
Border: 1px solid rgba(255, 255, 255, 0.08)
Border-radius: radius-xl
Padding: space-8
Display: flex column
Height: 100% (equal height in grid)
```

**States**
```
Default: border rgba(255, 255, 255, 0.08)
Hover: 
  - border rgba(37, 99, 235, 0.3)
  - translateY(-4px)
  - shadow-xl
  - transition: all 0.3s
```

**Content Layout**
- Icon: 48px circle, gradient-primary, space-3 margin bottom
- Title: text-xl, weight 600, slate-50, space-3 margin bottom
- Description: text-base, slate-400, line-height 1.6, flex-grow
- Badge: positioned top-right
- Action: margin-top space-4, full-width button

### Section Header

```
Text-align: center
Margin-bottom: space-16

Title: 
  - text-4xl (desktop), text-3xl (mobile)
  - weight 700
  - color slate-50
  - margin-bottom space-4

Description:
  - text-lg
  - color slate-400
  - max-width 700px, centered
```

### Footer

**Structure**
```
Background: navy-900
Border-top: 1px solid rgba(255, 255, 255, 0.05)
Padding: space-16 (top/bottom), space-8 (bottom)
Grid: repeat(auto-fit, minmax(200px, 1fr))
Gap: space-8
```

**Content**
- Section titles: text-sm, weight 600, slate-50, uppercase, letter-spacing 0.05em
- Links: text-sm, slate-400, hover slate-200
- Bottom: slate-500, text-sm, centered, border-top

---

## Icon System

**Style**
- Line icons (stroke, no fill)
- Stroke width: 2px (default), 2.5px (emphasis)
- Size: 24px × 24px
- Stroke linecap: round
- Stroke linejoin: round

**Used Icons**
- Arrow right: Navigation, CTAs
- Search: Discover step
- Checkbox: Delegate step
- Check circle: Achieve step
- Brain/User: Coach agent
- Book/Lines: Tutor agent
- Dollar sign: Finance agent
- Code brackets: Developer agent
- Play circle: Marketing agent
- Network/hierarchy: AI Workforce

**Color**
- Default: currentColor (inherits from parent)
- In gradients: white
- Muted states: slate-400

---

## Motion & Transitions

### Duration
```
Fast: 0.2s     // Hover, focus
Default: 0.3s  // Cards, buttons
Slow: 0.5s     // Page transitions
```

### Easing
```
Default: ease (cubic-bezier(0.25, 0.1, 0.25, 1))
Smooth: ease-in-out
Bounce: cubic-bezier(0.68, -0.55, 0.265, 1.55)
```

### Common Animations
```css
/* Button hover */
transform: translateY(-2px);
box-shadow: shadow-xl, shadow-glow;

/* Card hover */
transform: translateY(-4px);
border-color: change;
box-shadow: shadow-xl;

/* SVG pulse */
animate attributeName="r" values="8;12;8" dur="2s" repeatCount="indefinite"

/* Dashed line animation */
animate stroke-dashoffset values="0;8" dur="1-2s" repeatCount="indefinite"
```

---

## Responsive Breakpoints

```
Mobile: < 768px
Tablet: 768px - 1024px
Desktop: 1024px - 1440px
Wide: > 1440px
```

### Mobile Adaptations (< 768px)
- Navigation: Hamburger menu
- Hero title: text-4xl → text-3xl
- Hero padding: reduced by 33%
- Agent grid: 1 column
- Steps grid: 1 column
- Footer grid: 1 column
- Section padding: space-24 → space-16

---

## Accessibility Guidelines

### Color Contrast
- Body text (slate-200 on navy-950): 12.6:1 (AAA)
- Secondary text (slate-400 on navy-950): 5.8:1 (AA)
- Primary button: white on blue-600 (AAA)

### Focus States
```css
outline: 2px solid var(--blue-500);
outline-offset: 2px;
```

### Semantic HTML
- Proper heading hierarchy (h1 → h2 → h3)
- Nav wrapped in `<nav>` with aria-label
- Buttons have descriptive text or aria-labels
- Links have meaningful text, not "click here"
- All interactive elements keyboard accessible

### ARIA Labels
- Logo: aria-label="AIHiveHQ Home"
- Hamburger: aria-label="Toggle navigation menu"
- Icon-only buttons: aria-label describing action

---

## Technical Limitations of Streamlit

### Why Streamlit Cannot Achieve This Design Quality

#### 1. **Limited CSS Control**
- Streamlit uses predefined components with internal styling
- Custom CSS via `st.markdown(unsafe_allow_html=True)` is fragile and breaks on updates
- Cannot easily override deeply nested component styles
- Global CSS affects all instances, making precise targeting difficult

#### 2. **No Native Dark Mode Control**
- Streamlit's dark theme is auto-detected from browser/OS
- Cannot force a specific dark palette (navy-950 background)
- Theme colors are limited to config.toml presets
- Cannot achieve Linear/Vercel level of refined dark surfaces

#### 3. **Rigid Layout System**
- Column-based layout (`st.columns()`) is inflexible
- No CSS Grid or advanced flexbox control
- Equal-height cards require hacky workarounds
- Cannot achieve pixel-perfect spacing control

#### 4. **Component Limitations**
- Buttons have fixed Streamlit styling (rounded, specific padding)
- Cannot create gradient buttons with custom hover effects
- No native badge components
- Cards require extensive HTML injection, which is unmaintainable

#### 5. **No Animation Support**
- No CSS animation/transition control
- Hover effects limited to what `st.markdown` can inject
- Cannot create the hexagon network with animated connections
- No smooth scroll, parallax, or interactive effects

#### 6. **Typography Constraints**
- Font family limited to config.toml settings
- Cannot use Google Fonts reliably across all components
- Text gradient effects require HTML injection per element
- Inconsistent line-height and letter-spacing across components

#### 7. **Responsive Design Issues**
- Mobile responsiveness is basic and automatic
- Cannot implement custom hamburger navigation
- Breakpoint control is limited
- Cannot hide/show elements based on viewport width programmatically

#### 8. **Navigation Bar Limitations**
- No fixed/sticky navigation out of the box
- Cannot create backdrop-blur effect
- Sidebar is the primary navigation model (conflicts with design)
- Top nav requires HTML injection, which doesn't integrate with Streamlit's routing

#### 9. **Icon System**
- Must use emoji or external icon libraries (Font Awesome, etc.)
- Cannot inline custom SVG icons within Streamlit components
- No consistent stroke-width control
- Animated SVG requires HTML injection

#### 10. **State Management Conflicts**
- Streamlit reruns entire script on interaction
- Difficult to maintain UI state (e.g., mobile nav open/closed)
- CTA buttons trigger page reload, breaking smooth navigation
- Cannot implement client-side routing for smooth scrolling

---

## Recommended Frontend Architecture

### Option 1: Static Site + Streamlit Backend (Recommended)

**Public Marketing Site (This Design)**
- **Framework**: Vite + React or plain HTML/CSS/JS
- **Hosting**: Render Static Site (free tier)
- **Domain**: `aihivehq.com` or `aihivehq.onrender.com`
- **Features**: 
  - Full design control (Linear/Vercel quality)
  - Fast loading (static assets on CDN)
  - SEO-friendly
  - Easy to update (edit HTML/CSS/JS)

**Streamlit Coaching App (Existing)**
- **Keep at**: `aihive-coach.onrender.com`
- **Purpose**: Functional workspace for AIHive Coach
- **Access**: Link from "Try AIHive Coach" button on marketing site
- **User Flow**: Public site → Sign In (future) → Coach workspace

**Pros**
- Best of both worlds: premium marketing + functional app
- Marketing site is fast, SEO-friendly, and visually premium
- Streamlit app remains focused on functionality, not marketing
- Easy to maintain separately
- No paid services required (both on Render free tier)

**Cons**
- Two separate deployments to manage
- Need to implement user authentication later (Sign In placeholder for now)
- Cross-domain session management (future consideration)

---

### Option 2: Next.js Full-Stack (Future Scaling)

**When to Consider**
- Need server-side rendering for SEO
- Want unified authentication across marketing + app
- Ready to migrate Streamlit functionality to custom UI
- Need API routes for backend logic

**Tech Stack**
- Next.js 14+ (App Router)
- Tailwind CSS (matches this design system)
- Vercel hosting (free tier) or Render Web Service
- Python backend API (FastAPI) for AI logic

**Pros**
- Full control over UI/UX across all pages
- Unified codebase
- Better performance and SEO
- Professional routing and navigation

**Cons**
- Requires rewriting Streamlit app functionality
- More complex to maintain
- Steeper learning curve
- May need paid hosting for backend API

---

### Option 3: Keep Streamlit Only (Not Recommended)

**Reality Check**
- Current Streamlit app is functional but visually basic
- Cannot achieve Linear/Vercel quality design in Streamlit
- Marketing and product workspace have different UX needs
- Streamlit is best for internal tools, not public marketing

**Verdict**
- Not recommended for public-facing marketing site
- Keep Streamlit for Coach workspace functionality only

---

## Recommended Approach: Hybrid Architecture

### Phase 1: Static Marketing Site (This Proposal)
1. Deploy `design-proposal/index.html` as static site on Render
2. Free tier, no backend needed
3. "Try AIHive Coach" button links to existing Streamlit app
4. "Sign In" is placeholder (not functional)
5. Zero migration risk to existing app

### Phase 2: Connect to Streamlit Backend (Future)
1. Keep Streamlit at `aihive-coach.onrender.com`
2. Implement simple authentication (optional)
3. Marketing site → Sign In → Redirect to Streamlit app
4. Session management via cookies/JWT

### Phase 3: Future Agents (Future)
1. Build new agents in Next.js or React
2. Marketing site remains static
3. Each agent gets its own subdomain or path
4. Unified platform emerges gradually

---

## Hosting on Render (Free Tier)

### Static Site Setup

**1. Create New Static Site**
- In Render Dashboard: New → Static Site
- Connect to GitHub repo
- Build command: (none needed for plain HTML)
- Publish directory: `design-proposal`
- Auto-deploy: Yes

**2. Custom Domain (Optional)**
- Add custom domain in Render settings
- Update DNS records to point to Render
- Free SSL certificate included

**3. Environment Variables**
- None needed for static site

**4. Build Time**
- Instant (no build step)
- Served from Cloudflare CDN

### Linking to Streamlit App

**Current Setup**
- Streamlit app: `https://aihive-coach.onrender.com`
- Marketing site: `https://aihivehq.onrender.com` (or custom domain)

**Button Links**
```html
<!-- Try AIHive Coach button -->
<a href="https://aihive-coach.onrender.com" class="btn btn-secondary">
  Try AIHive Coach
</a>

<!-- Agent card CTA -->
<a href="https://aihive-coach.onrender.com" class="btn btn-primary btn-agent">
  Try Demo
</a>
```

**Future Authentication**
- Implement OAuth or JWT-based auth
- Marketing site "Sign In" → Auth service → Redirect to Streamlit with token
- Streamlit validates token and loads user session

---

## Design Checklist

### Visual Quality
- ✅ Premium dark surfaces (navy-950, navy-900)
- ✅ Refined gradients (blue-600 → violet-600)
- ✅ Consistent typography (Inter, proper scale)
- ✅ Polished hover effects (transform, shadow, glow)
- ✅ One consistent icon set (line icons, inline SVG)
- ✅ No emoji branding
- ✅ Own recognizable identity (not a Linear/Vercel copy)

### Content Requirements
- ✅ Brand positioning: "Your AI Workforce. One Platform."
- ✅ Hero with network visualization (hexagon hive motif)
- ✅ Six agent cards with accurate status badges
- ✅ How It Works (Discover, Delegate, Achieve)
- ✅ Why AIHiveHQ (practical AI messaging)
- ✅ Founder section (Rungnapa Mukdasanit, Founder)
- ✅ No invented metrics, testimonials, or partners

### Technical Quality
- ✅ Premium at 1440px
- ✅ Excellent at 390px
- ✅ No horizontal scroll on mobile
- ✅ Hamburger navigation on mobile
- ✅ Color contrast AA+ (most AAA)
- ✅ Focus states on all interactive elements
- ✅ Semantic HTML with proper headings
- ✅ ARIA labels where needed

---

## Conclusion

This design system establishes AIHiveHQ's visual identity: premium, modern, and purpose-built for a multi-agent AI platform. The recommended architecture (static marketing site + Streamlit backend) balances design quality with development pragmatism, allowing the platform to grow from one agent (Coach) to a full AI workforce without compromising user experience.

**Key Takeaway**: Streamlit is excellent for functional workspaces but cannot achieve the visual refinement expected of a modern SaaS marketing site. A hybrid approach delivers premium first impressions while keeping the existing Coach app fully functional.
