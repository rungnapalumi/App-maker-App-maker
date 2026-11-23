# AIHiveHQ Static Site Deployment Guide

## Render Static Site Configuration

This directory (`site/`) contains a production-ready static site for AIHiveHQ's marketing homepage.

### Deployment Instructions

1. **In Render Dashboard**:
   - Click **New +** → **Static Site**
   - Connect to your GitHub repository
   - Select the branch: `design/aihivehq-homepage-proposal` (or main after merge)

2. **Build & Publish Settings**:
   ```
   Build Command: (leave empty or use: echo "No build needed")
   Publish Directory: site
   ```

3. **Environment Variables**: None required

4. **Custom Domain** (Optional):
   - Add custom domain in Render settings
   - Update DNS records to point to Render
   - Free SSL certificate included

### Technical Details

- **No build step required**: Plain HTML/CSS/JS
- **Self-contained**: All assets inline or included
- **Responsive**: Optimized for desktop (1440px) and mobile (390px)
- **Accessibility**: AA+ contrast, focus states, semantic HTML
- **Fast loading**: Static files served from Cloudflare CDN

### File Structure

```
site/
├── index.html          # Main homepage
├── 404.html           # Custom 404 page
├── favicon.svg        # SVG favicon
├── docs/              # Documentation
│   ├── render-deployment.md
│   └── render.yaml.example
└── assets/            # (Optional) Future assets directory
    ├── css/
    └── img/
```

### Links

- **Try AIHive Coach**: https://aihive-coach.onrender.com
- **Sign In**: Placeholder (not functional, tooltip shows "Coming soon")

---

## Example render.yaml (Blueprint - DO NOT use at root)

**⚠️ Note**: Do NOT create a `render.yaml` at the repository root unless you want to trigger automatic blueprint sync. This is an example for reference only.

If you do want to use Render Blueprint, create `render.yaml` at the root with this content:

```yaml
services:
  - type: web
    name: aihivehq-static-site
    env: static
    buildCommand: echo "No build needed"
    staticPublishPath: site
    headers:
      - path: /*
        name: X-Frame-Options
        value: DENY
      - path: /*
        name: X-Content-Type-Options
        value: nosniff
      - path: /*
        name: Referrer-Policy
        value: strict-origin-when-cross-origin
    routes:
      - type: rewrite
        source: /*
        destination: /index.html
```

**For manual setup (recommended for preview)**: Use the Render Dashboard as described above.

---

## Verification Checklist

- ✅ No horizontal scroll at 1440px and 390px
- ✅ All anchor links (#agents, #how-it-works, #about) work
- ✅ No console errors in browser
- ✅ Basic accessibility (alt text, contrast, focus states)
- ✅ Hero decorative card shows "Demo" status
- ✅ Mobile decorative card hidden (no overlap with hexagons)
- ✅ Agent grid: 3 columns on desktop, 2 on tablet, 1 on mobile
- ✅ Agent names don't wrap ("Developer Agent", "Marketing Agent" on one line)
- ✅ Sign In shows "Coming soon" tooltip (not functional)

---

**Last Updated**: November 23, 2026  
**Version**: Preview (Iteration 2 + Polish)
