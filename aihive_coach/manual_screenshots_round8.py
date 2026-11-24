"""
Manual screenshot instructions for Round 8
This script provides instructions for manually capturing the required screenshots
"""

instructions = """
ROUND 8 MOBILE SCREENSHOTS - Manual Capture Instructions
========================================================

GOAL 1: Mobile Start CTA Above the Fold (375x667 viewport)
-----------------------------------------------------------
1. Open Chrome DevTools (F12)
2. Click the device toolbar icon (Ctrl+Shift+M)
3. Set device to "iPhone SE" or custom viewport: 375 x 667
4. Navigate to: http://localhost:8501
5. Click "AIHive Coach" in the sidebar
6. Take a screenshot of the VIEWPORT ONLY (what's visible without scrolling)
7. Save as: screenshots/mobile_coach_landing_above_fold_round8.png

Expected result: The "🚀 Start Your First Session" button should be visible
without any scrolling on a 375x667 viewport.


GOAL 2: Expanded Action Plan on Mobile (375px wide)
----------------------------------------------------
1. Keep DevTools open with 375px width (height can be taller, e.g., 1200px)
2. Navigate to: http://localhost:8501
3. Click "Start Coaching" in sidebar
4. Select any category (e.g., "Career & Leadership")
5. Type a message like: "I want to become more confident"
6. Wait for response, then click "Create Action Plan"
7. Click "Save Action Plan" (accept defaults)
8. Click "My Action Plans" in sidebar
9. EXPAND the action plan by clicking on it
10. Scroll to show the full expanded action plan
11. Take a FULL PAGE screenshot (not just viewport)
12. Save as: screenshots/mobile_action_plan_expanded_round8.png

Expected result: A mobile-width screenshot showing an expanded action plan
with all fields visible (Goal, Key Insight, Next Action, Target Date, 
Success Indicator).


Alternative: Use Playwright Script
----------------------------------
If manual capture is not feasible, use the Python script below.
"""

playwright_script = '''
# Simple Playwright script - run this if manual capture doesn't work
import asyncio
from playwright.async_api import async_playwright
from pathlib import Path

async def capture():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # headless=False to see what's happening
        
        # Mobile viewport
        context = await browser.new_context(viewport={"width": 375, "height": 667})
        page = await context.new_page()
        
        # Navigate and wait
        await page.goto("http://localhost:8501")
        await asyncio.sleep(5)
        
        # Capture coach landing (Goal 1)
        input("Navigate to AIHive Coach page, then press Enter...")
        await page.screenshot(path="screenshots/mobile_coach_landing_above_fold_round8.png", full_page=False)
        print("✓ Saved: mobile_coach_landing_above_fold_round8.png")
        
        # Capture action plan (Goal 2)
        input("Create and expand an action plan, then press Enter...")
        await page.screenshot(path="screenshots/mobile_action_plan_expanded_round8.png", full_page=True)
        print("✓ Saved: mobile_action_plan_expanded_round8.png")
        
        await browser.close()

asyncio.run(capture())
'''

if __name__ == "__main__":
    print(instructions)
    print("\n" + "="*60)
    print("PLAYWRIGHT SCRIPT")
    print("="*60)
    print(playwright_script)
    
    print("\n\nTo generate the screenshots, either:")
    print("1. Follow the manual instructions above, OR")
    print("2. Save the Playwright script to a file and run it")
