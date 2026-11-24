"""
Simple mobile screenshot capture for Round 8
Just captures the coach landing page viewport to verify Start button is above the fold
"""
import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
from PIL import Image


async def capture_mobile_coach_landing():
    """Capture mobile screenshot of coach landing page - viewport only"""
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        # Mobile viewport (375x667 is iPhone SE/8)
        mobile_viewport = {"width": 375, "height": 667}
        context = await browser.new_context(viewport=mobile_viewport, device_scale_factor=2)
        page = await context.new_page()
        
        base_url = "http://localhost:8501"
        
        print("Opening homepage...")
        await page.goto(base_url, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(3)
        
        # Navigate to AIHive Coach landing
        print("Clicking AIHive Coach button...")
        coach_button = await page.query_selector('button:has-text("AIHive Coach")')
        if coach_button:
            await coach_button.click()
            await asyncio.sleep(5)
        
        # Capture viewport screenshot (what's visible without scrolling)
        screenshot_path = Path(__file__).parent / "screenshots" / "mobile_coach_landing_above_fold_round8.png"
        screenshot_path.parent.mkdir(exist_ok=True)
        
        print(f"Capturing viewport screenshot to {screenshot_path}...")
        await page.screenshot(path=str(screenshot_path), full_page=False)
        
        img = Image.open(screenshot_path)
        width, height = img.size
        print(f"✓ Screenshot saved: {screenshot_path}")
        print(f"  Dimensions: {width}x{height} pixels")
        
        await browser.close()
        
        return {
            "mobile_coach_landing_above_fold_round8.png": f"{width}x{height}"
        }


async def setup_and_capture_action_plan():
    """Create a session, action plan, then capture the expanded action plan on mobile"""
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        # Desktop context for setup
        desktop_context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await desktop_context.new_page()
        
        base_url = "http://localhost:8501"
        
        print("\n=== Setting up action plan (desktop) ===")
        await page.goto(base_url, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(3)
        
        # Navigate to Start Coaching
        print("Navigating to coaching...")
        start_button = await page.query_selector('button:has-text("Start Coaching")')
        if start_button:
            await start_button.click()
            await asyncio.sleep(3)
        
        # Select Career category
        print("Selecting category...")
        career_button = await page.query_selector('button:has-text("Career")')
        if career_button:
            await career_button.click()
            await asyncio.sleep(3)
        
        # Send a message
        print("Sending message...")
        chat_input = await page.query_selector('textarea[aria-label*="chat"]')
        if not chat_input:
            chat_input = await page.query_selector('textarea')
        if chat_input:
            await chat_input.fill("I want to become more confident")
            await page.keyboard.press("Enter")
            await asyncio.sleep(5)
        
        # Create action plan
        print("Creating action plan...")
        create_plan_button = await page.query_selector('button:has-text("Create Action Plan")')
        if create_plan_button:
            await create_plan_button.click()
            await asyncio.sleep(3)
            
            # Submit the form
            save_button = await page.query_selector('button:has-text("Save Action Plan")')
            if save_button:
                await save_button.click()
                await asyncio.sleep(3)
        
        await desktop_context.close()
        
        print("\n=== Capturing on mobile ===")
        
        # Now switch to mobile and capture
        mobile_viewport = {"width": 375, "height": 1200}  # Taller viewport to show expanded plan
        mobile_context = await browser.new_context(viewport=mobile_viewport, device_scale_factor=2)
        mobile_page = await mobile_context.new_page()
        
        # Navigate to action plans
        await mobile_page.goto(base_url, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(3)
        
        print("Navigating to My Action Plans...")
        plans_button = await mobile_page.query_selector('button:has-text("My Action Plans")')
        if plans_button:
            await plans_button.click()
            await asyncio.sleep(3)
        
        # Expand the first action plan
        print("Expanding action plan...")
        expander = await mobile_page.query_selector('[data-testid="stExpander"] summary')
        if expander:
            await expander.click()
            await asyncio.sleep(2)
        
        # Scroll down a bit to center the expanded content
        await mobile_page.evaluate("window.scrollTo(0, 150)")
        await asyncio.sleep(1)
        
        # Capture screenshot
        screenshot_path = Path(__file__).parent / "screenshots" / "mobile_action_plan_expanded_round8.png"
        screenshot_path.parent.mkdir(exist_ok=True)
        
        print(f"Capturing expanded action plan to {screenshot_path}...")
        await mobile_page.screenshot(path=str(screenshot_path), full_page=True)
        
        img = Image.open(screenshot_path)
        width, height = img.size
        print(f"✓ Screenshot saved: {screenshot_path}")
        print(f"  Dimensions: {width}x{height} pixels")
        
        await browser.close()
        
        return {
            "mobile_action_plan_expanded_round8.png": f"{width}x{height}"
        }


if __name__ == "__main__":
    print("Mobile Screenshot Capture for Round 8")
    print("=" * 50)
    
    dimensions = {}
    
    # Goal 1: Capture coach landing page (verify Start button above fold)
    print("\n1. Capturing Mobile Coach Landing Page")
    landing_dims = asyncio.run(capture_mobile_coach_landing())
    dimensions.update(landing_dims)
    
    # Goal 2: Capture expanded action plan on mobile
    print("\n2. Creating and Capturing Expanded Action Plan")
    plan_dims = asyncio.run(setup_and_capture_action_plan())
    dimensions.update(plan_dims)
    
    print("\n" + "=" * 50)
    print("Screenshot Capture Complete!")
    print("\nFiles saved:")
    for filename, dims in dimensions.items():
        print(f"  ✓ {filename} ({dims})")
