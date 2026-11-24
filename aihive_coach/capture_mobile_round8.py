"""
Mobile screenshot capture for Round 8
Captures the action plans page with expanded action plan on mobile viewport
"""
import asyncio
import time
from playwright.async_api import async_playwright
from pathlib import Path


async def capture_mobile_screenshots():
    """Capture mobile screenshots of the action plans page"""
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        
        # Mobile viewport (375x667 is iPhone SE/8)
        mobile_viewport = {"width": 375, "height": 667}
        context = await browser.new_context(viewport=mobile_viewport, device_scale_factor=2)
        page = await context.new_page()
        
        # Base URL
        base_url = "http://localhost:8501"
        
        print("Opening homepage...")
        await page.goto(base_url, wait_until="networkidle", timeout=15000)
        await asyncio.sleep(2)
        
        # Create a coaching session and action plan first
        print("Creating coaching session...")
        
        # Navigate to Start Coaching
        await page.click('button:has-text("Start Coaching")')
        await asyncio.sleep(2)
        
        # Select Career & Leadership category
        await page.click('button:has-text("Career & Leadership")')
        await asyncio.sleep(2)
        
        # Send a message to start the conversation
        chat_input = await page.query_selector('textarea[data-testid="stChatInput"]')
        if chat_input:
            await chat_input.fill("I want to become more confident speaking up in team meetings")
            await chat_input.press("Enter")
            await asyncio.sleep(3)
            
            # Send another message
            chat_input = await page.query_selector('textarea[data-testid="stChatInput"]')
            await chat_input.fill("I worry that my ideas aren't good enough")
            await chat_input.press("Enter")
            await asyncio.sleep(3)
        
        # Create action plan
        print("Creating action plan...")
        await page.click('button:has-text("Create Action Plan")')
        await asyncio.sleep(2)
        
        # Fill and submit the action plan form
        await page.click('button:has-text("Save Action Plan")')
        await asyncio.sleep(2)
        
        # Navigate to My Action Plans page
        print("Navigating to My Action Plans...")
        await page.click('button:has-text("My Action Plans")')
        await asyncio.sleep(2)
        
        # Expand the first action plan
        print("Expanding action plan...")
        expander = await page.query_selector('[data-testid="stExpander"]')
        if expander:
            await expander.click()
            await asyncio.sleep(1)
            
            # Scroll to show the full expanded plan
            await page.evaluate("window.scrollTo(0, 200)")
            await asyncio.sleep(1)
        
        # Capture screenshot of expanded action plan
        screenshot_path = Path(__file__).parent / "screenshots" / "mobile_action_plan_expanded_round8.png"
        screenshot_path.parent.mkdir(exist_ok=True)
        
        print(f"Capturing screenshot to {screenshot_path}...")
        await page.screenshot(path=str(screenshot_path), full_page=True)
        
        # Get actual screenshot dimensions
        from PIL import Image
        img = Image.open(screenshot_path)
        width, height = img.size
        print(f"✓ Screenshot saved: {screenshot_path}")
        print(f"  Dimensions: {width}x{height} pixels")
        
        await browser.close()
        
        return {
            "mobile_action_plan_expanded_round8.png": f"{width}x{height}"
        }


async def capture_mobile_coach_landing():
    """Capture mobile screenshot of coach landing page to verify Start button is above the fold"""
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        # Mobile viewport (375x667 is iPhone SE/8)
        mobile_viewport = {"width": 375, "height": 667}
        context = await browser.new_context(viewport=mobile_viewport, device_scale_factor=2)
        page = await context.new_page()
        
        base_url = "http://localhost:8501"
        
        print("Opening homepage for mobile coach landing...")
        await page.goto(base_url, wait_until="networkidle", timeout=15000)
        await asyncio.sleep(2)
        
        # Navigate to AIHive Coach landing
        await page.click('button:has-text("AIHive Coach")')
        await asyncio.sleep(2)
        
        # Capture screenshot without scrolling - this should show the Start button above the fold
        screenshot_path = Path(__file__).parent / "screenshots" / "mobile_coach_landing_above_fold_round8.png"
        screenshot_path.parent.mkdir(exist_ok=True)
        
        print(f"Capturing mobile coach landing to {screenshot_path}...")
        # Use viewport screenshot to capture exactly what's visible (375x667)
        await page.screenshot(path=str(screenshot_path), full_page=False)
        
        from PIL import Image
        img = Image.open(screenshot_path)
        width, height = img.size
        print(f"✓ Screenshot saved: {screenshot_path}")
        print(f"  Dimensions: {width}x{height} pixels")
        
        await browser.close()
        
        return {
            "mobile_coach_landing_above_fold_round8.png": f"{width}x{height}"
        }


if __name__ == "__main__":
    print("Starting mobile screenshot capture for Round 8...")
    print("\nMake sure the Streamlit app is running on http://localhost:8501")
    print("Run: streamlit run app.py\n")
    
    input("Press Enter when the app is ready...")
    
    dimensions = {}
    
    # Capture coach landing (Goal 1 verification)
    print("\n=== Capturing Mobile Coach Landing (Goal 1) ===")
    landing_dims = asyncio.run(capture_mobile_coach_landing())
    dimensions.update(landing_dims)
    
    # Capture action plans page (Goal 2)
    print("\n=== Capturing Mobile Action Plans Page (Goal 2) ===")
    plans_dims = asyncio.run(capture_mobile_screenshots())
    dimensions.update(plans_dims)
    
    print("\n=== Screenshot Capture Complete ===")
    print("\nScreenshots saved with dimensions:")
    for filename, dims in dimensions.items():
        print(f"  - {filename}: {dims}")
