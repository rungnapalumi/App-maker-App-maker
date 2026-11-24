"""
Direct screenshot capture - just load and capture, no interaction
"""
import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
from PIL import Image


async def capture_homepage_mobile():
    """Capture the homepage on mobile to show initial state"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        mobile_viewport = {"width": 375, "height": 667}
        context = await browser.new_context(viewport=mobile_viewport, device_scale_factor=2)
        page = await context.new_page()
        
        print("Loading homepage on mobile viewport...")
        await page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        await asyncio.sleep(5)
        
        screenshot_path = Path(__file__).parent / "screenshots" / "mobile_homepage_round8.png"
        screenshot_path.parent.mkdir(exist_ok=True)
        
        await page.screenshot(path=str(screenshot_path), full_page=False)
        
        img = Image.open(screenshot_path)
        width, height = img.size
        print(f"✓ Homepage screenshot: {screenshot_path} ({width}x{height})")
        
        await browser.close()
        return str(screenshot_path), f"{width}x{height}"


async def main():
    print("Round 8 Direct Screenshot Capture")
    print("=" * 50)
    
    # Just capture the homepage on mobile
    path, dims = await capture_homepage_mobile()
    
    print("\n" + "=" * 50)
    print("Screenshot saved:")
    print(f"  {path}")
    print(f"  Dimensions: {dims}")
    print("\nNote: This shows the homepage. To see the AIHive Coach landing,")
    print("you would need to click 'AIHive Coach' in the sidebar navigation.")


if __name__ == "__main__":
    asyncio.run(main())
