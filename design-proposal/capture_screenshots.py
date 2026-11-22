"""
Screenshot capture script for AIHiveHQ design proposal
Captures desktop and mobile views as specified in requirements
"""
from playwright.sync_api import sync_playwright
import os
from pathlib import Path

def capture_screenshots():
    """Capture all required screenshots"""
    
    # Setup paths
    script_dir = Path(__file__).parent
    html_path = script_dir / "index.html"
    artifacts_dir = Path("/opt/cursor/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to file:// URL
    file_url = f"file://{html_path.absolute()}"
    
    print(f"📸 Capturing screenshots from: {file_url}\n")
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        # 1. Desktop 1440 - Full Page
        print("1️⃣  Desktop 1440px - Full Page...")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(file_url)
        page.wait_for_load_state("networkidle")
        page.screenshot(
            path=str(artifacts_dir / "desktop-1440-full.png"),
            full_page=True
        )
        print("   ✅ Saved: desktop-1440-full.png\n")
        page.close()
        
        # 2. Desktop 1440 - Hero Viewport
        print("2️⃣  Desktop 1440px - Hero Viewport...")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(file_url)
        page.wait_for_load_state("networkidle")
        page.screenshot(
            path=str(artifacts_dir / "desktop-1440-hero.png"),
            full_page=False
        )
        print("   ✅ Saved: desktop-1440-hero.png\n")
        page.close()
        
        # 3. Mobile 390 - Full Page
        print("3️⃣  Mobile 390px - Full Page...")
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(file_url)
        page.wait_for_load_state("networkidle")
        
        # Verify no horizontal scroll
        scroll_width = page.evaluate("document.documentElement.scrollWidth")
        client_width = page.evaluate("document.documentElement.clientWidth")
        if scroll_width > client_width:
            print(f"   ⚠️  WARNING: Horizontal scroll detected! scrollWidth={scroll_width}, clientWidth={client_width}")
        else:
            print(f"   ✅ No horizontal scroll (scrollWidth={scroll_width}, clientWidth={client_width})")
        
        page.screenshot(
            path=str(artifacts_dir / "mobile-390-full.png"),
            full_page=True
        )
        print("   ✅ Saved: mobile-390-full.png\n")
        page.close()
        
        # 4. Mobile 390 - Hero Viewport
        print("4️⃣  Mobile 390px - Hero Viewport...")
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(file_url)
        page.wait_for_load_state("networkidle")
        page.screenshot(
            path=str(artifacts_dir / "mobile-390-hero.png"),
            full_page=False
        )
        print("   ✅ Saved: mobile-390-hero.png\n")
        page.close()
        
        # 5. Mobile Nav Open
        print("5️⃣  Mobile 390px - Navigation Open...")
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(file_url)
        page.wait_for_load_state("networkidle")
        
        # Click hamburger menu
        page.click(".nav-hamburger")
        page.wait_for_timeout(500)  # Wait for animation
        
        page.screenshot(
            path=str(artifacts_dir / "mobile-390-nav-open.png"),
            full_page=False
        )
        print("   ✅ Saved: mobile-390-nav-open.png\n")
        page.close()
        
        # 6. Agent Cards Grid (Desktop, scrolled to cards)
        print("6️⃣  Desktop 1440px - Agent Cards Grid...")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(file_url)
        page.wait_for_load_state("networkidle")
        
        # Scroll to agent cards section
        page.evaluate("document.querySelector('#agents').scrollIntoView()")
        page.wait_for_timeout(500)
        
        # Take screenshot of viewport (shows grid)
        page.screenshot(
            path=str(artifacts_dir / "desktop-1440-agent-cards.png"),
            full_page=False
        )
        print("   ✅ Saved: desktop-1440-agent-cards.png\n")
        page.close()
        
        browser.close()
    
    print("✅ All screenshots captured successfully!")
    print(f"📂 Saved to: {artifacts_dir}\n")
    
    # List all screenshots with sizes
    print("📊 Screenshot Files:")
    screenshots = [
        "desktop-1440-full.png",
        "desktop-1440-hero.png",
        "mobile-390-full.png",
        "mobile-390-hero.png",
        "mobile-390-nav-open.png",
        "desktop-1440-agent-cards.png"
    ]
    
    for screenshot in screenshots:
        path = artifacts_dir / screenshot
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"   • {screenshot} ({size_kb:.1f} KB)")
        else:
            print(f"   ✗ {screenshot} (NOT FOUND)")

if __name__ == "__main__":
    capture_screenshots()
