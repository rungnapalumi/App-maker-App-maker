"""
Capture final screenshots for polish + deployable site
"""
from playwright.sync_api import sync_playwright
from pathlib import Path

def capture_final_screenshots():
    """Capture 5 required screenshots"""
    
    site_dir = Path(__file__).parent
    index_path = site_dir / "index.html"
    artifacts_dir = Path("/opt/cursor/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    file_url = f"file://{index_path.absolute()}"
    
    print(f"📸 Capturing final screenshots from: {file_url}\n")
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        # 1. Desktop 1440 - Hero Viewport
        print("1️⃣  Desktop 1440px - Hero Viewport...")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(file_url)
        page.wait_for_load_state("networkidle")
        page.screenshot(
            path=str(artifacts_dir / "final-desktop-1440-hero.png"),
            full_page=False
        )
        print("   ✅ Saved: final-desktop-1440-hero.png\n")
        page.close()
        
        # 2. Desktop 1440 - Full Page
        print("2️⃣  Desktop 1440px - Full Page...")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(file_url)
        page.wait_for_load_state("networkidle")
        page.screenshot(
            path=str(artifacts_dir / "final-desktop-1440-full.png"),
            full_page=True
        )
        print("   ✅ Saved: final-desktop-1440-full.png\n")
        page.close()
        
        # 3. Mobile 390 - Hero Viewport
        print("3️⃣  Mobile 390px - Hero Viewport...")
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(file_url)
        page.wait_for_load_state("networkidle")
        page.screenshot(
            path=str(artifacts_dir / "final-mobile-390-hero.png"),
            full_page=False
        )
        print("   ✅ Saved: final-mobile-390-hero.png\n")
        page.close()
        
        # 4. Mobile 390 - Full Page
        print("4️⃣  Mobile 390px - Full Page...")
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(file_url)
        page.wait_for_load_state("networkidle")
        page.screenshot(
            path=str(artifacts_dir / "final-mobile-390-full.png"),
            full_page=True
        )
        print("   ✅ Saved: final-mobile-390-full.png\n")
        page.close()
        
        # 5. Agent Grid (Desktop, scrolled to cards)
        print("5️⃣  Desktop 1440px - Agent Cards Grid...")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(file_url)
        page.wait_for_load_state("networkidle")
        
        # Scroll to agent cards section
        page.evaluate("document.querySelector('#agents').scrollIntoView()")
        page.wait_for_timeout(500)
        
        page.screenshot(
            path=str(artifacts_dir / "final-desktop-1440-agent-grid.png"),
            full_page=False
        )
        print("   ✅ Saved: final-desktop-1440-agent-grid.png\n")
        page.close()
        
        browser.close()
    
    print("✅ All screenshots captured successfully!")
    print(f"📂 Saved to: {artifacts_dir}\n")
    
    # List all screenshots with sizes
    print("📊 Screenshot Files:")
    screenshots = [
        "final-desktop-1440-hero.png",
        "final-desktop-1440-full.png",
        "final-mobile-390-hero.png",
        "final-mobile-390-full.png",
        "final-desktop-1440-agent-grid.png"
    ]
    
    for screenshot in screenshots:
        path = artifacts_dir / screenshot
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"   • {screenshot} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    capture_final_screenshots()
