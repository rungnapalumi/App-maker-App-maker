import os, time
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8501"
OUT = "/opt/cursor/artifacts"
os.makedirs(OUT, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    
    # Desktop screenshots
    print("Desktop screenshots...")
    dt = browser.new_context(viewport={"width": 1920, "height": 1080})
    d = dt.new_page()
    d.goto(APP_URL)
    time.sleep(5)
    d.screenshot(path=f"{OUT}/homepage_desktop.png", full_page=True)
    
    # Mobile screenshots (390px as requested)
    print("Mobile screenshots (390px)...")
    mb = browser.new_context(viewport={"width": 390, "height": 844})
    m = mb.new_page()
    m.goto(APP_URL)
    time.sleep(5)
    m.screenshot(path=f"{OUT}/homepage_mobile.png", full_page=True)
    
    # Product cards (desktop)
    print("Product cards...")
    d.evaluate("window.scrollTo(0, 500)")
    time.sleep(1)
    d.screenshot(path=f"{OUT}/product_cards_desktop.png", full_page=False)
    
    # Navigate to Coach landing
    print("Coach landing with example journey...")
    d.locator("button:has-text('Launch Coach')").click()
    time.sleep(3)
    d.screenshot(path=f"{OUT}/coach_landing_journey_desktop.png", full_page=True)
    
    # Start coaching
    print("Coaching workspace...")
    d.locator("button:has-text('Start Your First Session')").click()
    time.sleep(3)
    d.locator("button:has-text('Personal Growth')").first.click()
    time.sleep(3)
    d.screenshot(path=f"{OUT}/coaching_workspace_desktop.png", full_page=True)
    
    # Mobile coaching
    print("Mobile coaching workspace...")
    m.goto(d.url)
    time.sleep(3)
    m.screenshot(path=f"{OUT}/coaching_workspace_mobile.png", full_page=True)
    
    # Action plans (use existing data)
    print("Action plans mobile...")
    d.locator("button:has-text('My Action Plans')").click()
    time.sleep(3)
    m.goto(d.url)
    time.sleep(3)
    m.screenshot(path=f"{OUT}/action_plans_mobile.png", full_page=True)
    
    dt.close()
    mb.close()
    browser.close()
    
    print("\n✅ Screenshots captured!")
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.png') and ('homepage' in f or 'product' in f or 'coach' in f or 'action' in f):
            print(f"  {f}")
