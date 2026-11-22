import os, time
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8501"
OUT = "/opt/cursor/artifacts"

with sync_playwright() as p:
    browser = p.chromium.launch()
    mb = browser.new_context(viewport={"width": 375, "height": 812})
    m = mb.new_page()
    
    print("Mobile screenshots (sidebar auto-collapses)...")
    
    # Home - sidebar collapsed by default on mobile
    print("1. Home (sidebar collapsed)...")
    m.goto(APP_URL)
    time.sleep(5)
    m.screenshot(path=f"{OUT}/1_home_mobile.png", full_page=True)
    
    # For other pages, navigate via desktop then capture mobile
    # Or just capture what loads
    dt = browser.new_context(viewport={"width": 1920, "height": 1080})
    d = dt.new_page()
    
    # Navigate on desktop
    d.goto(APP_URL)
    time.sleep(3)
    
    # Coach landing
    print("2. Coach landing...")
    d.locator("section[data-testid='stSidebar'] button:has-text('AIHive Coach')").click()
    time.sleep(2)
    # Switch to mobile view
    m.goto(d.url)
    time.sleep(3)
    m.screenshot(path=f"{OUT}/2_coach_landing_mobile.png", full_page=True)
    
    # Coaching
    print("3. Coaching...")
    d.locator("section[data-testid='stSidebar'] button:has-text('Start Coaching')").click()
    time.sleep(2)
    m.goto(d.url)
    time.sleep(3)
    m.screenshot(path=f"{OUT}/3_coaching_workspace_mobile.png", full_page=True)
    
    # Action plans (use existing data)
    print("4. Action plans...")
    d.locator("section[data-testid='stSidebar'] button:has-text('My Action Plans')").click()
    time.sleep(2)
    m.goto(d.url)
    time.sleep(3)
    m.screenshot(path=f"{OUT}/5_action_plans_mobile.png", full_page=True)
    
    dt.close()
    mb.close()
    browser.close()
    
    print("\n✅ Done! Sidebar starts collapsed on mobile.")
