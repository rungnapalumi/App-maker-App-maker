import os, time
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8501"
OUT = "/opt/cursor/artifacts"

with sync_playwright() as p:
    browser = p.chromium.launch()
    mb = browser.new_context(viewport={"width": 375, "height": 812})
    m = mb.new_page()
    
    print("Capturing mobile screenshots (375px)...")
    
    print("1. Home...")
    m.goto(APP_URL)
    time.sleep(5)
    m.screenshot(path=f"{OUT}/1_home_mobile.png", full_page=True)
    
    print("2. Coach landing...")
    # Open sidebar, click, close sidebar
    m.locator("button[kind='header']").first.click()
    time.sleep(1)
    m.locator("section[data-testid='stSidebar'] button:has-text('AIHive Coach')").click()
    time.sleep(3)
    m.screenshot(path=f"{OUT}/2_coach_landing_mobile.png", full_page=True)
    
    print("3. Coaching workspace...")
    m.locator("button[kind='header']").first.click()
    time.sleep(1)
    m.locator("section[data-testid='stSidebar'] button:has-text('Start Coaching')").click()
    time.sleep(3)
    m.screenshot(path=f"{OUT}/3_coaching_workspace_mobile.png", full_page=True)
    
    print("4. Action plans...")
    m.locator("button[kind='header']").first.click()
    time.sleep(1)
    m.locator("section[data-testid='stSidebar'] button:has-text('My Action Plans')").click()
    time.sleep(3)
    m.screenshot(path=f"{OUT}/5_action_plans_mobile.png", full_page=True)
    
    mb.close()
    browser.close()
    
    print("\n✅ Mobile screenshots captured!")
