import os, time
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8501"
OUT = "/opt/cursor/artifacts"
os.makedirs(OUT, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    
    # Desktop
    dt = browser.new_context(viewport={"width": 1920, "height": 1080})
    d = dt.new_page()
    
    print("Desktop screenshots...")
    d.goto(APP_URL)
    time.sleep(4)
    d.screenshot(path=f"{OUT}/01_home_desktop.png", full_page=True)
    
    # Use sidebar button
    d.locator("section[data-testid='stSidebar'] button:has-text('AIHive Coach')").click()
    time.sleep(3)
    d.screenshot(path=f"{OUT}/02_coach_landing_desktop.png", full_page=True)
    
    d.locator("section[data-testid='stSidebar'] button:has-text('Start Coaching')").click()
    time.sleep(3)
    d.locator("button:has-text('Personal Growth')").first.click()
    time.sleep(3)
    
    chat_input = d.locator("[data-testid='stChatInput'] textarea")
    chat_input.fill("I want to improve my communication skills")
    time.sleep(1)
    d.keyboard.press("Enter")
    time.sleep(6)
    d.screenshot(path=f"{OUT}/03_coaching_desktop.png", full_page=True)
    
    d.locator("section[data-testid='stSidebar'] button:has-text('My Action Plans')").click()
    time.sleep(3)
    d.screenshot(path=f"{OUT}/04_action_plans_desktop.png", full_page=True)
    
    dt.close()
    
    # Mobile
    mb = browser.new_context(viewport={"width": 375, "height": 812})
    m = mb.new_page()
    
    print("Mobile screenshots...")
    m.goto(APP_URL)
    time.sleep(4)
    m.screenshot(path=f"{OUT}/01_home_mobile.png", full_page=True)
    
    m.locator("section[data-testid='stSidebar'] button:has-text('AIHive Coach')").click()
    time.sleep(3)
    m.screenshot(path=f"{OUT}/02_coach_landing_mobile.png", full_page=True)
    
    m.locator("section[data-testid='stSidebar'] button:has-text('Start Coaching')").click()
    time.sleep(3)
    m.locator("button:has-text('Career & Leadership')").first.click()
    time.sleep(4)
    m.screenshot(path=f"{OUT}/03_coaching_mobile.png", full_page=True)
    
    m.locator("section[data-testid='stSidebar'] button:has-text('My Action Plans')").click()
    time.sleep(3)
    m.screenshot(path=f"{OUT}/04_action_plans_mobile.png", full_page=True)
    
    mb.close()
    browser.close()
    
    print("\nScreenshots captured:")
    for f in sorted(os.listdir(OUT)):
        if f.startswith('0') and f.endswith('.png'):
            print(f"  ✓ {f}")
