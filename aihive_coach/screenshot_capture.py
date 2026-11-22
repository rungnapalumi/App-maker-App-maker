import os
import time
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8501"
OUTPUT_DIR = "/opt/cursor/artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    
    # Desktop
    desktop = browser.new_context(viewport={"width": 1920, "height": 1080})
    d = desktop.new_page()
    
    print("Desktop: Home")
    d.goto(APP_URL)
    time.sleep(4)
    d.screenshot(path=f"{OUTPUT_DIR}/01_home_desktop.png", full_page=True)
    
    print("Desktop: Coach Landing")
    d.locator("button:has-text('AIHive Coach')").click()
    time.sleep(3)
    d.screenshot(path=f"{OUTPUT_DIR}/02_coach_landing_desktop.png", full_page=True)
    
    print("Desktop: Coaching")
    d.locator("button:has-text('Start Coaching')").click()
    time.sleep(3)
    d.locator("button:has-text('Personal Growth')").click()
    time.sleep(3)
    d.locator("[data-testid='stChatInput'] textarea").fill("I want to improve my communication")
    time.sleep(1)
    d.keyboard.press("Enter")
    time.sleep(5)
    d.screenshot(path=f"{OUTPUT_DIR}/03_coaching_desktop.png", full_page=True)
    
    print("Desktop: Action Plans")
    d.locator("button:has-text('My Action Plans')").click()
    time.sleep(3)
    d.screenshot(path=f"{OUTPUT_DIR}/04_action_plans_desktop.png", full_page=True)
    
    desktop.close()
    
    # Mobile
    mobile = browser.new_context(viewport={"width": 375, "height": 812})
    m = mobile.new_page()
    
    print("Mobile: Home")
    m.goto(APP_URL)
    time.sleep(4)
    m.screenshot(path=f"{OUTPUT_DIR}/01_home_mobile.png", full_page=True)
    
    print("Mobile: Coach Landing")
    m.locator("button:has-text('AIHive Coach')").click()
    time.sleep(3)
    m.screenshot(path=f"{OUTPUT_DIR}/02_coach_landing_mobile.png", full_page=True)
    
    print("Mobile: Coaching")
    m.locator("button:has-text('Start Coaching')").click()
    time.sleep(3)
    m.locator("button:has-text('Career & Leadership')").click()
    time.sleep(3)
    m.screenshot(path=f"{OUTPUT_DIR}/03_coaching_mobile.png", full_page=True)
    
    print("Mobile: Action Plans")
    m.locator("button:has-text('My Action Plans')").click()
    time.sleep(3)
    m.screenshot(path=f"{OUTPUT_DIR}/04_action_plans_mobile.png", full_page=True)
    
    mobile.close()
    browser.close()
    
    print("\nDone! Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith('0') and f.endswith('.png'):
            print(f"  {f}")
