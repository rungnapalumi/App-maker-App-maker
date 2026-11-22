"""
Complete screenshot capture for all pages - polished UI
"""
import os
import time
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8501"
OUTPUT_DIR = "/opt/cursor/artifacts"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        print("Capturing all pages...")
        
        # Desktop
        print("\n=== DESKTOP SCREENSHOTS ===")
        desktop = browser.new_context(viewport={"width": 1920, "height": 1080})
        d = desktop.new_page()
        
        # Home
        print("1. Home page")
        d.goto(APP_URL)
        time.sleep(4)
        d.screenshot(path=f"{OUTPUT_DIR}/01_home_desktop.png", full_page=True)
        
        # Coach Landing
        print("2. Coach landing")
        d.locator("button:has-text('AIHive Coach')").click()
        time.sleep(3)
        d.screenshot(path=f"{OUTPUT_DIR}/02_coach_landing_desktop.png", full_page=True)
        
        # Coaching - start session
        print("3. Coaching workspace")
        d.locator("button:has-text('Start Coaching')").click()
        time.sleep(3)
        
        # Select category
        d.locator("button:has-text('Personal Growth')").click()
        time.sleep(3)
        
        # Send a message
        d.locator("[data-testid='stChatInput'] textarea").fill("I want to improve my communication skills and build confidence")
        time.sleep(1)
        d.keyboard.press("Enter")
        time.sleep(5)
        
        d.screenshot(path=f"{OUTPUT_DIR}/03_coaching_conversation_desktop.png", full_page=True)
        
        # Create action plan
        print("4. Action plan creation")
        d.locator("button:has-text('Create Action Plan')").click()
        time.sleep(3)
        d.screenshot(path=f"{OUTPUT_DIR}/04_action_plan_form_desktop.png", full_page=True)
        
        # Save plan
        d.locator("button:has-text('Save Action Plan')").first.click()
        time.sleep(3)
        
        # View action plans
        print("5. My Action Plans")
        d.locator("button:has-text('My Action Plans')").click()
        time.sleep(3)
        
        # Expand first plan
        if d.locator("summary").count() > 0:
            d.locator("summary").first.click()
            time.sleep(2)
        
        d.screenshot(path=f"{OUTPUT_DIR}/05_action_plans_list_desktop.png", full_page=True)
        
        desktop.close()
        
        # Mobile
        print("\n=== MOBILE SCREENSHOTS ===")
        mobile = browser.new_context(viewport={"width": 375, "height": 812})
        m = mobile.new_page()
        
        # Home mobile
        print("1. Home page (mobile)")
        m.goto(APP_URL)
        time.sleep(4)
        m.screenshot(path=f"{OUTPUT_DIR}/01_home_mobile.png", full_page=True)
        
        # Coach landing mobile
        print("2. Coach landing (mobile)")
        m.locator("button:has-text('AIHive Coach')").click()
        time.sleep(3)
        m.screenshot(path=f"{OUTPUT_DIR}/02_coach_landing_mobile.png", full_page=True)
        
        # Coaching mobile
        print("3. Coaching workspace (mobile)")
        m.locator("button:has-text('Start Coaching')").click()
        time.sleep(3)
        
        m.locator("button:has-text('Career & Leadership')").click()
        time.sleep(3)
        
        m.screenshot(path=f"{OUTPUT_DIR}/03_coaching_conversation_mobile.png", full_page=True)
        
        # Action plans mobile
        print("4. My Action Plans (mobile)")
        m.locator("button:has-text('My Action Plans')").click()
        time.sleep(3)
        m.screenshot(path=f"{OUTPUT_DIR}/05_action_plans_list_mobile.png", full_page=True)
        
        mobile.close()
        browser.close()
        
        print(f"\n✅ All screenshots saved to {OUTPUT_DIR}")
        print("\nCaptured files:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.endswith('.png') and f.startswith('0'):
                size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
                print(f"  - {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    capture_all()
