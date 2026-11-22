"""
Screenshot capture script for AIHive Coach MVP
Captures all 4 pages in desktop and mobile viewports
"""
import os
import time
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8501"
OUTPUT_DIR = "/opt/cursor/artifacts"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def capture_screenshots():
    """Capture screenshots of all pages"""
    
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch()
        
        # Desktop context
        desktop = browser.new_context(viewport={"width": 1920, "height": 1080})
        desktop_page = desktop.new_page()
        
        # Mobile context
        mobile = browser.new_context(viewport={"width": 375, "height": 812})
        mobile_page = mobile.new_page()
        
        print("Capturing screenshots...")
        
        # Page 1: Home
        print("1. Home page...")
        desktop_page.goto(APP_URL)
        time.sleep(3)
        desktop_page.screenshot(path=f"{OUTPUT_DIR}/1_home_desktop.png", full_page=True)
        
        mobile_page.goto(APP_URL)
        time.sleep(3)
        mobile_page.screenshot(path=f"{OUTPUT_DIR}/1_home_mobile.png", full_page=True)
        
        # Page 2: Coach Landing
        print("2. Coach landing page...")
        desktop_page.click("text=AIHive Coach")
        time.sleep(2)
        desktop_page.screenshot(path=f"{OUTPUT_DIR}/2_coach_landing_desktop.png", full_page=True)
        
        mobile_page.click("text=AIHive Coach")
        time.sleep(2)
        mobile_page.screenshot(path=f"{OUTPUT_DIR}/2_coach_landing_mobile.png", full_page=True)
        
        # Page 3: Coaching Workspace
        print("3. Coaching workspace...")
        desktop_page.click("text=Start Coaching")
        time.sleep(2)
        desktop_page.click("text=Personal Growth")
        time.sleep(2)
        
        # Send a message
        desktop_page.fill("textarea", "I want to improve my communication skills")
        time.sleep(1)
        desktop_page.click("button:has-text('Send')")
        time.sleep(4)
        
        desktop_page.screenshot(path=f"{OUTPUT_DIR}/3_coaching_workspace_desktop.png", full_page=True)
        
        mobile_page.click("text=Start Coaching")
        time.sleep(2)
        mobile_page.click("text=Career & Leadership")
        time.sleep(2)
        mobile_page.screenshot(path=f"{OUTPUT_DIR}/3_coaching_workspace_mobile.png", full_page=True)
        
        # Page 4: Create Action Plan
        print("4. Action plan...")
        desktop_page.click("text=Create Action Plan")
        time.sleep(3)
        desktop_page.screenshot(path=f"{OUTPUT_DIR}/4_action_plan_desktop.png", full_page=True)
        
        # Save action plan on desktop
        desktop_page.click("button:has-text('Save Action Plan')")
        time.sleep(3)
        
        # View action plans page
        desktop_page.click("text=My Action Plans")
        time.sleep(2)
        desktop_page.screenshot(path=f"{OUTPUT_DIR}/5_action_plans_list_desktop.png", full_page=True)
        
        # Mobile action plans
        mobile_page.click("text=My Action Plans")
        time.sleep(2)
        mobile_page.screenshot(path=f"{OUTPUT_DIR}/5_action_plans_list_mobile.png", full_page=True)
        
        print(f"\n✅ Screenshots saved to {OUTPUT_DIR}")
        print("\nCaptured:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.endswith('.png'):
                print(f"  - {f}")
        
        # Cleanup
        desktop.close()
        mobile.close()
        browser.close()


if __name__ == "__main__":
    capture_screenshots()
