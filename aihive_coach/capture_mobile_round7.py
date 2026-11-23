"""Round 7 Mobile Screenshots"""
import os, time, subprocess
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8501"
OUT = "/opt/cursor/artifacts/round7"
os.makedirs(OUT, exist_ok=True)

def md5(f):
    return subprocess.check_output(["md5sum", f]).decode().split()[0]

with sync_playwright() as p:
    b = p.chromium.launch()
    
    # Mobile context 390x844
    m = b.new_context(viewport={"width": 390, "height": 844})
    page = m.new_page()
    page.goto(APP_URL)
    time.sleep(5)
    
    # 1. Homepage - check if Start CTA is above fold
    print("1. Homepage mobile - checking Start CTA position...")
    page.screenshot(path=f"{OUT}/homepage_first_load.png")
    print(f"   {md5(f'{OUT}/homepage_first_load.png')}")
    
    # Navigate to Coach landing
    page.locator("button").filter(has_text="Launch Coach").click(force=True)
    time.sleep(5)
    
    # 2. Coach landing - Start CTA should be visible
    print("2. Coach landing mobile - Start CTA above fold...")
    page.screenshot(path=f"{OUT}/coach_landing_above_fold.png")
    print(f"   {md5(f'{OUT}/coach_landing_above_fold.png')}")
    
    # Start a session and create an action plan
    page.locator("button").filter(has_text="Start Your First Session").first.click(force=True)
    time.sleep(5)
    page.locator("button").filter(has_text="Personal Growth").first.click(force=True)
    time.sleep(5)
    
    # Send message
    chat_input = page.locator("[data-testid='stChatInput'] textarea")
    chat_input.fill("I want to improve my confidence in meetings")
    page.keyboard.press("Enter")
    time.sleep(6)
    
    # Create action plan
    page.locator("button:has-text('Create Action Plan')").click(force=True)
    time.sleep(5)
    
    # Save action plan
    page.locator("button:has-text('Save Action Plan')").first.click(force=True)
    time.sleep(3)
    
    # Navigate to Action Plans
    # Expand Quick Navigation
    try:
        page.locator("summary:has-text('Quick Navigation')").click()
        time.sleep(1)
    except:
        pass
    
    page.locator("button:has-text('My Plans')").click(force=True)
    time.sleep(5)
    
    # 3. Action Plans with demo plan visible
    print("3. Action Plans mobile with demo plan...")
    page.screenshot(path=f"{OUT}/action_plans_with_plan_mobile.png", full_page=True)
    print(f"   {md5(f'{OUT}/action_plans_with_plan_mobile.png')}")
    
    m.close()
    b.close()
    
print("\n✅ Mobile screenshots complete")
print(f"Saved to: {OUT}")
