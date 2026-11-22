import os, time
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8501"
OUT = "/opt/cursor/artifacts"
os.makedirs(OUT, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    
    print("=== RUNNING FULL DEMO FLOW ===\n")
    
    # Desktop - Full flow
    dt = browser.new_context(viewport={"width": 1920, "height": 1080})
    d = dt.new_page()
    
    print("1. Navigate to home...")
    d.goto(APP_URL)
    time.sleep(5)
    d.screenshot(path=f"{OUT}/1_home_desktop.png", full_page=True)
    
    print("2. Coach landing...")
    d.locator("section[data-testid='stSidebar'] button:has-text('AIHive Coach')").click()
    time.sleep(3)
    d.screenshot(path=f"{OUT}/2_coach_landing_desktop.png", full_page=True)
    
    print("3. Start coaching session...")
    d.locator("section[data-testid='stSidebar'] button:has-text('Start Coaching')").click()
    time.sleep(3)
    d.locator("button:has-text('Personal Growth')").first.click()
    time.sleep(3)
    
    print("4. Having conversation (3 messages)...")
    # Message 1
    chat = d.locator("[data-testid='stChatInput'] textarea")
    chat.fill("I want to improve my confidence when speaking in meetings")
    d.keyboard.press("Enter")
    time.sleep(5)
    
    # Message 2
    chat.fill("I often second-guess myself and worry about what others think")
    d.keyboard.press("Enter")
    time.sleep(5)
    
    # Message 3
    chat.fill("I think I need to prepare more and practice speaking up")
    d.keyboard.press("Enter")
    time.sleep(5)
    
    d.screenshot(path=f"{OUT}/3_coaching_conversation_desktop.png", full_page=True)
    
    print("5. Creating action plan...")
    d.locator("button:has-text('Create Action Plan')").click()
    time.sleep(3)
    
    # Edit the action plan
    goal_field = d.locator("textarea").first
    goal_field.fill("Speak up confidently in team meetings by sharing at least one idea per meeting")
    time.sleep(1)
    
    d.screenshot(path=f"{OUT}/4_action_plan_form_desktop.png", full_page=True)
    
    print("6. Saving action plan...")
    d.locator("button:has-text('Save Action Plan')").first.click()
    time.sleep(3)
    
    print("7. Viewing action plans...")
    d.locator("section[data-testid='stSidebar'] button:has-text('My Action Plans')").click()
    time.sleep(3)
    
    # Expand the plan
    if d.locator("summary").count() > 0:
        d.locator("summary").first.click()
        time.sleep(2)
        
        # Add progress review
        print("8. Adding progress review...")
        d.locator("button:has-text('Review Progress')").click()
        time.sleep(2)
        
        progress_text = d.locator("textarea").last
        progress_text.fill("I spoke up twice in this week's team meeting. It felt good! Still nervous but getting easier.")
        time.sleep(1)
        
        d.locator("button:has-text('Save Progress Note')").click()
        time.sleep(3)
        
        # Re-expand to see progress
        d.locator("summary").first.click()
        time.sleep(1)
        d.locator("summary").first.click()
        time.sleep(1)
    
    d.screenshot(path=f"{OUT}/5_action_plans_with_data_desktop.png", full_page=True)
    
    dt.close()
    
    # Mobile - Key screens
    print("\n=== MOBILE SCREENSHOTS ===\n")
    mb = browser.new_context(viewport={"width": 375, "height": 812})
    m = mb.new_page()
    
    print("Mobile 1. Home...")
    m.goto(APP_URL)
    time.sleep(5)
    m.screenshot(path=f"{OUT}/1_home_mobile.png", full_page=True)
    
    print("Mobile 2. Coach landing...")
    m.locator("section[data-testid='stSidebar'] button:has-text('AIHive Coach')").click()
    time.sleep(3)
    m.screenshot(path=f"{OUT}/2_coach_landing_mobile.png", full_page=True)
    
    print("Mobile 3. Coaching (start session)...")
    m.locator("section[data-testid='stSidebar'] button:has-text('Start Coaching')").click()
    time.sleep(3)
    # Just show category selection
    m.screenshot(path=f"{OUT}/3_coaching_workspace_mobile.png", full_page=True)
    
    print("Mobile 4. Action plans...")
    m.locator("section[data-testid='stSidebar'] button:has-text('My Action Plans')").click()
    time.sleep(3)
    m.screenshot(path=f"{OUT}/5_action_plans_mobile.png", full_page=True)
    
    mb.close()
    browser.close()
    
    print("\n✅ All screenshots captured!")
    print("\nFiles created:")
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.png') and (f.startswith('1_') or f.startswith('2_') or f.startswith('3_') or f.startswith('4_') or f.startswith('5_')):
            size = os.path.getsize(os.path.join(OUT, f)) / 1024
            print(f"  {f} ({size:.1f} KB)")
