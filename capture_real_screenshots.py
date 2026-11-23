"""
Capture real screenshots from running Streamlit app
Run this after starting: streamlit run aihive_coach/app.py --server.port=8502
"""
import time
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8502"

def capture_screenshots():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        # Screenshot 1: Desktop topic picker (1440x900)
        print("\n1. Capturing desktop topic picker (1440x900)...")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(BASE_URL, timeout=30000)
        time.sleep(4)
        
        # Navigate to tutor
        page.get_by_role("button", name="Launch Tutor →").click()
        time.sleep(3)
        
        # Verify we're on topic picker
        assert "What do you want to learn today?" in page.content()
        
        page.screenshot(path="tutor_desktop_topic_picker.png", full_page=True)
        print("   ✓ Saved: tutor_desktop_topic_picker.png")
        page.close()
        
        # Screenshot 2-4: Mobile flow (390x844)
        print("\n2. Starting mobile lesson flow (390x844)...")
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(BASE_URL, timeout=30000)
        time.sleep(4)
        
        # Navigate to tutor via homepage button
        page.get_by_role("button", name="Launch Tutor →").click()
        time.sleep(3)
        
        # Select Photosynthesis
        page.get_by_text("🌱 Photosynthesis").click()
        time.sleep(1)
        
        # Start learning (Beginner is default)
        page.get_by_role("button", name="🚀 Start Learning").click()
        time.sleep(5)
        
        print("\n3. Answering first question (wrong answer)...")
        # Answer incorrectly
        chat_input = page.get_by_placeholder("Type your answer here...")
        chat_input.fill("I don't know")
        chat_input.press("Enter")
        time.sleep(5)
        
        # Capture mid-lesson with feedback
        page.screenshot(path="tutor_mobile_mid_lesson.png", full_page=True)
        print("   ✓ Saved: tutor_mobile_mid_lesson.png")
        
        print("\n4. Answering follow-up (correct answer)...")
        # Answer correctly
        time.sleep(2)
        chat_input = page.get_by_placeholder("Type your answer here...")
        chat_input.fill("Plants use sunlight, water, and carbon dioxide to make food through photosynthesis.")
        chat_input.press("Enter")
        time.sleep(5)
        
        print("\n5. Ending lesson...")
        # End lesson
        page.get_by_role("button", name="✅ End Lesson").click()
        time.sleep(5)
        
        # Capture summary
        page.screenshot(path="tutor_mobile_summary.png", full_page=True)
        print("   ✓ Saved: tutor_mobile_summary.png")
        
        print("\n6. Navigating to My Study Sessions...")
        # Go to sessions list
        page.get_by_role("button", name="📖 View My Study Sessions").click()
        time.sleep(4)
        
        # Capture sessions list
        page.screenshot(path="tutor_mobile_sessions_list.png", full_page=True)
        print("   ✓ Saved: tutor_mobile_sessions_list.png")
        
        page.close()
        browser.close()
    
    print("\n✅ All 4 screenshots captured!")
    print("\nGenerating MD5 checksums...")
    
    import hashlib
    for filename in ["tutor_desktop_topic_picker.png", "tutor_mobile_mid_lesson.png", 
                     "tutor_mobile_summary.png", "tutor_mobile_sessions_list.png"]:
        with open(filename, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{md5}  {filename}")


if __name__ == "__main__":
    print("=" * 70)
    print("SCREENSHOT CAPTURE SCRIPT")
    print("=" * 70)
    print("\nPrerequisites:")
    print("1. Start Streamlit: streamlit run aihive_coach/app.py --server.port=8502")
    print("2. Wait for it to be ready (~10 seconds)")
    print("3. Then run this script")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    try:
        capture_screenshots()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Streamlit is running on port 8502!")
        import traceback
        traceback.print_exc()
