"""
Screenshot capture script for AIHive Tutor
Captures desktop topic picker, mobile lesson, and mobile summary screens
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Start streamlit app in background
import subprocess
import signal

# Kill any existing streamlit processes
subprocess.run("pkill -f streamlit", shell=True)
time.sleep(1)

# Start streamlit
proc = subprocess.Popen(
    ["python3", "-m", "streamlit", "run", "aihive_coach/app.py", "--server.port=8501", "--server.headless=true"],
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

print("Waiting for Streamlit to start...")
time.sleep(10)

try:
    from playwright.sync_api import sync_playwright
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        # Screenshot 1: Desktop topic picker (1200px wide)
        print("Capturing desktop topic picker...")
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        page.goto("http://localhost:8501")
        time.sleep(3)
        
        # Navigate to tutor
        page.click("text=Start Learning")
        time.sleep(2)
        
        page.screenshot(path="tutor_desktop_topic_picker.png")
        print("✓ Captured: tutor_desktop_topic_picker.png")
        page.close()
        
        # Screenshot 2: Mobile mid-lesson with feedback (390px wide)
        print("Capturing mobile mid-lesson...")
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto("http://localhost:8501")
        time.sleep(3)
        
        # Navigate to tutor
        page.click("text=Start Learning")
        time.sleep(2)
        
        # Select a topic
        page.click("text=Photosynthesis")
        time.sleep(1)
        
        # Start learning
        page.click("text=Start Learning")
        time.sleep(3)
        
        # Answer a question
        page.fill("input[aria-label='Type your answer here...']", "Plants use sunlight to make food!")
        page.keyboard.press("Enter")
        time.sleep(3)
        
        page.screenshot(path="tutor_mobile_mid_lesson.png")
        print("✓ Captured: tutor_mobile_mid_lesson.png")
        page.close()
        
        # Screenshot 3: Mobile session summary (390px wide)
        print("Capturing mobile session summary...")
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto("http://localhost:8501")
        time.sleep(3)
        
        # Navigate to tutor
        page.click("text=Start Learning")
        time.sleep(2)
        
        # Select topic
        page.click("text=Python Programming")
        time.sleep(1)
        
        # Start learning
        page.click("text=Start Learning")
        time.sleep(3)
        
        # Quick interaction to get to summary
        # Answer a few times
        for i in range(2):
            page.fill("input[aria-label='Type your answer here...']", "Python is a programming language that is easy to learn and use.")
            page.keyboard.press("Enter")
            time.sleep(2)
        
        # End lesson
        page.click("text=End Lesson")
        time.sleep(3)
        
        page.screenshot(path="tutor_mobile_summary.png")
        print("✓ Captured: tutor_mobile_summary.png")
        page.close()
        
        browser.close()
        
    print("\n✅ All screenshots captured!")
    
finally:
    # Kill streamlit
    proc.terminate()
    proc.wait()
    subprocess.run("pkill -f streamlit", shell=True)
