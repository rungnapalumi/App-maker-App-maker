"""
Workforce E2E Test Fixtures
"""
import pytest
import subprocess
import time
import os
import sys
from playwright.sync_api import sync_playwright

STREAMLIT_PORT = 8501
APP_URL = f"http://localhost:{STREAMLIT_PORT}"


@pytest.fixture(scope="module")
def streamlit_app():
    """Start Streamlit app for testing"""
    # Kill any existing streamlit on this port
    subprocess.run(f"pkill -f 'streamlit.*{STREAMLIT_PORT}'", shell=True)
    time.sleep(1)
    
    # Get absolute path to aihive_coach/app.py
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    app_path = os.path.join(repo_root, "aihive_coach", "app.py")
    
    env = os.environ.copy()
    env['PYTHONPATH'] = repo_root
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app_path, 
         f"--server.port={STREAMLIT_PORT}", 
         "--server.headless=true",
         "--server.fileWatcherType=none"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=repo_root,
        env=env
    )
    
    # Wait for app to be ready
    print(f"Starting Streamlit on port {STREAMLIT_PORT}...")
    time.sleep(10)
    
    yield proc
    
    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except:
        proc.kill()
    subprocess.run(f"pkill -f 'streamlit.*{STREAMLIT_PORT}'", shell=True)


@pytest.fixture(scope="module")
def browser(streamlit_app):
    """Create Playwright browser"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture(scope="function")
def page(browser):
    """Create a new page for each test"""
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    yield page
    page.close()


@pytest.fixture(scope="module")
def app_url():
    """Get the app URL"""
    return APP_URL
