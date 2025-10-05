#!/usr/bin/env python3
"""
Run the Streamlit app without email prompts
"""

import subprocess
import sys
import os

def run_streamlit_app():
    """Run the Streamlit app with proper configuration"""
    
    # Set environment variables to skip prompts
    env = os.environ.copy()
    env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    env['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    # Run the app
    cmd = ['.venv\\Scripts\\streamlit', 'run', 'app.py', '--server.port', '8501']
    
    print("🚀 Starting Book Genre Classification App...")
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")
    except FileNotFoundError:
        print("❌ Virtual environment not found!")
        print("Please run setup.bat first to create the virtual environment.")

if __name__ == "__main__":
    run_streamlit_app()





