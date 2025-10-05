#!/usr/bin/env python3
"""
Setup script for Book Genre Classification Project
This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("📚 Book Genre Classification Project Setup")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Warning: Not in a virtual environment")
        print("   Consider creating one with: python -m venv .venv")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Install basic requirements
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return
    
    if not run_command("pip install -r requirements.txt", "Installing project dependencies"):
        return
    
    # Download NLTK data
    print("\n🔄 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("✅ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the Jupyter notebook: jupyter notebook book_genre_classification.ipynb")
    print("2. Or run the Streamlit app: streamlit run app.py")
    print("3. Make sure to train the models first by running the notebook!")

if __name__ == "__main__":
    main()
