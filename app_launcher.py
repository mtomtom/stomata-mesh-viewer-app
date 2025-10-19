#!/usr/bin/env python3
"""
Simple launcher for the Stomata Analysis Streamlit app.
This script will check dependencies and provide instructions.
"""

import subprocess
import sys
import os

def main():
    print("🔬 Stomata Analysis Tool")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Check for other required packages
    required = ["pandas", "numpy", "plotly", "trimesh", "scikit-learn", "scipy"]
    missing = []
    
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is available")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} not found")
    
    if missing:
        print(f"\n📦 Installing missing packages: {', '.join(missing)}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing)
    
    print("\n🚀 Starting Streamlit app...")
    print("The app will open at: http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    print("=" * 50)
    
    # Start the app
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped.")
    except FileNotFoundError:
        print("❌ streamlit_app.py not found!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()