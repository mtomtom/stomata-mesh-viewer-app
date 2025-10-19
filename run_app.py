#!/usr/bin/env python3
"""
Launcher script for the Stomata Analysis Streamlit app.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'trimesh',
        'scikit-learn',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def main():
    """Main launcher function."""
    print("🔬 Stomata Analysis Tool Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('streamlit_app.py'):
        print("❌ Error: streamlit_app.py not found!")
        print("Please run this script from the stomata_analysis directory.")
        return
    
    # Check dependencies
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\n💡 To install missing packages, run:")
        print("   pip install -r requirements.txt")
        print("\n   Or install individually:")
        for package in missing:
            print(f"   pip install {package}")
        return
    
    print("✅ All dependencies found!")
    
    # Launch Streamlit app
    print("\n🚀 Starting Streamlit app...")
    print("The app will open in your web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nTo stop the app, press Ctrl+C in this terminal.")
    print("=" * 40)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user.")
    except Exception as e:
        print(f"\n❌ Error starting app: {e}")
        print("💡 Try running directly: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()