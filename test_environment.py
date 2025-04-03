#!/usr/bin/env python3
"""
Environment Test for Virtual Mouse

This script checks if all required dependencies are correctly installed
and verifies that the camera is working properly.
"""

import sys
import subprocess
import importlib
import platform

def print_status(name, status, message=""):
    """Print colored status messages"""
    if status:
        print(f"✅ {name}: OK {message}")
    else:
        print(f"❌ {name}: FAILED {message}")
    return status

def check_python_version():
    """Check if Python version is 3.6 or later"""
    version = sys.version_info
    status = version.major == 3 and version.minor >= 6
    message = f"(Python {version.major}.{version.minor}.{version.micro})"
    return print_status("Python version >= 3.6", status, message)

def check_dependency(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        else:
            version = "unknown version"
        return print_status(f"{package_name}", True, f"({version})")
    except ImportError:
        return print_status(f"{package_name}", False, "(not installed)")

def check_camera():
    """Check if the camera is accessible"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return print_status("Camera", False, "(not accessible)")
        
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return print_status("Camera", False, "(cannot capture images)")
        
        height, width = frame.shape[:2]
        cap.release()
        return print_status("Camera", True, f"(resolution: {width}x{height})")
    except Exception as e:
        return print_status("Camera", False, f"(error: {str(e)})")

def suggest_installation():
    """Suggest installation commands for missing packages"""
    print("\n" + "="*50)
    print("Installation Instructions")
    print("="*50)
    print("If any dependencies are missing, you can install them using pip:")
    print("\npip install -r requirements.txt\n")
    print("Or install individual packages:")
    print("pip install opencv-python")
    print("pip install numpy")
    print("pip install pyautogui")

def main():
    print("\n" + "="*50)
    print("Virtual Mouse Environment Test")
    print("="*50)
    
    # System information
    print(f"\nSystem: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check dependencies
    print("\nChecking dependencies:")
    cv2_ok = check_dependency("opencv-python", "cv2")
    numpy_ok = check_dependency("numpy")
    pyautogui_ok = check_dependency("pyautogui")
    
    # Check camera
    print("\nChecking hardware:")
    camera_ok = check_camera()
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    all_ok = python_ok and cv2_ok and numpy_ok and pyautogui_ok and camera_ok
    
    if all_ok:
        print("\n✅ All tests passed! Your environment is ready for the Virtual Mouse.")
        print("\nTo run the virtual mouse, execute:")
        print("python virtual_mouse.py")
        print("\nFor usage examples:")
        print("python examples.py")
    else:
        print("\n❌ Some tests failed. Please fix the issues before running the Virtual Mouse.")
        suggest_installation()

if __name__ == "__main__":
    main() 