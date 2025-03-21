"""
Development installation script
"""
import os
import subprocess
import sys


def main():
    # Get the directory containing this script
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Installing MihrabAI Agent in development mode...")
    
    # Install in development mode with test dependencies
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
            cwd=root_dir,
            check=True
        )
        print("\nInstallation successful!")
        print("\nYou can now run tests with:")
        print("  pytest")
        print("\nOr use the package in your Python code:")
        print('  from mihrabai import create_agent')
    except subprocess.CalledProcessError as e:
        print(f"\nInstallation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()