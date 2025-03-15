"""
Run Groq Example

This script sets the GROQ_API_KEY environment variable and runs the Groq example.
"""
import os
import sys
import subprocess

def main():
    # Check if GROQ_API_KEY is provided as command line argument
    if len(sys.argv) > 1:
        groq_api_key = sys.argv[1]
    else:
        # Prompt for GROQ_API_KEY if not provided
        groq_api_key = input("Please enter your GROQ_API_KEY: ")
    
    if not groq_api_key:
        print("Error: GROQ_API_KEY is required")
        return
    
    # Set environment variable
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    # Choose which example to run
    print("\nChoose an example to run:")
    print("1. Simple Groq Test")
    print("2. Advanced Handoff Example")
    print("3. Comprehensive Groq Advanced Demo")
    choice = input("Enter your choice (1, 2, or 3): ")
    
    if choice == "1":
        # Run simple Groq test
        print("\nRunning simple Groq test...")
        subprocess.run(["python", "examples/groq_test.py"])
    elif choice == "2":
        # Run advanced handoff example
        print("\nRunning advanced handoff example...")
        subprocess.run(["python", "examples/advanced_handoff_example.py"])
    elif choice == "3":
        # Run comprehensive Groq advanced demo
        print("\nRunning comprehensive Groq advanced demo...")
        subprocess.run(["python", "examples/groq_advanced_demo.py"])
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
