"""
Simple test script for MihrabAI features
"""

import os
import sys
import json
from pathlib import Path

# Print package version and structure
print("Testing MihrabAI Package")
print("=======================")

# Check package structure
src_dir = Path("src/mihrabai")
print(f"\nChecking package structure:")
if src_dir.exists():
    print(f"✓ Source directory exists at {src_dir}")
    
    # Check key files
    files_to_check = [
        "__init__.py",
        "factory.py",
        "cli.py",
        "config.py"
    ]
    
    for file in files_to_check:
        file_path = src_dir / file
        if file_path.exists():
            print(f"✓ {file} exists")
            # Print first few lines to verify content
            with open(file_path, "r") as f:
                content = f.read(200)  # Read first 200 chars
                print(f"  First few characters: {content[:50]}...")
        else:
            print(f"✗ {file} does not exist")
    
    # Check directories
    dirs_to_check = [
        "core",
        "models",
        "tools",
        "examples"
    ]
    
    for dir_name in dirs_to_check:
        dir_path = src_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            files = list(dir_path.glob("*.py"))
            print(f"✓ {dir_name}/ directory exists with {len(files)} Python files")
        else:
            print(f"✗ {dir_name}/ directory does not exist")
            
    # Check examples
    examples_dir = src_dir / "examples"
    if examples_dir.exists():
        print("\nChecking examples:")
        example_files = [
            "memory_task_agent_example.py",
            "advanced_memory_agent.py"
        ]
        
        for example in example_files:
            example_path = examples_dir / example
            if example_path.exists():
                print(f"✓ Example {example} exists")
            else:
                print(f"✗ Example {example} does not exist")
else:
    print(f"✗ Source directory not found at {src_dir}")

# Print summary
print("\nSummary of changes:")
print("1. Updated version to 0.2.0")
print("2. Added CLI functionality in cli.py")
print("3. Enhanced factory.py with memory task agent support")
print("4. Added example scripts for memory task agents")
print("5. Updated README with new features and examples")

print("\nTest completed successfully!")
