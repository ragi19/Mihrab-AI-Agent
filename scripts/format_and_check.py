#!/usr/bin/env python3
"""
Code quality check script.
Formats code with black, sorts imports with isort,
runs flake8 linting, mypy type checking, and tests.
"""
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n=== Running {description} ===")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error: {description} failed")
        sys.exit(result.returncode)
    print(f"{description} completed successfully")

def main():
    # Get repository root directory
    repo_root = Path(__file__).parent.parent
    src_dir = repo_root / "src"
    tests_dir = repo_root / "tests"
    
    # Format with black
    run_command(
        f"black {src_dir} {tests_dir}",
        "black formatter"
    )
    
    # Sort imports
    run_command(
        f"isort {src_dir} {tests_dir}",
        "isort import sorting"
    )
    
    # Run flake8
    run_command(
        f"flake8 {src_dir} {tests_dir}",
        "flake8 linting"
    )
    
    # Run mypy
    run_command(
        f"mypy {src_dir} {tests_dir}",
        "mypy type checking"
    )
    
    # Run tests with coverage
    run_command(
        f"pytest --cov={src_dir} {tests_dir} --cov-report term-missing",
        "pytest with coverage"
    )

if __name__ == "__main__":
    main()