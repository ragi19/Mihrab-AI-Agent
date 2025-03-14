"""
Code formatting and quality check script
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str) -> bool:
    """Run a command and return whether it succeeded"""
    print(f"\nRunning {description}...")
    try:
        subprocess.run(command, check=True)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"{description} failed", file=sys.stderr)
        return False


def main() -> None:
    # Get project root directory
    root_dir = Path(__file__).parent.parent

    # Commands to run
    commands = [
        (["black", "."], "Code formatting"),
        (["isort", "."], "Import sorting"),
        (["flake8"], "Linting"),
        (["mypy", "src"], "Type checking"),
        (["pytest", "--cov=src"], "Tests with coverage"),
    ]

    # Run each command
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False

    if not success:
        sys.exit(1)

    print("\nAll checks passed successfully!")


if __name__ == "__main__":
    main()
