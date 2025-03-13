"""
Logging utilities and configuration
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure logging for the framework

    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (default: None, logs to console only)
        format_string: Custom log format (default: uses DEFAULT_LOG_FORMAT)
    """
    # Convert string level to integer if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    # Create formatter
    formatter = logging.Formatter(format_string or DEFAULT_LOG_FORMAT)

    # Configure root logger
    root_logger = logging.getLogger("llm_agents")
    root_logger.setLevel(level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def configure_logging(
    console_level: Union[str, int] = "INFO",
    file_level: Optional[Union[str, int]] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure logging with separate console and file levels

    Args:
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: same as console_level)
        log_file: Path to log file (default: None, logs to console only)
        format_string: Custom log format (default: uses DEFAULT_LOG_FORMAT)
    """
    # Convert string levels to integers if needed
    if isinstance(console_level, str):
        console_level = getattr(logging, console_level.upper())

    if file_level is None:
        file_level = console_level
    elif isinstance(file_level, str):
        file_level = getattr(logging, file_level.upper())

    # Create formatter
    formatter = logging.Formatter(format_string or DEFAULT_LOG_FORMAT)

    # Configure root logger
    root_logger = logging.getLogger("llm_agents")
    root_logger.setLevel(
        min(console_level, file_level if file_level is not None else console_level)
    )

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name

    Args:
        name: Logger name, will be prefixed with 'llm_agents.'

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"llm_agents.{name}")
