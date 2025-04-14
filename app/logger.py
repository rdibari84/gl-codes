import logging
import sys
from pathlib import Path
from typing import Optional

# Define log levels
TRACE_LEVEL = 5  # Custom level below DEBUG


def setup_logger(
    name: str = "app",
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    silence_others: bool = True,
) -> logging.Logger:
    """
    Configure a logger with consistent formatting that optionally silences other loggers.

    Args:
        name: The name of the logger
        log_level: The minimum log level to display
        log_file: Optional file path to write logs to
        silence_others: Whether to silence all other loggers

    Returns:
        The configured logger
    """
    # Register TRACE level if needed
    if not hasattr(logging, "TRACE"):
        logging.addLevelName(TRACE_LEVEL, "TRACE")
        setattr(logging, "TRACE", TRACE_LEVEL)

    # Get or create the logger
    logger = logging.getLogger(name)

    # Don't propagate to the root logger
    logger.propagate = False

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Set the log level
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-1s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file is not None:
        # Ensure parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Silence other loggers if requested
    if silence_others:
        silence_other_loggers(exclude=[name])

    return logger


def silence_other_loggers(exclude: list = None) -> None:
    """
    Silence all loggers except those in the exclude list.

    Args:
        exclude: List of logger names to exclude from silencing
    """
    exclude = exclude or []

    # Silence root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    # Get all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        # Skip excluded loggers
        if any(
            logger_name == excluded or logger_name.startswith(f"{excluded}.")
            for excluded in exclude
        ):
            continue

        # Set other loggers to WARNING level
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)


# Add trace method to Logger class
def trace(self, message, *args, **kwargs):
    """Log a message at TRACE level."""
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)


# Add the method to the Logger class
logging.Logger.trace = trace


# Create default application logger
logger = setup_logger(
    name="gl_predictor",
    log_level=logging.INFO,
    silence_others=True,
)
