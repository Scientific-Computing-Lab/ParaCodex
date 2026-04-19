"""
Centralized logging configuration for the pipeline.

Provides a structured logger that replaces all print statements with proper logging.
Supports different log levels and can be configured via environment variables.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class PipelineLogger:
    """Structured logger for the pipeline with consistent formatting."""
    
    def __init__(self, name: str, log_level: Optional[str] = None):
        """Initialize logger with name and optional log level.
        
        Args:
            name: Logger name (typically module name)
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                      If None, reads from LOG_LEVEL env var or defaults to INFO
        """
        self.logger = logging.getLogger(name)
        
        # Set log level
        if log_level is None:
            import os
            log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(self.logger.level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)
    
    def success(self, message: str, *args, **kwargs):
        """Log success message (info level with checkmark)."""
        self.logger.info(f"✓ {message}", *args, **kwargs)
    
    def failure(self, message: str, *args, **kwargs):
        """Log failure message (error level with cross)."""
        self.logger.error(f"✗ {message}", *args, **kwargs)


def get_logger(name: str, log_level: Optional[str] = None) -> PipelineLogger:
    """Get or create a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        log_level: Optional log level override
        
    Returns:
        PipelineLogger instance
    """
    return PipelineLogger(name, log_level)


# Module-level logger for convenience
_logger = None


def get_module_logger() -> PipelineLogger:
    """Get the module-level logger (singleton pattern)."""
    global _logger
    if _logger is None:
        _logger = get_logger(__name__)
    return _logger
