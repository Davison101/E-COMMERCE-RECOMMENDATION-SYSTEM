"""
Logging configuration module for LightGCN Recommender System.
Provides centralized logging setup with file and console handlers.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from utils.config import get_config


def setup_logging(
    name: Optional[str] = None,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        name: Logger name (defaults to root logger)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    config = get_config()
    
    # Use config defaults if not provided
    if level is None:
        level = config.logging.get('level', 'INFO')
    if log_file is None:
        log_file = config.logging.get('file', 'logs/recommender.log')
    if format_string is None:
        format_string = config.logging.get(
            'format', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        max_bytes = config.logging.get('max_bytes', 10485760)
        backup_count = config.logging.get('backup_count', 5)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize root logger on import
setup_logging()