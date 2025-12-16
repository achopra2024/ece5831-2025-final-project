"""
Logging utilities.
"""

import logging
import os
from datetime import datetime


def setup_logger(name, log_dir='logs', level=logging.INFO):
    """
    Set up logger with file and console handlers.
    
    Args:
        name (str): Logger name
        log_dir (str): Directory to save log files
        level: Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{name}_{timestamp}.log')
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
