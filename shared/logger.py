#!/usr/bin/env python3
"""
Shared Claims Pipeline Logger Setup
"""

import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Returns a unified claims system logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Format logs beautifully
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
