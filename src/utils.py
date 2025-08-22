# src/utils.py

# This file is for shared utility functions, constants, or helper classes.
# For example, you might add logging configuration here.

import logging

def setup_logging():
    """Sets up basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )