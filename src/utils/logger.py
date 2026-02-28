import logging
import sys

#a simple logger for handling logs
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s | %(name)s | %(message)s')

    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
