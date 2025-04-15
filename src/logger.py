import logging
import os

def get_logger(name: str, log_file: str):
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Evita duplicar handlers si se llama más de una vez
    if not logger.handlers:
        handler = logging.FileHandler(f"logs/{log_file}", encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
