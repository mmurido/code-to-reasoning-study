import logging
from pathlib import Path


def init_app_logger():
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        stream = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        stream.setFormatter(fmt)
        logger.addHandler(stream)

    return logger


def setup_phase_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"phase_{log_path.stem}")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    return logger
