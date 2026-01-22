import logging


def init_app_logger(name="app", level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger(name)
