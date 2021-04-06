
from datetime import datetime
import logging
import os
from typing import Optional

from expats.settings import SETTINGS


DEFAULT_LOG_DIR = os.path.join(
    SETTINGS.home_root_path, f"log/{datetime.now().strftime('%Y%m%d_%H:%M')}"
)
LOG_FILENAME = "log.txt"


def init_setup_log(log_dir: Optional[str] = None):
    """setup logging to be called only once at initial stage

    Args:
        log_dir (Optional[str]): path tp logging directory. Defaults to None, which means DEFAULT_LOG_DIR.
    """
    _log_dir = log_dir if log_dir else DEFAULT_LOG_DIR
    if not os.path.exists(_log_dir):
        os.makedirs(_log_dir)

    root_logger = logging.getLogger()
    if SETTINGS.is_debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(_log_dir, LOG_FILENAME)
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)


def get_logger(name: Optional[str] = None):
    return logging.getLogger(name)
