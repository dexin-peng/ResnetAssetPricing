import os
import logging
from datetime import datetime


class Logger(logging.Logger):
    def __init__(self, user_level='info', log_dir=None):
        super().__init__('runner')
        self.log_dir = log_dir
        self.user_level = user_level
        self.setLevel(logging.DEBUG)

    def set_logger(self):
        self._change_handler(self.log_dir)

    def _change_handler(self, log_dir):
        self.log_dir = log_dir
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(self.log_dir, exist_ok=True)
        self.handlers.clear()

        fh = logging.FileHandler(os.path.join(self.log_dir, f'{timestamp}.log'))
        ch = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        fh.setLevel(logging.DEBUG)
        level_map = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
        }
        console_level = level_map.get(self.user_level, logging.INFO)
        if os.environ.get("ResAssetPricing_PROGRESS_ACTIVE") == "1":
            console_level = logging.ERROR
        ch.setLevel(console_level)

        self.addHandler(fh)
        self.addHandler(ch)

    def debug(self, message):
        self.log(logging.DEBUG, message)

    def info(self, message):
        self.log(logging.INFO, message)

    def warning(self, message):
        self.log(logging.WARNING, message)

    def error(self, message):
        self.log(logging.ERROR, message, exc_info=True)
        
