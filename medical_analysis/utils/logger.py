"""Professional logger utility for the system."""
import logging
import logging.config
import os
from .config import get_config

_LOGGER_CACHE = {}

# Ensure the logs directory exists before any logging config
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, 'app.log')

DEFAULT_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] %(levelname)s %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': LOG_FILE_PATH,
            'mode': 'a',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}

def get_logger(name: str):
    """Get a logger with professional configuration."""
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    try:
        config = get_config()
        logging_config = config.get('logging', DEFAULT_LOGGING_CONFIG)
        # Force the file handler to use our LOG_FILE_PATH
        if 'handlers' in logging_config and 'file' in logging_config['handlers']:
            logging_config['handlers']['file']['filename'] = LOG_FILE_PATH
        logging.config.dictConfig(logging_config)
    except Exception:
        # Fallback config, always use logs/app.log
        fallback_config = DEFAULT_LOGGING_CONFIG.copy()
        fallback_config['handlers']['file']['filename'] = LOG_FILE_PATH
        logging.config.dictConfig(fallback_config)
    logger = logging.getLogger(name)
    _LOGGER_CACHE[name] = logger
    return logger 