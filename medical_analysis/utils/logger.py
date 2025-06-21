"""Professional logger utility for the system."""
import logging
import logging.config
from .config import get_config

_LOGGER_CACHE = {}

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
            'filename': 'app.log',
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
        logging.config.dictConfig(logging_config)
    except Exception:
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
    logger = logging.getLogger(name)
    _LOGGER_CACHE[name] = logger
    return logger 