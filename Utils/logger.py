import logging
import os
import yaml
import time
import functools
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Custom formatter that includes timestamp, log level, module, and function name"""
    
    def format(self, record):
        # Add timestamp in ISO format
        record.iso_timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        # Add module and function name if not present
        if not hasattr(record, 'module'):
            record.module = record.name
        if not hasattr(record, 'funcName'):
            record.funcName = record.function if hasattr(record, 'function') else 'N/A'
            
        return super().format(record)

def setup_logger(name):
    """
    Set up and configure logger based on config.yaml settings with enhanced features
    """
    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(config['logging']['file'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config['logging']['level']))
    
    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create handlers
    # 1. Main rotating file handler for all logs
    file_handler = RotatingFileHandler(
        config['logging']['file'],
        maxBytes=config['logging']['max_size'],
        backupCount=config['logging']['backup_count']
    )
    
    # 2. Error log handler for errors and above
    error_log_path = os.path.join(log_dir, 'error.log')
    error_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=config['logging']['max_size'],
        backupCount=config['logging']['backup_count']
    )
    error_handler.setLevel(logging.ERROR)
    
    # 3. Daily rotating handler for detailed logs
    daily_log_path = os.path.join(log_dir, f'daily_{datetime.now().strftime("%Y%m%d")}.log')
    daily_handler = TimedRotatingFileHandler(
        daily_log_path,
        when='midnight',
        interval=1,
        backupCount=7  # Keep logs for 7 days
    )
    
    # 4. Console handler
    console_handler = logging.StreamHandler()
    
    # Create formatters
    detailed_format = '%(iso_timestamp)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d | %(message)s'
    simple_format = '%(iso_timestamp)s | %(levelname)-8s | %(message)s'
    
    detailed_formatter = CustomFormatter(detailed_format)
    simple_formatter = CustomFormatter(simple_format)
    
    # Set formatters
    file_handler.setFormatter(detailed_formatter)
    error_handler.setFormatter(detailed_formatter)
    daily_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(daily_handler)
    logger.addHandler(console_handler)
    
    # Add some initial logging
    logger.debug(f"Logger '{name}' initialized with level {config['logging']['level']}")
    
    return logger

def log_execution_time(func):
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Function {func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

def log_model_usage(func):
    """Decorator to log LLM model usage."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            result = func(*args, **kwargs)
            # Log model usage details
            logger.info(f"Model used in {func.__name__}: {args[0].primary_model.__class__.__name__}")
            return result
        except Exception as e:
            logger.error(f"Model usage error in {func.__name__}: {str(e)}")
            raise
    return wrapper 