import os
import yaml
from medical_analysis.utils.logger import get_logger

def load_prompts():
    """Load prompts from prompts.yaml file."""
    logger = get_logger(__name__)
    try:
        prompts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'prompts.yaml')
        with open(prompts_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load prompts.yaml: {e}")
        return {} 