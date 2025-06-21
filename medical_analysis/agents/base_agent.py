import os
from medical_analysis.utils.prompt_utils import load_prompts
from medical_analysis.utils.logger import get_logger

class MedicalAgent:
    """Base class for all medical specialty agents."""
    def __init__(self, specialty: str, model: str = "gemini-pro", use_gpu: bool = True):
        self.specialty = specialty
        self.model = model
        self.use_gpu = use_gpu
        self.logger = get_logger(self.__class__.__name__)
        self.prompts = load_prompts()

    def analyze(self, content: str, use_cache: bool = True):
        raise NotImplementedError("Each agent must implement its own analyze method.")

    def answer_query(self, query: str, context: dict):
        raise NotImplementedError("Each agent must implement its own answer_query method.") 