"""Base class for all medical specialty agents."""
import os
import time
from pathlib import Path
import yaml
import torch

class MedicalAgent:
    """Base class for medical specialists with enhanced prompting."""
    def __init__(self, specialty: str, model: str = "gemini-pro", use_gpu: bool = True):
        self.specialty = specialty
        self.model_name = model
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.prompts = self._load_prompts()
        self.llm = None  # To be initialized by subclasses or LangGraph

    def _load_prompts(self):
        prompt_file = Path(__file__).parent.parent.parent / "prompts.yaml"
        with open(prompt_file) as f:
            prompts = yaml.safe_load(f)
            return prompts.get(self.specialty, {})

    def analyze(self, content: str, use_cache: bool = True):
        raise NotImplementedError("This method should be implemented by subclasses or LangGraph nodes.")

    def answer_query(self, query: str, context: dict):
        raise NotImplementedError("This method should be implemented by subclasses or LangGraph nodes.") 