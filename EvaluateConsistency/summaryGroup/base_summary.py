import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from abc import ABC, abstractmethod


class SummaryGenerator(ABC):
    """Summary Generator Abstract Base Class"""

    def __init__(self, model_path, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print(f"[{self.__class__.__name__}] Use Device: {self.device.type}")
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def generate_summary(self, text, **kwargs):
        """An abstract method for generating text summaries"""
        pass

    def preprocess_text(self, text):
        """General text preprocessing: Clean up whitespace and special characters"""
        text = text.replace("\n", " ").replace("\r", " ").strip()
        return " ".join(text.split())