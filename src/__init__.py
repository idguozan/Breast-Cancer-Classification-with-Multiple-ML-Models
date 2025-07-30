"""
Breast cancer classification package
"""

__version__ = "1.0.0"
__author__ = "Breast Cancer Analysis Team"
__description__ = "Advanced machine learning models for breast cancer classification"

from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .visualizer import Visualizer

__all__ = ['DataProcessor', 'ModelTrainer', 'Visualizer']
