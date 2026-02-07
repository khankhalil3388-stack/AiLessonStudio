"""
Textbook Processing Module
Handles textbook upload, extraction, and analysis.
"""

from .processor import AdvancedTextbookProcessor
from .extractor import ContentExtractor
from .diagram_extractor import DiagramExtractor
from .ocr_handler import OCRHandler

__all__ = [
    'AdvancedTextbookProcessor',
    'ContentExtractor',
    'DiagramExtractor',
    'OCRHandler'
]