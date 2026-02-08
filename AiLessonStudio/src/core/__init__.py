"""
AI Lesson Studio - Core Module
Contains the main AI engine, lesson generator, and content analyzer.
"""

from .ai_engine import CompleteAIEngine
from .lesson_generator import LessonGenerator
from .qa_system import IntelligentQASystem
from .content_analyzer import ContentAnalyzer

# Create aliases for backward compatibility
AIEngine = CompleteAIEngine
QASystem = IntelligentQASystem

__all__ = [
    'CompleteAIEngine',
    'AIEngine',  # Alias for backward compatibility
    'LessonGenerator',
    'IntelligentQASystem',
    'QASystem',  # Alias for backward compatibility
    'ContentAnalyzer'
]