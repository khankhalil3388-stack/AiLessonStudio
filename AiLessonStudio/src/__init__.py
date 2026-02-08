"""
AI Lesson Studio - Complete Cloud Computing Education Platform
"""
__version__ = "2.0.0"
__author__ = "AI Lesson Studio Team"
__description__ = "Complete AI-powered cloud computing education platform"

# Core modules
from .core import (
    CompleteAIEngine,
    LessonGenerator,
    IntelligentQASystem,
    ContentAnalyzer
)

# Database modules
from .database import (
    Base,
    User,
    Textbook,
    LearningSession,
    AssessmentResult,
    GeneratedContent,
    CRUDOperations,
    SessionManager
)

# Textbook modules
from .textbook import (
    TextbookExtractor,
    TextbookProcessor,
    TableExtractor,
    OCRHandler,
    DiagramExtractor
)

# Utils modules
from .utils import (
    FileHandler,
    Validator,
    Exporter,
    CacheManager
)

# Modules
from .modules import (
    diagrams,
    code_executor,
    assessments,
    multimedia,
    analytics
)

__all__ = [
    'CompleteAIEngine',
    'LessonGenerator',
    'IntelligentQASystem',
    'ContentAnalyzer',
    'Base',
    'User',
    'Textbook',
    'LearningSession',
    'AssessmentResult',
    'GeneratedContent',
    'CRUDOperations',
    'SessionManager',
    'TextbookExtractor',
    'TextbookProcessor',
    'TableExtractor',
    'OCRHandler',
    'DiagramExtractor',
    'FileHandler',
    'Validator',
    'Exporter',
    'CacheManager'
]