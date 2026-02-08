import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.ai_engine import CompleteAIEngine
from src.textbook.processor import AdvancedTextbookProcessor
from config import config


def test_ai_engine_integration():
    """Test AI engine integration"""
    ai_engine = CompleteAIEngine(config)

    # Test question answering
    context = "Cloud computing is the delivery of computing services over the internet."
    question = "What is cloud computing?"

    response = ai_engine.answer_question(question, context)
    assert response is not None
    assert hasattr(response, 'answer')
    assert hasattr(response, 'confidence')

    # Test text generation
    lesson = ai_engine.generate_lesson_content(
        topic="Virtualization",
        style="academic",
        length="short"
    )
    assert lesson is not None
    assert len(lesson.answer) > 0


def test_textbook_processor_integration():
    """Test textbook processor integration"""
    processor = AdvancedTextbookProcessor(config)

    # Test with sample content
    sample_text = """
    Chapter 1: Introduction to Cloud Computing

    Cloud computing provides on-demand access to computing resources.

    Chapter 2: Service Models

    There are three main service models: IaaS, PaaS, and SaaS.
    """

    # Test chapter extraction
    chapters = processor._extract_by_numbered_chapters(sample_text)
    assert len(chapters) >= 2

    # Test concept extraction
    concepts = processor._extract_key_concepts_from_text(sample_text)
    assert isinstance(concepts, list)


def test_system_initialization():
    """Test complete system initialization"""
    # Test config loading
    assert config is not None
    assert hasattr(config, 'models')
    assert hasattr(config, 'processing')
    assert hasattr(config, 'learning')

    # Test directory setup
    assert config.DATA_DIR.exists()
    assert config.TEXTBOOKS_DIR.exists()

    print("✅ All integration tests passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])