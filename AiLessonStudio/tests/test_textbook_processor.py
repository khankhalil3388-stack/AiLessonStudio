import pytest
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.textbook.processor import AdvancedTextbookProcessor
from config import config


def test_processor_initialization():
    """Test textbook processor initialization"""
    processor = AdvancedTextbookProcessor(config)
    assert processor is not None
    assert hasattr(processor, 'chapters')
    assert hasattr(processor, 'concepts')


def test_chapter_extraction():
    """Test chapter extraction from sample text"""
    processor = AdvancedTextbookProcessor(config)

    # Create sample text with chapters
    sample_text = """
    Chapter 1: Introduction to Cloud Computing

    Cloud computing is the delivery of computing services over the internet.

    Chapter 2: Cloud Service Models

    There are three main service models: IaaS, PaaS, and SaaS.

    Chapter 3: Deployment Models

    Cloud deployment models include public, private, and hybrid clouds.
    """

    # Test extraction
    chapters = processor._extract_by_numbered_chapters(sample_text)
    assert len(chapters) >= 2
    assert 'chapter_1' in chapters
    assert 'Introduction to Cloud Computing' in chapters['chapter_1']['title']


def test_concept_extraction():
    """Test concept extraction from text"""
    processor = AdvancedTextbookProcessor(config)

    sample_text = """
    Virtualization is a key technology in cloud computing. 
    It allows multiple virtual machines to run on a single physical server.
    Hypervisors manage these virtual machines.
    """

    concepts = processor._extract_key_concepts_from_text(sample_text)
    assert isinstance(concepts, list)
    assert len(concepts) > 0


def test_text_summarization():
    """Test text summarization"""
    processor = AdvancedTextbookProcessor(config)

    sample_text = """
    Cloud computing provides on-demand availability of computer system resources, 
    especially data storage and computing power, without direct active management 
    by the user. Large clouds often have functions distributed over multiple locations, 
    each of which is a data center. Cloud computing relies on sharing of resources 
    to achieve coherence and typically uses a pay-as-you-go model.
    """

    summary = processor._summarize_text(sample_text)
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) < len(sample_text)


def test_difficulty_assessment():
    """Test text difficulty assessment"""
    processor = AdvancedTextbookProcessor(config)

    # Simple text
    simple_text = "Cloud computing is easy to use. It helps businesses grow."
    assert processor._assess_difficulty(simple_text) == "beginner"

    # Complex text
    complex_text = """
    The orchestration of containerized applications through Kubernetes 
    necessitates comprehensive understanding of declarative configuration 
    and imperative commands for optimal resource utilization across 
    distributed computing environments.
    """
    assert processor._assess_difficulty(complex_text) in ["intermediate", "advanced"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])