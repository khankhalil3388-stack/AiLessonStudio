import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.lesson_generator import LessonGenerator
from config import config


def test_lesson_generator_initialization():
    """Test lesson generator initialization"""
    generator = LessonGenerator(config)
    assert generator is not None
    assert hasattr(generator, 'generate_lesson')


def test_generate_lesson_structure():
    """Test lesson generation structure"""
    generator = LessonGenerator(config)

    topic = "Cloud Computing Basics"
    lesson = generator.generate_lesson(topic, difficulty="beginner")

    # Check structure
    assert 'title' in lesson
    assert 'content' in lesson
    assert 'learning_objectives' in lesson
    assert 'key_concepts' in lesson

    # Check content types
    assert isinstance(lesson['learning_objectives'], list)
    assert isinstance(lesson['key_concepts'], list)
    assert isinstance(lesson['content'], str)


def test_difficulty_levels():
    """Test different difficulty levels"""
    generator = LessonGenerator(config)

    topic = "Virtualization"

    beginner_lesson = generator.generate_lesson(topic, difficulty="beginner")
    intermediate_lesson = generator.generate_lesson(topic, difficulty="intermediate")
    advanced_lesson = generator.generate_lesson(topic, difficulty="advanced")

    # Each should have content
    assert len(beginner_lesson['content']) > 0
    assert len(intermediate_lesson['content']) > 0
    assert len(advanced_lesson['content']) > 0


def test_lesson_with_context():
    """Test lesson generation with context"""
    generator = LessonGenerator(config)

    topic = "AWS EC2"
    context = "Amazon Elastic Compute Cloud (EC2) provides scalable computing capacity."

    lesson = generator.generate_lesson(
        topic=topic,
        context=context,
        difficulty="intermediate"
    )

    assert topic.lower() in lesson['title'].lower()
    assert len(lesson['content']) > len(context)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])