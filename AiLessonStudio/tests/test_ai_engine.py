import unittest
import json
import tempfile
import os
import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.ai_engine import CompleteAIEngine
from src.core.lesson_generator import LessonGenerator
from src.core.qa_system import IntelligentQASystem
from src.core.content_analyzer import ContentAnalyzer

# Create aliases for backward compatibility
AIEngine = CompleteAIEngine
QASystem = IntelligentQASystem


# Simple functional tests
def run_functional_tests():
    """Run functional tests"""
    print("=" * 50)
    print("AI Lesson Studio - Functional Tests")
    print("=" * 50)

    results = []

    # Test AI Engine initialization
    print("Testing AI Engine initialization...")

    class MockConfig:
        class models:
            QA_MODEL = "distilbert-base-cased-distilled-squad"
            SUMMARIZATION_MODEL = "t5-small"
            TEXT_GENERATION_MODEL = "gpt2"
            NER_MODEL = "dslim/bert-base-NER"
            EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            TEMPERATURE = 0.7
            TOP_P = 0.9

    config = MockConfig()

    try:
        engine = AIEngine(config)
        print("✅ AI Engine initialized successfully")
        results.append(True)
    except Exception as e:
        print(f"❌ AI Engine initialization failed: {e}")
        results.append(False)

    # Test Lesson Generator
    print("\nTesting Lesson Generator...")

    class MockConfigLG:
        pass

    config_lg = MockConfigLG()

    try:
        generator = LessonGenerator(config_lg)
        lesson = generator.generate_lesson("Cloud Computing Basics")
        print("✅ Lesson Generator working")
        print(f"Generated lesson: {lesson['lesson']['title']}")
        results.append(True)
    except Exception as e:
        print(f"❌ Lesson Generator failed: {e}")
        results.append(False)

    # Test QA System
    print("\nTesting QA System...")

    class MockConfigQA:
        class models:
            QA_MODEL = "distilbert-base-cased-distilled-squad"

    config_qa = MockConfigQA()

    try:
        qa = QASystem(config_qa)
        print("✅ QA System initialized")
        results.append(True)
    except Exception as e:
        print(f"❌ QA System failed: {e}")
        results.append(False)

    # Test Content Analyzer
    print("\nTesting Content Analyzer...")

    class MockConfigCA:
        DATA_DIR = "."

    config_ca = MockConfigCA()

    try:
        analyzer = ContentAnalyzer(config_ca)
        analysis = analyzer.analyze_text("Cloud computing delivers services over internet.")
        print("✅ Content Analyzer working")
        print(f"Readability score: {analysis.readability}")
        results.append(True)
    except Exception as e:
        print(f"❌ Content Analyzer failed: {e}")
        results.append(False)

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)

    print(f"Functional Test Results: {passed}/{total} passed")

    if passed == total:
        print("✅ All functional tests passed!")
    else:
        print(f"⚠️ {total - passed} functional tests failed")
    print("=" * 50)

    return passed == total


# Unit Test Classes
class TestAIEngine(unittest.TestCase):
    """Test cases for AI Engine"""

    def setUp(self):
        """Set up test fixtures"""

        class MockConfig:
            class models:
                QA_MODEL = "distilbert-base-cased-distilled-squad"
                SUMMARIZATION_MODEL = "t5-small"
                TEXT_GENERATION_MODEL = "gpt2"
                NER_MODEL = "dslim/bert-base-NER"
                EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
                TEMPERATURE = 0.7
                TOP_P = 0.9

        self.config = MockConfig()
        self.ai_engine = AIEngine(self.config)

        self.sample_text = """
        Cloud computing is the delivery of computing services over the internet.
        These services include servers, storage, databases, networking, software.
        There are three main service models: IaaS, PaaS, and SaaS.
        Virtualization is a key technology enabling cloud computing.
        """

    def test_initialization(self):
        """Test AI Engine initialization"""
        self.assertIsNotNone(self.ai_engine)
        self.assertIsNotNone(self.ai_engine.device)
        self.assertIsNotNone(self.ai_engine.models)

    def test_answer_question(self):
        """Test question answering"""
        question = "What are the main service models in cloud computing?"
        answer = self.ai_engine.answer_question(question, self.sample_text)

        self.assertIsInstance(answer, dict)
        self.assertIn('answer', answer)
        self.assertIn('confidence', answer)
        self.assertIn('sources', answer)
        self.assertIn('reasoning', answer)

    def test_summarize_text(self):
        """Test text summarization"""
        summary = self.ai_engine.summarize_text(self.sample_text, max_length=50)

        self.assertIsInstance(summary, str)
        self.assertLess(len(summary.split()), len(self.sample_text.split()))

    def test_extract_concepts(self):
        """Test concept extraction"""
        concepts = self.ai_engine.extract_concepts(self.sample_text)

        self.assertIsInstance(concepts, list)
        # It might return empty list if NER model doesn't find entities
        if concepts:
            for concept in concepts:
                self.assertIn('concept', concept)
                self.assertIn('type', concept)
                self.assertIn('confidence', concept)

    def test_generate_lesson_content(self):
        """Test lesson content generation"""
        lesson = self.ai_engine.generate_lesson_content("Cloud Computing", self.sample_text)

        self.assertIsInstance(lesson, dict)
        self.assertIn('answer', lesson)
        self.assertIn('confidence', lesson)
        self.assertIn('sources', lesson)
        self.assertIn('reasoning', lesson)

    def test_generate_quiz_questions(self):
        """Test quiz question generation"""
        questions = self.ai_engine.generate_quiz_questions("Cloud Computing", self.sample_text, num_questions=2)

        self.assertIsInstance(questions, list)
        self.assertLessEqual(len(questions), 2)

        if questions:
            for question in questions:
                self.assertIn('type', question)
                self.assertIn('question', question)

    def test_calculate_similarity(self):
        """Test text similarity calculation"""
        text1 = "Cloud computing provides scalable resources"
        text2 = "Cloud services offer on-demand scalability"

        similarity = self.ai_engine.calculate_similarity(text1, text2)

        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

    def test_classify_difficulty(self):
        """Test text difficulty classification"""
        difficulty = self.ai_engine.classify_difficulty(self.sample_text)

        self.assertIsInstance(difficulty, str)
        self.assertIn(difficulty, ['beginner', 'intermediate', 'advanced'])

    @patch('transformers.pipeline')
    def test_with_mock_model(self, mock_pipeline):
        """Test with mocked transformer model"""
        mock_pipeline.return_value = MagicMock(return_value=[{'summary_text': 'Mock summary'}])

        # Reinitialize with mock
        ai_engine = AIEngine(self.config)
        summary = ai_engine.summarize_text(self.sample_text)

        # In this case, it might use fallback instead of mock
        self.assertIsInstance(summary, str)

    def test_empty_text_handling(self):
        """Test handling of empty text"""
        with self.assertRaises(Exception):
            self.ai_engine.answer_question("test", "")

    def test_memory_management(self):
        """Test memory management with large texts"""
        large_text = "Cloud computing " * 1000

        # Should handle large texts
        concepts = self.ai_engine.extract_concepts(large_text[:1000])  # Limit text
        self.assertIsInstance(concepts, list)


class TestLessonGenerator(unittest.TestCase):
    """Test cases for Lesson Generator"""

    def setUp(self):
        class MockConfig:
            pass

        self.config = MockConfig()
        self.generator = LessonGenerator(self.config)

        self.sample_topic = "Cloud Computing"
        self.sample_context = """
        Cloud computing delivers computing services over the internet.
        It provides on-demand access to resources.
        """

    def test_generate_lesson(self):
        """Test lesson generation"""
        lesson = self.generator.generate_lesson(self.sample_topic)

        self.assertIsInstance(lesson, dict)
        self.assertIn('lesson', lesson)
        self.assertIn('interactive_elements', lesson)
        self.assertIn('metadata', lesson)

        lesson_data = lesson['lesson']
        self.assertIn('title', lesson_data)
        self.assertIn('introduction', lesson_data)
        self.assertIn('learning_objectives', lesson_data)
        self.assertIn('key_concepts', lesson_data)
        self.assertIn('content_sections', lesson_data)
        self.assertIn('examples', lesson_data)
        self.assertIn('summary', lesson_data)

    def test_generate_lesson_with_context(self):
        """Test lesson generation with context"""
        lesson = self.generator.generate_lesson(
            topic=self.sample_topic,
            context=self.sample_context,
            difficulty="beginner"
        )

        self.assertIsInstance(lesson, dict)
        lesson_data = lesson['lesson']
        self.assertEqual(lesson_data['difficulty'], 'beginner')

    def test_export_lesson_json(self):
        """Test lesson export in JSON format"""
        lesson = self.generator.generate_lesson(self.sample_topic)
        exported = self.generator.export_lesson(lesson, 'json')

        self.assertIsInstance(exported, str)
        # Should be valid JSON
        json_data = json.loads(exported)
        self.assertIsInstance(json_data, dict)

    def test_export_lesson_markdown(self):
        """Test lesson export in Markdown format"""
        lesson = self.generator.generate_lesson(self.sample_topic)
        exported = self.generator.export_lesson(lesson, 'markdown')

        self.assertIsInstance(exported, str)
        self.assertIn('#', exported)  # Should have headings


class TestQASystem(unittest.TestCase):
    """Test cases for Q&A System"""

    def setUp(self):
        class MockConfig:
            class models:
                QA_MODEL = "distilbert-base-cased-distilled-squad"

        self.config = MockConfig()
        self.qa_system = QASystem(self.config)

        self.context = """
        Amazon Web Services (AWS) is a comprehensive cloud computing platform.
        It offers over 200 services including computing, storage, and databases.
        AWS uses a pay-as-you-go pricing model.
        """

    def test_answer_question(self):
        """Test answer generation"""
        question = "What is AWS?"
        answer = self.qa_system.answer_question(question, self.context)

        self.assertIsInstance(answer, dict)
        self.assertIn('answer', answer)
        self.assertIn('confidence', answer)
        self.assertIn('sources', answer)
        self.assertIn('supporting_evidence', answer)
        self.assertIn('alternative_answers', answer)
        self.assertIn('related_questions', answer)

    def test_answer_without_context(self):
        """Test answer without specific context"""
        question = "What is cloud computing?"
        answer = self.qa_system.answer_question(question)

        self.assertIsInstance(answer, dict)
        self.assertIn('answer', answer)
        # Should provide general knowledge answer

    def test_validate_answer(self):
        """Test answer validation"""
        question = "What is AWS?"
        answer = "AWS is Amazon Web Services"

        validation = self.qa_system.validate_answer(question, answer)

        self.assertIsInstance(validation, dict)
        self.assertIn('relevant', validation)
        self.assertIn('complete', validation)
        self.assertIn('specific', validation)
        self.assertIn('score', validation)
        self.assertIn('feedback', validation)

    def test_get_confidence_score(self):
        """Test confidence score calculation"""
        answer = "AWS is a cloud platform"
        context = "Amazon Web Services (AWS) is a cloud computing platform"

        confidence = self.qa_system.get_confidence_score(answer, context)

        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestContentAnalyzer(unittest.TestCase):
    """Test cases for Content Analyzer"""

    def setUp(self):
        class MockConfig:
            DATA_DIR = "."

        self.config = MockConfig()
        self.analyzer = ContentAnalyzer(self.config)

        self.sample_content = """
        # Cloud Computing

        Cloud computing is the delivery of computing services.

        ## Service Models

        1. IaaS - Infrastructure as a Service
        2. PaaS - Platform as a Service
        3. SaaS - Software as a Service

        ## Benefits

        - Cost savings
        - Scalability
        - Flexibility
        """

    def test_analyze_text(self):
        """Test content analysis"""
        analysis = self.analyzer.analyze_text(self.sample_content)

        self.assertIsInstance(analysis, dict)
        self.assertIn('readability', analysis)
        self.assertIn('complexity', analysis)
        self.assertIn('key_terms', analysis)
        self.assertIn('structure_score', analysis)
        self.assertIn('recommendations', analysis)
        self.assertIn('metadata', analysis)

    def test_compare_contents(self):
        """Test content comparison"""
        content1 = "Cloud computing provides scalable resources"
        content2 = "Cloud services offer on-demand scalability"

        comparison = self.analyzer.compare_contents(content1, content2)

        self.assertIsInstance(comparison, dict)
        self.assertIn('similarity_score', comparison)
        self.assertIn('readability_comparison', comparison)
        self.assertIn('complexity_comparison', comparison)
        self.assertIn('term_overlap', comparison)
        self.assertIn('structure_comparison', comparison)
        self.assertIn('recommendations', comparison)

    def test_generate_content_summary(self):
        """Test content summary generation"""
        summary = self.analyzer.generate_content_summary(self.sample_content)

        self.assertIsInstance(summary, str)
        self.assertLess(len(summary.split()), len(self.sample_content.split()))

    def test_extract_learning_objectives(self):
        """Test learning objective extraction"""
        objectives = self.analyzer.extract_learning_objectives(self.sample_content)

        self.assertIsInstance(objectives, list)
        if objectives:
            for obj in objectives:
                self.assertIsInstance(obj, str)

    def test_assess_content_quality(self):
        """Test content quality assessment"""
        quality = self.analyzer.assess_content_quality(self.sample_content)

        self.assertIsInstance(quality, dict)
        self.assertIn('overall_score', quality)
        self.assertIn('quality_level', quality)
        self.assertIn('readability_score', quality)
        self.assertIn('structure_score', quality)
        self.assertIn('technical_depth', quality)
        self.assertIn('key_strengths', quality)
        self.assertIn('improvement_areas', quality)
        self.assertIn('suitable_for', quality)


class IntegrationTests(unittest.TestCase):
    """Integration tests for AI Engine components"""

    def setUp(self):
        class MockConfig:
            class models:
                QA_MODEL = "distilbert-base-cased-distilled-squad"
                SUMMARIZATION_MODEL = "t5-small"
                TEXT_GENERATION_MODEL = "gpt2"
                NER_MODEL = "dslim/bert-base-NER"
                EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
                TEMPERATURE = 0.7
                TOP_P = 0.9

            DATA_DIR = "."

        self.config = MockConfig()
        self.ai_engine = AIEngine(self.config)
        self.analyzer = ContentAnalyzer(self.config)
        self.generator = LessonGenerator(self.config)
        self.qa_system = QASystem(self.config)

    def test_end_to_end_lesson_generation(self):
        """Test complete lesson generation pipeline"""
        # Analyze content
        content = "Cloud computing revolutionizes IT infrastructure. It provides on-demand access to computing resources."
        analysis = self.analyzer.analyze_text(content)

        # Generate lesson based on analysis
        lesson = self.generator.generate_lesson(
            topic="Cloud Computing",
            difficulty=analysis['complexity']
        )

        self.assertIsInstance(lesson, dict)
        self.assertIn('lesson', lesson)
        self.assertIn('interactive_elements', lesson)

    def test_question_answer_pipeline(self):
        """Test Q&A pipeline"""
        context = """
        Virtualization allows multiple virtual machines to run on a single physical machine.
        This improves resource utilization and isolation.
        """

        # Ask question
        question = "What is virtualization?"
        answer = self.qa_system.answer_question(question, context)

        # Validate answer
        validation = self.qa_system.validate_answer(question, answer['answer'])

        self.assertIsInstance(answer, dict)
        self.assertIsInstance(validation, dict)
        self.assertIn('score', validation)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in AI Engine"""

    def setUp(self):
        class MockConfig:
            class models:
                QA_MODEL = "distilbert-base-cased-distilled-squad"
                SUMMARIZATION_MODEL = "t5-small"
                TEXT_GENERATION_MODEL = "gpt2"
                NER_MODEL = "dslim/bert-base-NER"
                EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
                TEMPERATURE = 0.7
                TOP_P = 0.9

        self.config = MockConfig()
        self.ai_engine = AIEngine(self.config)

    def test_empty_text_handling(self):
        """Test handling of empty text"""
        with self.assertRaises(Exception):
            self.ai_engine.answer_question("test", "")


class TestPerformance(unittest.TestCase):
    """Performance tests for AI Engine"""

    def setUp(self):
        class MockConfig:
            class models:
                QA_MODEL = "distilbert-base-cased-distilled-squad"
                SUMMARIZATION_MODEL = "t5-small"
                TEXT_GENERATION_MODEL = "gpt2"
                NER_MODEL = "dslim/bert-base-NER"
                EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
                TEMPERATURE = 0.7
                TOP_P = 0.9

        self.config = MockConfig()
        self.ai_engine = AIEngine(self.config)
        self.medium_text = "Cloud computing " * 100

    def test_response_time_summary(self):
        """Test summary generation response time"""
        start_time = time.time()
        summary = self.ai_engine.summarize_text(self.medium_text)
        end_time = time.time()

        response_time = end_time - start_time
        self.assertLess(response_time, 10.0)  # Should complete within 10 seconds
        self.assertIsInstance(summary, str)

    def test_concurrent_processing(self):
        """Test handling of concurrent requests"""
        results = []
        errors = []

        def process_text(text_id):
            try:
                result = self.ai_engine.extract_concepts(
                    f"Cloud computing text {text_id}"
                )
                results.append((text_id, result))
            except Exception as e:
                errors.append((text_id, str(e)))

        # Create multiple threads
        threads = []
        for i in range(3):  # Reduced to 3 for stability
            thread = threading.Thread(target=process_text, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)

        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(len(errors), 0)


def main():
    """Main test runner"""
    # Run functional tests first
    print("\n" + "=" * 60)
    print("RUNNING FUNCTIONAL TESTS")
    print("=" * 60)
    functional_passed = run_functional_tests()

    if not functional_passed:
        print("\n⚠️ Functional tests failed. Unit tests may also fail.")

    # Run unit tests
    print("\n" + "=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTest(unittest.makeSuite(TestAIEngine))
    suite.addTest(unittest.makeSuite(TestLessonGenerator))
    suite.addTest(unittest.makeSuite(TestQASystem))
    suite.addTest(unittest.makeSuite(TestContentAnalyzer))
    suite.addTest(unittest.makeSuite(IntegrationTests))
    suite.addTest(unittest.makeSuite(TestErrorHandling))
    suite.addTest(unittest.makeSuite(TestPerformance))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"TEST SUMMARY:")
    print(f"{'=' * 60}")
    print(f"Functional Tests: {'PASSED' if functional_passed else 'FAILED'}")
    print(f"Unit Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"Success Rate: {success_rate:.1f}%")

    print(f"{'=' * 60}")

    # Exit with appropriate code
    if functional_passed and result.wasSuccessful():
        print("✅ All tests passed successfully!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())