import unittest
import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from core.ai_engine import AIEngine
from core.lesson_generator import LessonGenerator
from core.qa_system import QASystem
from core.content_analyzer import ContentAnalyzer


class TestAIEngine(unittest.TestCase):
    """Test cases for AI Engine"""

    def setUp(self):
        """Set up test fixtures"""
        self.ai_engine = AIEngine()
        self.sample_text = """
        Cloud computing is the delivery of computing services over the internet.
        These services include servers, storage, databases, networking, software.
        There are three main service models: IaaS, PaaS, and SaaS.
        Virtualization is a key technology enabling cloud computing.
        """

        self.sample_chapter = {
            "title": "Introduction to Cloud Computing",
            "content": self.sample_text,
            "sections": [
                {"title": "What is Cloud Computing", "content": "Cloud computing definition..."},
                {"title": "Service Models", "content": "IaaS, PaaS, SaaS explanation..."},
                {"title": "Deployment Models", "content": "Public, private, hybrid clouds..."}
            ]
        }

    def test_initialization(self):
        """Test AI Engine initialization"""
        self.assertIsNotNone(self.ai_engine)
        self.assertIsNotNone(self.ai_engine.nlp_model)
        self.assertIsNotNone(self.ai_engine.tokenizer)

    def test_extract_key_concepts(self):
        """Test key concept extraction"""
        concepts = self.ai_engine.extract_key_concepts(self.sample_text)

        self.assertIsInstance(concepts, list)
        self.assertGreater(len(concepts), 0)

        # Check for expected concepts
        expected_concepts = ['cloud computing', 'computing services', 'service models']
        for concept in expected_concepts:
            self.assertIn(concept.lower(), [c.lower() for c in concepts])

    def test_generate_summary(self):
        """Test text summarization"""
        summary = self.ai_engine.generate_summary(self.sample_text, max_length=50)

        self.assertIsInstance(summary, str)
        self.assertLess(len(summary.split()), len(self.sample_text.split()))
        self.assertIn('cloud computing', summary.lower())

    def test_answer_question(self):
        """Test question answering"""
        question = "What are the main service models in cloud computing?"
        answer = self.ai_engine.answer_question(self.sample_text, question)

        self.assertIsInstance(answer, dict)
        self.assertIn('answer', answer)
        self.assertIn('confidence', answer)
        self.assertIn('service models', answer['answer'].lower())

    def test_generate_quiz_questions(self):
        """Test quiz question generation"""
        questions = self.ai_engine.generate_quiz_questions(self.sample_text, num_questions=3)

        self.assertIsInstance(questions, list)
        self.assertEqual(len(questions), 3)

        for question in questions:
            self.assertIn('question', question)
            self.assertIn('options', question)
            self.assertIn('correct_answer', question)
            self.assertIn('explanation', question)

    def test_analyze_sentiment(self):
        """Test sentiment analysis of text"""
        sentiment = self.ai_engine.analyze_sentiment(self.sample_text)

        self.assertIsInstance(sentiment, dict)
        self.assertIn('label', sentiment)
        self.assertIn('score', sentiment)
        self.assertIn(sentiment['label'], ['POSITIVE', 'NEGATIVE', 'NEUTRAL'])

    def test_extract_entities(self):
        """Test named entity recognition"""
        entities = self.ai_engine.extract_entities(self.sample_text)

        self.assertIsInstance(entities, list)
        if entities:  # Some models might not find entities in this text
            for entity in entities:
                self.assertIn('text', entity)
                self.assertIn('label', entity)
                self.assertIn('start', entity)
                self.assertIn('end', entity)

    def test_generate_flashcards(self):
        """Test flashcard generation"""
        flashcards = self.ai_engine.generate_flashcards(self.sample_text, num_cards=2)

        self.assertIsInstance(flashcards, list)
        self.assertEqual(len(flashcards), 2)

        for card in flashcards:
            self.assertIn('term', card)
            self.assertIn('definition', card)
            self.assertIn('example', card)

    def test_calculate_text_difficulty(self):
        """Test text difficulty calculation"""
        difficulty = self.ai_engine.calculate_text_difficulty(self.sample_text)

        self.assertIsInstance(difficulty, dict)
        self.assertIn('level', difficulty)
        self.assertIn('score', difficulty)
        self.assertIn('metrics', difficulty)

        self.assertIn(difficulty['level'], ['Beginner', 'Intermediate', 'Advanced'])

    def test_generate_learning_objectives(self):
        """Test learning objective generation"""
        objectives = self.ai_engine.generate_learning_objectives(self.sample_text, num_objectives=3)

        self.assertIsInstance(objectives, list)
        self.assertEqual(len(objectives), 3)

        for obj in objectives:
            self.assertIsInstance(obj, str)
            self.assertTrue(obj.strip())

    @patch('transformers.pipeline')
    def test_with_mock_model(self, mock_pipeline):
        """Test with mocked transformer model"""
        mock_model = MagicMock()
        mock_model.return_value = [{'summary_text': 'Mock summary'}]
        mock_pipeline.return_value = mock_model

        ai_engine = AIEngine(model_name='mock-model')
        summary = ai_engine.generate_summary(self.sample_text)

        self.assertEqual(summary, 'Mock summary')

    def tearDown(self):
        """Clean up after tests"""
        pass


class TestLessonGenerator(unittest.TestCase):
    """Test cases for Lesson Generator"""

    def setUp(self):
        self.generator = LessonGenerator()
        self.chapter_data = {
            "title": "Cloud Computing Basics",
            "sections": [
                {"title": "Introduction", "content": "Cloud computing introduction..."},
                {"title": "Benefits", "content": "Cost savings, scalability, flexibility..."},
                {"title": "Challenges", "content": "Security, compliance, dependency..."}
            ],
            "key_concepts": ["cloud", "virtualization", "scalability"]
        }

    def test_generate_lesson_structure(self):
        """Test lesson structure generation"""
        lesson = self.generator.generate_lesson(self.chapter_data)

        self.assertIsInstance(lesson, dict)
        self.assertIn('title', lesson)
        self.assertIn('sections', lesson)
        self.assertIn('learning_objectives', lesson)
        self.assertIn('assessment', lesson)

    def test_generate_interactive_content(self):
        """Test interactive content generation"""
        interactive = self.generator.generate_interactive_content(
            self.chapter_data['sections'][0]
        )

        self.assertIsInstance(interactive, dict)
        self.assertIn('type', interactive)
        self.assertIn('content', interactive)

    def test_adapt_difficulty(self):
        """Test difficulty adaptation"""
        easy_content = self.generator.adapt_difficulty(
            self.chapter_data['sections'][0]['content'],
            'beginner'
        )

        advanced_content = self.generator.adapt_difficulty(
            self.chapter_data['sections'][0]['content'],
            'advanced'
        )

        self.assertIsInstance(easy_content, str)
        self.assertIsInstance(advanced_content, str)
        # Advanced content should be longer/more detailed
        self.assertGreater(len(advanced_content.split()), len(easy_content.split()))


class TestQASystem(unittest.TestCase):
    """Test cases for Q&A System"""

    def setUp(self):
        self.qa_system = QASystem()
        self.context = """
        Amazon Web Services (AWS) is a comprehensive cloud computing platform.
        It offers over 200 services including computing, storage, and databases.
        AWS uses a pay-as-you-go pricing model.
        """

    def test_answer_generation(self):
        """Test answer generation"""
        question = "What is AWS?"
        answer = self.qa_system.answer_question(self.context, question)

        self.assertIsInstance(answer, dict)
        self.assertIn('answer', answer)
        self.assertIn('confidence', answer)
        self.assertIn('sources', answer)
        self.assertIn('aws', answer['answer'].lower())

    def test_question_generation(self):
        """Test question generation from context"""
        questions = self.qa_system.generate_questions(self.context, num_questions=2)

        self.assertIsInstance(questions, list)
        self.assertEqual(len(questions), 2)

        for question in questions:
            self.assertIn('question', question)
            self.assertIn('answer', question)
            self.assertIn('type', question)

    def test_evaluate_answer(self):
        """Test answer evaluation"""
        student_answer = "AWS is a cloud platform"
        correct_answer = "AWS is Amazon Web Services, a cloud computing platform"

        evaluation = self.qa_system.evaluate_answer(student_answer, correct_answer)

        self.assertIsInstance(evaluation, dict)
        self.assertIn('score', evaluation)
        self.assertIn('feedback', evaluation)
        self.assertIn('confidence', evaluation)

        self.assertGreaterEqual(evaluation['score'], 0)
        self.assertLessEqual(evaluation['score'], 100)


class TestContentAnalyzer(unittest.TestCase):
    """Test cases for Content Analyzer"""

    def setUp(self):
        self.analyzer = ContentAnalyzer()
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

    def test_analyze_structure(self):
        """Test content structure analysis"""
        analysis = self.analyzer.analyze_structure(self.sample_content)

        self.assertIsInstance(analysis, dict)
        self.assertIn('headings', analysis)
        self.assertIn('sections', analysis)
        self.assertIn('lists', analysis)
        self.assertIn('complexity', analysis)

    def test_extract_keywords(self):
        """Test keyword extraction"""
        keywords = self.analyzer.extract_keywords(self.sample_content, top_n=5)

        self.assertIsInstance(keywords, list)
        self.assertLessEqual(len(keywords), 5)

        expected_keywords = ['cloud', 'computing', 'service', 'models']
        for keyword in expected_keywords:
            self.assertIn(keyword.lower(), [k.lower() for k in keywords])

    def test_calculate_readability(self):
        """Test readability calculation"""
        readability = self.analyzer.calculate_readability(self.sample_content)

        self.assertIsInstance(readability, dict)
        self.assertIn('flesch_reading_ease', readability)
        self.assertIn('grade_level', readability)
        self.assertIn('complexity', readability)

    def test_identify_concepts(self):
        """Test concept identification"""
        concepts = self.analyzer.identify_concepts(self.sample_content)

        self.assertIsInstance(concepts, list)

        expected_concepts = ['iaas', 'paas', 'saas']
        for concept in expected_concepts:
            self.assertIn(concept, [c.lower() for c in concepts])

    def test_generate_metadata(self):
        """Test metadata generation"""
        metadata = self.analyzer.generate_metadata(self.sample_content)

        self.assertIsInstance(metadata, dict)
        self.assertIn('title', metadata)
        self.assertIn('summary', metadata)
        self.assertIn('keywords', metadata)
        self.assertIn('estimated_time', metadata)

    def test_detect_learning_gaps(self):
        """Test learning gap detection"""
        student_performance = {
            'quiz_scores': [85, 70, 90, 65],
            'completed_topics': ['basics', 'storage', 'networking'],
            'time_spent': {'basics': 120, 'storage': 90, 'networking': 180}
        }

        gaps = self.analyzer.detect_learning_gaps(student_performance)

        self.assertIsInstance(gaps, dict)
        self.assertIn('weak_areas', gaps)
        self.assertIn('recommendations', gaps)
        self.assertIn('predicted_difficulty', gaps)


class IntegrationTests(unittest.TestCase):
    """Integration tests for AI Engine components"""

    def test_end_to_end_lesson_generation(self):
        """Test complete lesson generation pipeline"""
        analyzer = ContentAnalyzer()
        ai_engine = AIEngine()
        generator = LessonGenerator()

        # Analyze content
        metadata = analyzer.generate_metadata("""
        Cloud computing revolutionizes IT infrastructure.
        It provides on-demand access to computing resources.
        """)

        # Generate lesson
        lesson_data = {
            'title': metadata['title'],
            'content': "Sample content about cloud computing",
            'key_concepts': ai_engine.extract_key_concepts("Cloud computing concepts")
        }

        lesson = generator.generate_lesson(lesson_data)

        self.assertIsInstance(lesson, dict)
        self.assertIn('title', lesson)
        self.assertIn('content', lesson)
        self.assertIn('assessment', lesson)

    def test_question_answer_pipeline(self):
        """Test Q&A pipeline"""
        qa_system = QASystem()
        ai_engine = AIEngine()

        context = """
        Virtualization allows multiple virtual machines to run on a single physical machine.
        This improves resource utilization and isolation.
        """

        # Generate questions
        questions = qa_system.generate_questions(context, num_questions=1)

        # Generate answer using AI engine
        if questions:
            question = questions[0]['question']
            answer = ai_engine.answer_question(context, question)

            self.assertIsInstance(answer, dict)
            self.assertIn('answer', answer)

    def test_performance_analysis_pipeline(self):
        """Test performance analysis pipeline"""
        analyzer = ContentAnalyzer()

        # Simulate student performance data
        performance_data = {
            'assessments': [
                {'topic': 'Virtualization', 'score': 85, 'time': 45},
                {'topic': 'Networking', 'score': 70, 'time': 60},
                {'topic': 'Security', 'score': 60, 'time': 90}
            ],
            'engagement': {
                'lessons_completed': 15,
                'avg_time_per_lesson': 30,
                'quiz_attempts': 20
            }
        }

        # Analyze performance
        analysis = analyzer.analyze_performance(performance_data)

        self.assertIsInstance(analysis, dict)
        self.assertIn('overall_score', analysis)
        self.assertIn('weak_areas', analysis)
        self.assertIn('recommendations', analysis)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in AI Engine"""

    def setUp(self):
        self.ai_engine = AIEngine()

    def test_empty_text_handling(self):
        """Test handling of empty text"""
        with self.assertRaises(ValueError):
            self.ai_engine.extract_key_concepts("")

        with self.assertRaises(ValueError):
            self.ai_engine.generate_summary("")

    def test_invalid_parameters(self):
        """Test handling of invalid parameters"""
        with self.assertRaises(ValueError):
            self.ai_engine.generate_quiz_questions(self.sample_text, num_questions=0)

        with self.assertRaises(ValueError):
            self.ai_engine.generate_quiz_questions(self.sample_text, num_questions=-1)

    def test_memory_management(self):
        """Test memory management with large texts"""
        large_text = "Cloud computing " * 1000

        # Should handle large texts without crashing
        concepts = self.ai_engine.extract_key_concepts(large_text)
        self.assertIsInstance(concepts, list)

    def test_model_fallback(self):
        """Test model fallback mechanisms"""
        # This would test fallback to simpler models if primary fails
        # Implementation depends on specific fallback logic
        pass


class TestPerformance(unittest.TestCase):
    """Performance tests for AI Engine"""

    def setUp(self):
        self.ai_engine = AIEngine()
        self.medium_text = "Cloud computing " * 100

    def test_response_time_summary(self):
        """Test summary generation response time"""
        import time

        start_time = time.time()
        summary = self.ai_engine.generate_summary(self.medium_text)
        end_time = time.time()

        response_time = end_time - start_time
        self.assertLess(response_time, 5.0)  # Should complete within 5 seconds
        self.assertIsInstance(summary, str)

    def test_concurrent_processing(self):
        """Test handling of concurrent requests"""
        import threading

        results = []
        errors = []

        def process_text(text_id):
            try:
                result = self.ai_engine.extract_key_concepts(
                    f"Cloud computing text {text_id}"
                )
                results.append((text_id, result))
            except Exception as e:
                errors.append((text_id, str(e)))

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_text, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)


if __name__ == '__main__':
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
    print(f"Test Summary:")
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print(f"{'=' * 60}")

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)