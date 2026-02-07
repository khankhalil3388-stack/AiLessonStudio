import re
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class AssessmentResult:
    """Assessment evaluation result"""
    assessment_id: str
    user_id: str
    total_score: float
    max_score: float
    percentage: float
    question_results: List[Dict[str, Any]]
    time_spent: float
    difficulty_adjusted: bool
    feedback: str
    recommendations: List[str]


@dataclass
class QuestionEvaluation:
    """Individual question evaluation"""
    question_id: str
    question_type: str
    user_answer: Any
    correct_answer: Any
    score: float
    max_score: float
    is_correct: bool
    feedback: str
    confidence: float
    alternative_answers: List[str]


class AssessmentEvaluator:
    """Intelligent assessment evaluation system"""

    def __init__(self, config):
        self.config = config
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.concept_mappings = {}

        print("✅ Assessment Evaluator initialized")

    def evaluate_assessment(self, assessment_data: Dict[str, Any],
                            user_answers: Dict[str, Any],
                            user_id: str = "anonymous") -> AssessmentResult:
        """Evaluate complete assessment"""
        question_results = []
        total_score = 0
        max_score = 0

        # Evaluate each question
        for question in assessment_data.get('questions', []):
            question_id = question.get('id', '')
            question_type = question.get('type', 'multiple_choice')

            if question_id in user_answers:
                user_answer = user_answers[question_id]
                evaluation = self.evaluate_question(question, user_answer)
                question_results.append(evaluation)

                total_score += evaluation.score
                max_score += evaluation.max_score

        # Calculate percentage
        percentage = (total_score / max_score * 100) if max_score > 0 else 0

        # Generate overall feedback
        feedback = self._generate_overall_feedback(percentage, question_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(question_results)

        # Create assessment result
        assessment_id = f"assessment_{user_id}_{datetime.now().timestamp()}"

        result = AssessmentResult(
            assessment_id=assessment_id,
            user_id=user_id,
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            question_results=[q.__dict__ for q in question_results],
            time_spent=assessment_data.get('time_spent', 0),
            difficulty_adjusted=assessment_data.get('difficulty_adjusted', False),
            feedback=feedback,
            recommendations=recommendations
        )

        return result

    def evaluate_question(self, question: Dict[str, Any],
                          user_answer: Any) -> QuestionEvaluation:
        """Evaluate individual question based on type"""
        question_id = question.get('id', '')
        question_type = question.get('type', 'multiple_choice')
        max_score = question.get('points', 1.0)
        correct_answer = question.get('correct_answer')

        # Initialize evaluation
        evaluation = QuestionEvaluation(
            question_id=question_id,
            question_type=question_type,
            user_answer=user_answer,
            correct_answer=correct_answer,
            score=0.0,
            max_score=max_score,
            is_correct=False,
            feedback="",
            confidence=0.0,
            alternative_answers=[]
        )

        # Evaluate based on question type
        if question_type == 'multiple_choice':
            evaluation = self._evaluate_multiple_choice(question, user_answer, evaluation)

        elif question_type == 'true_false':
            evaluation = self._evaluate_true_false(question, user_answer, evaluation)

        elif question_type == 'short_answer':
            evaluation = self._evaluate_short_answer(question, user_answer, evaluation)

        elif question_type == 'code':
            evaluation = self._evaluate_code_question(question, user_answer, evaluation)

        elif question_type == 'matching':
            evaluation = self._evaluate_matching(question, user_answer, evaluation)

        elif question_type == 'fill_blank':
            evaluation = self._evaluate_fill_blank(question, user_answer, evaluation)

        # Generate feedback
        evaluation.feedback = self._generate_question_feedback(evaluation)

        return evaluation

    def _evaluate_multiple_choice(self, question: Dict[str, Any],
                                  user_answer: str,
                                  evaluation: QuestionEvaluation) -> QuestionEvaluation:
        """Evaluate multiple choice question"""
        correct_answer = question.get('correct_answer', '')
        options = question.get('options', [])

        # Check if answer is correct
        is_correct = str(user_answer).strip().lower() == str(correct_answer).strip().lower()

        evaluation.is_correct = is_correct
        evaluation.score = evaluation.max_score if is_correct else 0.0

        # Calculate confidence (simulated)
        if is_correct:
            evaluation.confidence = 0.9
        else:
            # Check if answer is a valid option
            valid_options = [opt.split('.')[0].strip().lower() for opt in options]
            if str(user_answer).strip().lower() in valid_options:
                evaluation.confidence = 0.3  # Wrong but valid option
            else:
                evaluation.confidence = 0.1  # Invalid option

        # Get alternatives (other correct options for multi-correct questions)
        if question.get('multiple_correct', False):
            evaluation.alternative_answers = question.get('correct_answers', [])

        return evaluation

    def _evaluate_true_false(self, question: Dict[str, Any],
                             user_answer: str,
                             evaluation: QuestionEvaluation) -> QuestionEvaluation:
        """Evaluate true/false question"""
        correct_answer = str(question.get('correct_answer', 'true')).lower()
        user_answer_str = str(user_answer).lower()

        is_correct = user_answer_str == correct_answer

        evaluation.is_correct = is_correct
        evaluation.score = evaluation.max_score if is_correct else 0.0
        evaluation.confidence = 0.8 if is_correct else 0.2

        return evaluation

    def _evaluate_short_answer(self, question: Dict[str, Any],
                               user_answer: str,
                               evaluation: QuestionEvaluation) -> QuestionEvaluation:
        """Evaluate short answer question with semantic analysis"""
        correct_answer = question.get('correct_answer', '')
        acceptable_answers = question.get('acceptable_answers', [correct_answer])

        # Convert to strings
        user_answer_str = str(user_answer).strip().lower()
        acceptable_strs = [str(ans).strip().lower() for ans in acceptable_answers]

        # Check for exact match
        if user_answer_str in acceptable_strs:
            evaluation.is_correct = True
            evaluation.score = evaluation.max_score
            evaluation.confidence = 0.95
            return evaluation

        # Calculate semantic similarity
        similarity_scores = []
        for acceptable in acceptable_strs:
            similarity = self._calculate_text_similarity(user_answer_str, acceptable)
            similarity_scores.append(similarity)

        max_similarity = max(similarity_scores) if similarity_scores else 0

        # Determine if answer is correct based on similarity threshold
        threshold = question.get('similarity_threshold', 0.7)
        is_correct = max_similarity >= threshold

        evaluation.is_correct = is_correct
        evaluation.confidence = max_similarity

        # Partial credit for close answers
        if is_correct:
            evaluation.score = evaluation.max_score
        else:
            # Give partial credit for some similarity
            if max_similarity > 0.4:
                evaluation.score = evaluation.max_score * max_similarity * 0.7
            else:
                evaluation.score = 0.0

        # Find alternative acceptable answers with high similarity
        evaluation.alternative_answers = [
            ans for ans, sim in zip(acceptable_strs, similarity_scores)
            if sim >= 0.6
        ]

        return evaluation

    def _evaluate_code_question(self, question: Dict[str, Any],
                                user_answer: str,
                                evaluation: QuestionEvaluation) -> QuestionEvaluation:
        """Evaluate coding question"""
        # For code questions, we would normally execute the code
        # Here we use pattern matching and keyword detection

        expected_output = question.get('expected_output', '')
        required_keywords = question.get('required_keywords', [])
        forbidden_patterns = question.get('forbidden_patterns', [])

        user_code = str(user_answer).lower()

        # Check for required keywords
        keyword_score = 0
        for keyword in required_keywords:
            if keyword.lower() in user_code:
                keyword_score += 1

        keyword_ratio = keyword_score / max(1, len(required_keywords))

        # Check for forbidden patterns
        forbidden_found = False
        for pattern in forbidden_patterns:
            if re.search(pattern, user_code, re.IGNORECASE):
                forbidden_found = True
                break

        # Calculate score
        if forbidden_found:
            evaluation.score = 0.0
            evaluation.confidence = 0.1
        else:
            evaluation.score = evaluation.max_score * keyword_ratio
            evaluation.confidence = keyword_ratio

        evaluation.is_correct = evaluation.score >= evaluation.max_score * 0.8

        return evaluation

    def _evaluate_matching(self, question: Dict[str, Any],
                           user_answer: Dict[str, str],
                           evaluation: QuestionEvaluation) -> QuestionEvaluation:
        """Evaluate matching question"""
        correct_matches = question.get('correct_matches', {})

        if not isinstance(user_answer, dict):
            evaluation.score = 0.0
            evaluation.confidence = 0.1
            return evaluation

        # Count correct matches
        correct_count = 0
        total_matches = len(correct_matches)

        for key, correct_value in correct_matches.items():
            if key in user_answer and user_answer[key] == correct_value:
                correct_count += 1

        # Calculate score
        accuracy = correct_count / max(1, total_matches)
        evaluation.score = evaluation.max_score * accuracy
        evaluation.is_correct = accuracy >= 0.8
        evaluation.confidence = accuracy

        return evaluation

    def _evaluate_fill_blank(self, question: Dict[str, Any],
                             user_answer: Dict[str, str],
                             evaluation: QuestionEvaluation) -> QuestionEvaluation:
        """Evaluate fill in the blank question"""
        blanks = question.get('blanks', {})

        if not isinstance(user_answer, dict):
            evaluation.score = 0.0
            evaluation.confidence = 0.1
            return evaluation

        # Check each blank
        correct_count = 0
        total_blanks = len(blanks)

        for blank_id, correct_value in blanks.items():
            if blank_id in user_answer:
                user_value = str(user_answer[blank_id]).strip().lower()
                correct = str(correct_value).strip().lower()

                # Allow for multiple acceptable answers
                acceptable = question.get('acceptable_answers', {}).get(blank_id, [correct])
                acceptable_strs = [str(ans).strip().lower() for ans in acceptable]

                if user_value in acceptable_strs:
                    correct_count += 1
                else:
                    # Check similarity
                    similarity = self._calculate_text_similarity(user_value, correct)
                    if similarity >= 0.8:
                        correct_count += 1

        # Calculate score
        accuracy = correct_count / max(1, total_blanks)
        evaluation.score = evaluation.max_score * accuracy
        evaluation.is_correct = accuracy >= 0.8
        evaluation.confidence = accuracy

        return evaluation

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not text1 or not text2:
            return 0.0

        # Simple similarity measures
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Exact match
        if text1_lower == text2_lower:
            return 1.0

        # Token overlap
        tokens1 = set(text1_lower.split())
        tokens2 = set(text2_lower.split())

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        jaccard = intersection / union

        # Word order similarity (simplified)
        words1 = text1_lower.split()
        words2 = text2_lower.split()

        common_words = [w for w in words1 if w in words2]
        if not common_words:
            return jaccard

        # Position similarity
        positions1 = [i for i, w in enumerate(words1) if w in common_words]
        positions2 = [i for i, w in enumerate(words2) if w in common_words]

        if positions1 and positions2:
            pos_sim = 1 - (abs(len(positions1) - len(positions2)) / max(len(positions1), len(positions2)))
            pos_sim *= 0.3  # Weight factor
        else:
            pos_sim = 0

        # Combined similarity
        similarity = jaccard * 0.7 + pos_sim

        return min(1.0, similarity)

    def _generate_question_feedback(self, evaluation: QuestionEvaluation) -> str:
        """Generate personalized feedback for a question"""
        if evaluation.is_correct:
            feedback_templates = [
                "Excellent! You correctly answered this question.",
                "Well done! Your understanding of this concept is solid.",
                "Perfect answer! You've mastered this topic.",
                "Correct! Your knowledge is accurately applied here."
            ]
            return np.random.choice(feedback_templates)

        # Incorrect answer feedback
        feedback = "Let's review this concept:\n"

        if evaluation.question_type == 'multiple_choice':
            feedback += f"The correct answer was: {evaluation.correct_answer}\n"
            feedback += "Consider reviewing the key differences between the options."

        elif evaluation.question_type == 'true_false':
            feedback += f"This statement is {evaluation.correct_answer}.\n"
            feedback += "Pay attention to the specific conditions mentioned."

        elif evaluation.question_type == 'short_answer':
            feedback += f"Expected answer: {evaluation.correct_answer[:100]}...\n"
            if evaluation.alternative_answers:
                feedback += f"Also acceptable: {', '.join(evaluation.alternative_answers[:2])}\n"
            feedback += "Try to include key terms in your answers."

        elif evaluation.question_type == 'code':
            feedback += "Check your code for:\n"
            feedback += "1. Correct syntax and indentation\n"
            feedback += "2. Proper use of functions/methods\n"
            feedback += "3. Logical flow and edge cases"

        return feedback

    def _generate_overall_feedback(self, percentage: float,
                                   question_results: List[QuestionEvaluation]) -> str:
        """Generate overall assessment feedback"""
        if percentage >= 90:
            feedback = "Outstanding performance! You have excellent understanding of the material."
        elif percentage >= 75:
            feedback = "Very good work! You have a solid grasp of most concepts."
        elif percentage >= 60:
            feedback = "Good effort! You understand the basics but need more practice on some topics."
        elif percentage >= 40:
            feedback = "You're making progress. Focus on reviewing the core concepts."
        else:
            feedback = "Let's focus on the fundamentals. Review the basic concepts before moving forward."

        # Add specific insights
        question_types = [q.question_type for q in question_results]
        type_counts = {}
        for q_type in question_types:
            type_counts[q_type] = type_counts.get(q_type, 0) + 1

        # Find weakest question type
        if question_results:
            incorrect_by_type = {}
            for evaluation in question_results:
                if not evaluation.is_correct:
                    q_type = evaluation.question_type
                    incorrect_by_type[q_type] = incorrect_by_type.get(q_type, 0) + 1

            if incorrect_by_type:
                weakest_type = max(incorrect_by_type.items(), key=lambda x: x[1])[0]
                feedback += f"\n\nYou had difficulty with {weakest_type.replace('_', ' ')} questions. Consider focusing on this type."

        return feedback

    def _generate_recommendations(self, question_results: List[QuestionEvaluation]) -> List[str]:
        """Generate learning recommendations based on assessment results"""
        recommendations = []

        # Analyze incorrect answers
        incorrect_concepts = []
        for evaluation in question_results:
            if not evaluation.is_correct:
                # Extract concept from question ID or metadata
                concept = evaluation.question_id.split('_')[0] if '_' in evaluation.question_id else "unknown"
                incorrect_concepts.append(concept)

        # Generate recommendations
        if incorrect_concepts:
            unique_concepts = list(set(incorrect_concepts))
            if len(unique_concepts) <= 3:
                for concept in unique_concepts:
                    recommendations.append(f"Review concept: {concept}")
            else:
                recommendations.append(f"Focus on the {len(unique_concepts)} concepts you missed")

        # Time-based recommendation
        slow_questions = [q for q in question_results if q.confidence < 0.5]
        if len(slow_questions) > len(question_results) / 2:
            recommendations.append("Take your time to fully understand each question before answering")

        # Question type recommendation
        question_types = [q.question_type for q in question_results]
        type_accuracy = {}

        for q_type in set(question_types):
            type_questions = [q for q in question_results if q.question_type == q_type]
            accuracy = sum(1 for q in type_questions if q.is_correct) / len(type_questions)
            type_accuracy[q_type] = accuracy

        # Recommend practice for low-accuracy question types
        for q_type, accuracy in type_accuracy.items():
            if accuracy < 0.6:
                recommendations.append(f"Practice more {q_type.replace('_', ' ')} questions")

        # Limit recommendations
        return recommendations[:5]

    def calculate_difficulty_level(self, assessment_data: Dict[str, Any],
                                   user_performance: float) -> str:
        """Calculate appropriate difficulty level for next assessment"""
        current_difficulty = assessment_data.get('difficulty', 'medium')

        if user_performance >= 80:
            # Excellent performance - increase difficulty
            if current_difficulty == 'beginner':
                return 'intermediate'
            elif current_difficulty == 'intermediate':
                return 'advanced'
            else:
                return 'advanced'  # Already at highest

        elif user_performance >= 60:
            # Good performance - maintain or slightly increase
            if current_difficulty == 'beginner':
                return 'intermediate'
            else:
                return current_difficulty

        elif user_performance >= 40:
            # Average performance - maintain
            return current_difficulty

        else:
            # Poor performance - decrease difficulty
            if current_difficulty == 'advanced':
                return 'intermediate'
            elif current_difficulty == 'intermediate':
                return 'beginner'
            else:
                return 'beginner'  # Already at lowest

    def export_evaluation_report(self, result: AssessmentResult,
                                 format: str = 'json') -> str:
        """Export evaluation report in specified format"""
        if format.lower() == 'json':
            return json.dumps(result.__dict__, indent=2, default=str)

        elif format.lower() == 'text':
            report = f"""
            ASSESSMENT REPORT
            =================

            Assessment ID: {result.assessment_id}
            User ID: {result.user_id}
            Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            SCORE SUMMARY
            -------------
            Total Score: {result.total_score:.1f}/{result.max_score:.1f}
            Percentage: {result.percentage:.1f}%
            Time Spent: {result.time_spent:.0f} seconds

            PERFORMANCE FEEDBACK
            --------------------
            {result.feedback}

            RECOMMENDATIONS
            ---------------
            """

            for i, rec in enumerate(result.recommendations, 1):
                report += f"{i}. {rec}\n"

            report += f"""
            DETAILED QUESTION ANALYSIS
            --------------------------
            """

            for i, q_result in enumerate(result.question_results, 1):
                report += f"""
                Question {i} ({q_result['question_type']}):
                - Correct: {'Yes' if q_result['is_correct'] else 'No'}
                - Score: {q_result['score']:.1f}/{q_result['max_score']:.1f}
                - Confidence: {q_result['confidence']:.1%}
                - Feedback: {q_result['feedback'][:100]}...
                """

            return report

        else:
            return f"Format {format} not supported"