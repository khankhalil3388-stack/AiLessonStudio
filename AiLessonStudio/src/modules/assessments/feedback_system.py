import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Feedback:
    """Structured feedback for assessments"""
    score_feedback: str
    strength_analysis: List[str]
    improvement_areas: List[str]
    recommendations: List[str]
    motivational_message: str


class FeedbackSystem:
    """Intelligent feedback generation system"""

    def __init__(self, config):
        self.config = config

        # Feedback templates
        self.feedback_templates = {
            'excellent': {
                'score_range': (90, 100),
                'templates': [
                    "Outstanding performance! You have excellent understanding of this topic.",
                    "Exceptional work! Your knowledge demonstrates mastery of the concepts.",
                    "Perfect score! Your understanding is comprehensive and accurate."
                ],
                'motivational': [
                    "Keep up the great work! You're on track to become an expert.",
                    "Your dedication to learning is paying off brilliantly!",
                    "This level of understanding will serve you well in real-world applications."
                ]
            },
            'good': {
                'score_range': (75, 89),
                'templates': [
                    "Very good work! You have a solid grasp of most concepts.",
                    "Good performance! Your understanding is developing well.",
                    "Well done! You're making excellent progress in your learning."
                ],
                'motivational': [
                    "You're doing great! With a little more practice, you'll reach excellence.",
                    "Your progress is impressive! Keep building on this foundation.",
                    "You're well on your way to mastering this topic!"
                ]
            },
            'average': {
                'score_range': (60, 74),
                'templates': [
                    "Good effort! You understand the basics but need more practice on some topics.",
                    "You're making progress! Focus on the areas that need improvement.",
                    "Decent performance! Review the key concepts to strengthen your understanding."
                ],
                'motivational': [
                    "Every expert was once a beginner. Keep practicing!",
                    "Learning is a journey. You're making good progress!",
                    "With consistent effort, you'll see continuous improvement."
                ]
            },
            'needs_improvement': {
                'score_range': (40, 59),
                'templates': [
                    "You're making progress. Focus on reviewing the core concepts.",
                    "There's room for improvement. Let's focus on the fundamentals.",
                    "Don't be discouraged! Learning takes time and practice."
                ],
                'motivational': [
                    "Every mistake is a learning opportunity. Keep going!",
                    "Persistence is key to mastery. You can do this!",
                    "The most important step is the next one. Keep learning!"
                ]
            },
            'poor': {
                'score_range': (0, 39),
                'templates': [
                    "Let's focus on the basics. Review the fundamental concepts.",
                    "We need to strengthen your foundation. Start with core concepts.",
                    "This is a starting point. Every expert begins with the basics."
                ],
                'motivational': [
                    "The journey of a thousand miles begins with a single step.",
                    "Every expert was once a beginner. You're starting your journey!",
                    "Learning is progress, no matter how small. Keep going!"
                ]
            }
        }

        # Strength analysis templates
        self.strength_templates = [
            "Strong understanding of {concept}",
            "Good grasp of {concept} principles",
            "Effective application of {concept}",
            "Solid knowledge of {concept} fundamentals",
            "Excellent comprehension of {concept} concepts"
        ]

        # Improvement area templates
        self.improvement_templates = [
            "Review {concept} for better understanding",
            "Practice more {concept} exercises",
            "Study {concept} implementation examples",
            "Focus on {concept} fundamentals",
            "Work on {concept} application scenarios"
        ]

        # Recommendation templates
        self.recommendation_templates = [
            "Complete the interactive lesson on {concept}",
            "Try the practice exercises for {concept}",
            "Review the diagrams and examples for {concept}",
            "Take another quiz focusing on {concept}",
            "Explore real-world applications of {concept}"
        ]

        print("✅ Feedback System initialized")

    def generate_feedback(self, assessment_result: Dict[str, Any],
                          question_results: List[Dict[str, Any]]) -> Feedback:
        """Generate comprehensive feedback for assessment"""
        score = assessment_result.get('percentage', 0)

        # Determine performance level
        performance_level = self._get_performance_level(score)

        # Generate score feedback
        score_feedback = self._generate_score_feedback(score, performance_level)

        # Analyze strengths and weaknesses
        strengths = self._analyze_strengths(question_results)
        improvements = self._analyze_improvement_areas(question_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(improvements)

        # Motivational message
        motivational = self._get_motivational_message(performance_level)

        return Feedback(
            score_feedback=score_feedback,
            strength_analysis=strengths,
            improvement_areas=improvements,
            recommendations=recommendations,
            motivational_message=motivational
        )

    def _get_performance_level(self, score: float) -> str:
        """Determine performance level based on score"""
        for level, data in self.feedback_templates.items():
            min_score, max_score = data['score_range']
            if min_score <= score <= max_score:
                return level
        return 'average'

    def _generate_score_feedback(self, score: float, performance_level: str) -> str:
        """Generate feedback based on score"""
        templates = self.feedback_templates[performance_level]['templates']
        feedback = random.choice(templates)

        # Add score-specific details
        if score >= 90:
            feedback += f" You scored {score:.1f}% - excellent work!"
        elif score >= 75:
            feedback += f" Your score of {score:.1f}% shows strong understanding."
        elif score >= 60:
            feedback += f" With {score:.1f}%, you're on the right track."
        else:
            feedback += f" Your score of {score:.1f}% indicates areas for focus."

        return feedback

    def _analyze_strengths(self, question_results: List[Dict[str, Any]]) -> List[str]:
        """Analyze strengths from question results"""
        strengths = []

        # Find correctly answered questions
        correct_questions = [q for q in question_results if q.get('is_correct', False)]

        if not correct_questions:
            return ["You're beginning your learning journey - every step forward is progress!"]

        # Extract concepts from correct questions
        concepts = set()
        for question in correct_questions[:3]:  # Limit to top 3
            # Extract concept from question ID or text
            q_text = question.get('text', '')
            if 'What is' in q_text:
                concept = q_text.split('What is')[-1].split('?')[0].strip()
                concepts.add(concept)

        # Generate strength statements
        for concept in list(concepts)[:3]:  # Limit to 3 strengths
            template = random.choice(self.strength_templates)
            strength = template.format(concept=concept)
            strengths.append(strength)

        return strengths or ["Good effort on the questions you answered correctly."]

    def _analyze_improvement_areas(self, question_results: List[Dict[str, Any]]) -> List[str]:
        """Analyze areas needing improvement"""
        improvements = []

        # Find incorrectly answered questions
        incorrect_questions = [q for q in question_results if not q.get('is_correct', False)]

        if not incorrect_questions:
            return ["Consider exploring more advanced topics to challenge yourself further."]

        # Extract concepts from incorrect questions
        concepts = set()
        for question in incorrect_questions[:3]:  # Limit to top 3
            # Extract concept from question
            q_text = question.get('text', '')

            # Try to extract topic
            keywords = ['scalability', 'availability', 'security', 'performance',
                        'architecture', 'implementation', 'deployment', 'monitoring']

            for keyword in keywords:
                if keyword in q_text.lower():
                    concepts.add(keyword)
                    break

            # If no keyword found, use generic concept
            if not concepts and 'What is' in q_text:
                concept = q_text.split('What is')[-1].split('?')[0].strip()
                concepts.add(concept)

        # Generate improvement statements
        for concept in list(concepts)[:3]:  # Limit to 3 improvements
            template = random.choice(self.improvement_templates)
            improvement = template.format(concept=concept)
            improvements.append(improvement)

        return improvements or ["Review the questions you missed to understand the concepts better."]

    def _generate_recommendations(self, improvements: List[str]) -> List[str]:
        """Generate learning recommendations based on improvements needed"""
        recommendations = []

        for improvement in improvements[:3]:  # Limit to 3 recommendations
            # Extract concept from improvement text
            concept = improvement.split('{concept}')[1].split(' ')[0] if '{concept}' in improvement else "key concepts"

            template = random.choice(self.recommendation_templates)
            recommendation = template.format(concept=concept)
            recommendations.append(recommendation)

        # Add general recommendations
        general_recommendations = [
            "Schedule regular review sessions to reinforce learning",
            "Practice applying concepts in the cloud simulator",
            "Join study groups or discussions to learn from others",
            "Set specific learning goals for each study session"
        ]

        if len(recommendations) < 3:
            recommendations.extend(random.sample(general_recommendations, 3 - len(recommendations)))

        return recommendations

    def _get_motivational_message(self, performance_level: str) -> str:
        """Get motivational message based on performance"""
        motivational_templates = self.feedback_templates[performance_level]['motivational']
        return random.choice(motivational_templates)

    def generate_question_feedback(self, question_result: Dict[str, Any]) -> str:
        """Generate specific feedback for a single question"""
        if question_result.get('is_correct', False):
            # Positive feedback for correct answers
            positive_feedback = [
                "Excellent! You correctly identified this concept.",
                "Well done! Your understanding is accurate.",
                "Perfect! You've mastered this aspect.",
                "Great job! Your answer demonstrates good comprehension.",
                "Correct! Your knowledge is on point."
            ]
            feedback = random.choice(positive_feedback)

            # Add explanation if available
            if question_result.get('explanation'):
                feedback += f" {question_result['explanation']}"

            return feedback
        else:
            # Constructive feedback for incorrect answers
            feedback = "Let's review this concept:\n"

            if question_result.get('correct_answer'):
                feedback += f"\nThe correct answer was: {question_result['correct_answer']}"

            if question_result.get('explanation'):
                feedback += f"\n\nExplanation: {question_result['explanation']}"

            # Learning tip
            tips = [
                "\n\nTip: Review the related lesson material for better understanding.",
                "\n\nTip: Practice similar questions to reinforce this concept.",
                "\n\nTip: Try the interactive examples to see this in action.",
                "\n\nTip: Create flashcards for key terms related to this topic."
            ]
            feedback += random.choice(tips)

            return feedback

    def generate_progress_feedback(self, progress_data: Dict[str, Any]) -> str:
        """Generate feedback based on learning progress"""
        mastery_percentage = progress_data.get('mastery_percentage', 0)
        total_concepts = progress_data.get('total_concepts', 0)
        mastered_concepts = progress_data.get('mastered_concepts', 0)

        if mastery_percentage >= 80:
            return f"""
            🎉 Outstanding Progress!

            You've mastered {mastered_concepts} out of {total_concepts} concepts ({mastery_percentage:.1f}%).

            Your dedication to learning is impressive! Consider exploring advanced topics or 
            challenging yourself with more complex scenarios in the cloud simulator.
            """
        elif mastery_percentage >= 60:
            return f"""
            📚 Good Progress!

            You're making solid progress with {mastered_concepts} out of {total_concepts} concepts mastered ({mastery_percentage:.1f}%).

            Keep up the consistent effort! Focus on the concepts you're still learning, 
            and you'll see continued improvement.
            """
        elif mastery_percentage >= 40:
            return f"""
            📖 Steady Learning!

            You've mastered {mastered_concepts} out of {total_concepts} concepts ({mastery_percentage:.1f}%).

            Learning is a journey, and you're on the right path. Take time to review 
            challenging concepts and practice regularly for better retention.
            """
        else:
            return f"""
            🌱 Beginning Your Journey!

            You've started learning with {mastered_concepts} out of {total_concepts} concepts ({mastery_percentage:.1f}%).

            Every expert was once a beginner. Focus on the fundamentals, take your time 
            with each concept, and don't hesitate to review material as needed.
            """

    def export_feedback(self, feedback: Feedback, format: str = 'text') -> str:
        """Export feedback in specified format"""
        if format.lower() == 'json':
            return json.dumps(feedback.__dict__, indent=2, default=str)

        elif format.lower() == 'text':
            text = "📊 ASSESSMENT FEEDBACK\n"
            text += "=" * 50 + "\n\n"

            text += "📝 SCORE FEEDBACK\n"
            text += "-" * 20 + "\n"
            text += f"{feedback.score_feedback}\n\n"

            if feedback.strength_analysis:
                text += "✅ YOUR STRENGTHS\n"
                text += "-" * 20 + "\n"
                for strength in feedback.strength_analysis:
                    text += f"• {strength}\n"
                text += "\n"

            if feedback.improvement_areas:
                text += "🎯 AREAS FOR IMPROVEMENT\n"
                text += "-" * 20 + "\n"
                for area in feedback.improvement_areas:
                    text += f"• {area}\n"
                text += "\n"

            if feedback.recommendations:
                text += "📚 RECOMMENDATIONS\n"
                text += "-" * 20 + "\n"
                for rec in feedback.recommendations:
                    text += f"• {rec}\n"
                text += "\n"

            text += "💪 MOTIVATIONAL MESSAGE\n"
            text += "-" * 20 + "\n"
            text += f"{feedback.motivational_message}\n"

            return text

        else:
            return f"Format {format} not supported"