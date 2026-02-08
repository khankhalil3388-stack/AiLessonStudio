import random
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class QuizQuestion:
    """Quiz question structure"""
    id: str
    type: str
    text: str
    options: List[str]
    correct_answer: Any
    explanation: str
    difficulty: str
    points: float


class QuizGenerator:
    """Intelligent quiz generation system"""

    def __init__(self, config):
        self.config = config

        # Question templates
        self.templates = {
            'multiple_choice': self._generate_multiple_choice,
            'true_false': self._generate_true_false,
            'short_answer': self._generate_short_answer,
            'matching': self._generate_matching,
            'fill_blank': self._generate_fill_blank,
            'code': self._generate_code_question
        }

        print("✅ Quiz Generator initialized")

    def generate_quiz(self, topic: str,
                      context: str = "",
                      num_questions: int = 10,
                      difficulty: str = "intermediate",
                      question_types: List[str] = None) -> Dict[str, Any]:
        """Generate a complete quiz"""
        if question_types is None:
            question_types = ['multiple_choice', 'true_false', 'short_answer']

        questions = []
        total_points = 0

        for i in range(num_questions):
            # Select question type
            q_type = random.choice(question_types)

            # Generate question
            if q_type in self.templates:
                question = self.templates[q_type](topic, context, difficulty, i)
                questions.append(question)
                total_points += question.points

        # Create quiz metadata
        quiz_id = f"quiz_{topic.replace(' ', '_')}_{datetime.now().timestamp()}"

        return {
            'id': quiz_id,
            'topic': topic,
            'difficulty': difficulty,
            'total_questions': len(questions),
            'total_points': total_points,
            'estimated_time': num_questions * 2,  # 2 minutes per question
            'questions': [q.__dict__ for q in questions],
            'generated_at': datetime.now().isoformat(),
            'instructions': self._get_quiz_instructions(difficulty)
        }

    def _generate_multiple_choice(self, topic: str, context: str,
                                  difficulty: str, question_num: int) -> QuizQuestion:
        """Generate multiple choice question"""
        # Question templates based on difficulty
        templates = {
            'beginner': [
                f"What is the primary purpose of {topic}?",
                f"Which of these is a key feature of {topic}?",
                f"How does {topic} benefit cloud computing?",
                f"When would you typically use {topic}?"
            ],
            'intermediate': [
                f"Which architecture pattern is commonly used with {topic}?",
                f"What is a key consideration when implementing {topic}?",
                f"How does {topic} handle scalability challenges?",
                f"What security measures are important for {topic}?"
            ],
            'advanced': [
                f"What optimization technique would you use for {topic} at scale?",
                f"How does {topic} integrate with other cloud services?",
                f"What are the performance implications of {topic} architecture?",
                f"How would you troubleshoot {topic} in production?"
            ]
        }

        # Get appropriate templates
        diff_templates = templates.get(difficulty, templates['intermediate'])
        question_text = random.choice(diff_templates)

        # Generate options
        correct_option = self._get_correct_option(topic, difficulty)
        incorrect_options = self._get_incorrect_options(topic, difficulty, 3)

        options = [correct_option] + incorrect_options
        random.shuffle(options)

        # Find correct index
        correct_index = options.index(correct_option)
        correct_letter = chr(65 + correct_index)  # A, B, C, D

        return QuizQuestion(
            id=f"mc_{question_num}",
            type='multiple_choice',
            text=question_text,
            options=[f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)],
            correct_answer=correct_letter,
            explanation=self._get_explanation(topic, 'multiple_choice', difficulty),
            difficulty=difficulty,
            points=1.0
        )

    def _generate_true_false(self, topic: str, context: str,
                             difficulty: str, question_num: int) -> QuizQuestion:
        """Generate true/false question"""
        # True/False statements
        true_statements = [
            f"{topic} enables automatic scaling of resources.",
            f"{topic} improves application availability and reliability.",
            f"{topic} follows cloud computing best practices.",
            f"{topic} can be integrated with other cloud services."
        ]

        false_statements = [
            f"{topic} requires manual intervention for all operations.",
            f"{topic} is only available in on-premises environments.",
            f"{topic} cannot scale beyond initial capacity.",
            f"{topic} doesn't support security features."
        ]

        # Choose true or false
        is_true = random.choice([True, False])

        if is_true:
            statement = random.choice(true_statements)
            correct_answer = "True"
        else:
            statement = random.choice(false_statements)
            correct_answer = "False"

        return QuizQuestion(
            id=f"tf_{question_num}",
            type='true_false',
            text=statement,
            options=["True", "False"],
            correct_answer=correct_answer,
            explanation=self._get_explanation(topic, 'true_false', difficulty, is_true),
            difficulty=difficulty,
            points=0.5
        )

    def _generate_short_answer(self, topic: str, context: str,
                               difficulty: str, question_num: int) -> QuizQuestion:
        """Generate short answer question"""
        questions = {
            'beginner': [
                f"Describe {topic} in your own words.",
                f"What problem does {topic} solve?",
                f"Name one benefit of using {topic}."
            ],
            'intermediate': [
                f"Explain how {topic} achieves scalability.",
                f"Describe the architecture of {topic}.",
                f"What are the key components of {topic}?"
            ],
            'advanced': [
                f"Discuss performance optimization for {topic}.",
                f"Explain security considerations for {topic}.",
                f"Describe integration patterns for {topic}."
            ]
        }

        diff_questions = questions.get(difficulty, questions['intermediate'])
        question_text = random.choice(diff_questions)

        return QuizQuestion(
            id=f"sa_{question_num}",
            type='short_answer',
            text=question_text,
            options=[],
            correct_answer=self._get_sample_answer(topic, difficulty),
            explanation="Your answer should demonstrate understanding of key concepts.",
            difficulty=difficulty,
            points=2.0
        )

    def _generate_matching(self, topic: str, context: str,
                           difficulty: str, question_num: int) -> QuizQuestion:
        """Generate matching question"""
        # Create matching pairs
        pairs = {
            'Scalability': 'Ability to handle increasing load',
            'Availability': 'System uptime and reliability',
            'Elasticity': 'Automatic resource adjustment',
            'Fault Tolerance': 'System resilience to failures',
            'Load Balancing': 'Distributing traffic across servers'
        }

        left_items = list(pairs.keys())
        right_items = list(pairs.values())
        random.shuffle(right_items)

        return QuizQuestion(
            id=f"match_{question_num}",
            type='matching',
            text=f"Match the cloud computing concepts with their definitions:",
            options=[f"Match {item} with correct definition" for item in left_items],
            correct_answer=pairs,
            explanation="These are fundamental cloud computing concepts.",
            difficulty=difficulty,
            points=1.5
        )

    def _generate_fill_blank(self, topic: str, context: str,
                             difficulty: str, question_num: int) -> QuizQuestion:
        """Generate fill in the blank question"""
        blanks = {
            'beginner': f"{topic} provides [scalability] for cloud applications.",
            'intermediate': f"The [architecture] of {topic} enables [fault tolerance] and high availability.",
            'advanced': f"Optimizing {topic} requires understanding of [performance metrics], [resource allocation], and [cost management]."
        }

        question_text = blanks.get(difficulty, blanks['intermediate'])

        return QuizQuestion(
            id=f"fill_{question_num}",
            type='fill_blank',
            text=question_text,
            options=[],
            correct_answer=self._extract_blanks(question_text),
            explanation="Fill in the missing terms based on cloud computing principles.",
            difficulty=difficulty,
            points=1.0
        )

    def _generate_code_question(self, topic: str, context: str,
                                difficulty: str, question_num: int) -> QuizQuestion:
        """Generate coding question"""
        code_templates = {
            'beginner': f"""
# Complete the function to implement basic {topic}
def implement_{topic.lower().replace(' ', '_')}():
    print("Implementing {topic}")
    # Your code here
    return "success"
""",
            'intermediate': f"""
# Implement {topic} configuration
class {topic.replace(' ', '')}Config:
    def __init__(self):
        self.settings = {{}}

    def optimize_for_performance(self):
        # Add performance optimization settings
        pass
""",
            'advanced': f"""
# Design scalable {topic} architecture
class Scalable{topic.replace(' ', '')}:
    def handle_traffic_spike(self, request_count):
        # Implement auto-scaling logic
        # Consider load balancing, resource allocation
        pass
"""
        }

        question_text = f"Complete the code implementation for {topic}:"
        code = code_templates.get(difficulty, code_templates['intermediate'])

        return QuizQuestion(
            id=f"code_{question_num}",
            type='code',
            text=question_text,
            options=[code],
            correct_answer="Check for proper implementation and best practices",
            explanation="Focus on clean, efficient code following cloud best practices.",
            difficulty=difficulty,
            points=3.0
        )

    def _get_correct_option(self, topic: str, difficulty: str) -> str:
        """Get correct option for multiple choice"""
        options = {
            'beginner': [
                "Provides scalable cloud resources",
                "Enables on-demand computing",
                "Offers pay-as-you-go pricing",
                "Automates infrastructure management"
            ],
            'intermediate': [
                "Uses microservices architecture",
                "Implements auto-scaling groups",
                "Employs load balancing",
                "Utilizes distributed databases"
            ],
            'advanced': [
                "Implements circuit breaker pattern",
                "Uses blue-green deployment",
                "Employs canary releases",
                "Implements chaos engineering"
            ]
        }

        diff_options = options.get(difficulty, options['intermediate'])
        return random.choice(diff_options)

    def _get_incorrect_options(self, topic: str, difficulty: str, count: int) -> List[str]:
        """Get incorrect options for multiple choice"""
        incorrect_options = [
            "Requires manual scaling",
            "Only works on-premises",
            "Has fixed capacity limits",
            "Lacks security features",
            "Cannot integrate with other services",
            "Is expensive to implement",
            "Has poor performance",
            "Limited to specific regions"
        ]

        return random.sample(incorrect_options, count)

    def _get_explanation(self, topic: str, q_type: str, difficulty: str,
                         is_true: bool = None) -> str:
        """Get explanation for question"""
        explanations = {
            'multiple_choice': f"This question tests understanding of {topic} fundamentals in cloud computing.",
            'true_false': f"This statement is {'correct' if is_true else 'incorrect'} about {topic}.",
            'short_answer': f"This assesses comprehensive understanding of {topic}.",
            'matching': f"Matching tests knowledge of cloud computing terminology.",
            'fill_blank': f"Completing blanks requires understanding of {topic} concepts.",
            'code': f"Code implementation demonstrates practical knowledge of {topic}."
        }

        return explanations.get(q_type, f"Explanation for {topic} question.")

    def _get_sample_answer(self, topic: str, difficulty: str) -> str:
        """Get sample answer for short answer questions"""
        answers = {
            'beginner': f"{topic} is a cloud computing technology that provides scalable and on-demand resources for applications.",
            'intermediate': f"{topic} achieves scalability through distributed architecture, load balancing, and auto-scaling mechanisms that adjust resources based on demand.",
            'advanced': f"{topic} optimization involves performance tuning, cost management through resource right-sizing, implementing caching strategies, and using appropriate data storage solutions based on access patterns."
        }

        return answers.get(difficulty, answers['intermediate'])

    def _extract_blanks(self, text: str) -> Dict[str, str]:
        """Extract blanks from fill-in-the-blank text"""
        import re
        blanks = re.findall(r'\[(.*?)\]', text)
        answers = {}

        for blank in blanks:
            # Simple mapping of blanks to answers
            answer_map = {
                'scalability': 'ability to handle increasing load',
                'architecture': 'system design and structure',
                'fault tolerance': 'resilience to failures',
                'performance metrics': 'measurements of system efficiency',
                'resource allocation': 'distribution of computing resources',
                'cost management': 'optimization of cloud spending'
            }
            answers[blank] = answer_map.get(blank, 'correct term')

        return answers

    def _get_quiz_instructions(self, difficulty: str) -> str:
        """Get quiz instructions based on difficulty"""
        instructions = {
            'beginner': """
            Welcome to the beginner quiz! 
            - Read each question carefully
            - Take your time to think about answers
            - Review explanations to learn from mistakes
            - This quiz is designed to build foundational knowledge
            """,
            'intermediate': """
            Intermediate level quiz instructions:
            - Questions require deeper understanding
            - Consider real-world applications
            - Apply cloud computing principles
            - Focus on architectural concepts
            """,
            'advanced': """
            Advanced quiz - for experienced learners:
            - Questions test comprehensive knowledge
            - Consider scalability, security, and optimization
            - Apply best practices and patterns
            - Think about production scenarios
            """
        }

        return instructions.get(difficulty, instructions['intermediate']).strip()

    def export_quiz(self, quiz_data: Dict[str, Any], format: str = 'json') -> str:
        """Export quiz in specified format"""
        if format.lower() == 'json':
            return json.dumps(quiz_data, indent=2, default=str)

        elif format.lower() == 'text':
            text = f"QUIZ: {quiz_data['topic']}\n"
            text += f"Difficulty: {quiz_data['difficulty'].title()}\n"
            text += f"Questions: {quiz_data['total_questions']}\n"
            text += f"Total Points: {quiz_data['total_points']}\n"
            text += f"Estimated Time: {quiz_data['estimated_time']} minutes\n\n"
            text += f"Instructions:\n{quiz_data['instructions']}\n\n"

            for i, question in enumerate(quiz_data['questions'], 1):
                text += f"\nQ{i}. {question['text']}\n"

                if question['type'] == 'multiple_choice':
                    for option in question['options']:
                        text += f"  {option}\n"
                elif question['type'] == 'true_false':
                    text += "  [ ] True   [ ] False\n"
                elif question['type'] == 'short_answer':
                    text += "  Answer: ________________________________\n"
                elif question['type'] == 'matching':
                    text += "  Match the items with their definitions\n"

                text += f"\n"

            return text

        else:
            return f"Format {format} not supported"