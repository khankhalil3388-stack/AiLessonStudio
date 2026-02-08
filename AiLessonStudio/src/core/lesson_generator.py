import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import random


@dataclass
class LessonContent:
    """Structured lesson content"""
    title: str
    introduction: str
    learning_objectives: List[str]
    key_concepts: List[str]
    content_sections: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]
    summary: str
    difficulty: str
    estimated_time: int  # minutes


class LessonGenerator:
    """Advanced lesson generation system"""

    def __init__(self, config):
        self.config = config

        # Lesson templates
        self.templates = {
            'beginner': {
                'sections': ['Introduction', 'What is it?', 'Why it matters?', 'Simple Example', 'Key Takeaways'],
                'style': 'simple',
                'examples_per_section': 1
            },
            'intermediate': {
                'sections': ['Overview', 'Core Concepts', 'Architecture', 'Implementation', 'Best Practices',
                             'Case Study'],
                'style': 'technical',
                'examples_per_section': 2
            },
            'advanced': {
                'sections': ['Deep Dive', 'Advanced Concepts', 'Performance Considerations', 'Scalability', 'Security',
                             'Real-world Implementation'],
                'style': 'expert',
                'examples_per_section': 3
            }
        }

        print("✅ Lesson Generator initialized")

    def generate_lesson(self, topic: str,
                        difficulty: str = "intermediate",
                        context: str = "",
                        target_audience: str = "students") -> Dict[str, Any]:
        """Generate a complete lesson"""
        template = self.templates.get(difficulty.lower(), self.templates['intermediate'])

        # Generate lesson structure
        lesson = LessonContent(
            title=self._generate_title(topic, difficulty),
            introduction=self._generate_introduction(topic, context, difficulty),
            learning_objectives=self._generate_learning_objectives(topic, difficulty),
            key_concepts=self._extract_key_concepts(topic, context),
            content_sections=self._generate_content_sections(topic, template),
            examples=self._generate_examples(topic, template['examples_per_section']),
            summary=self._generate_summary(topic),
            difficulty=difficulty,
            estimated_time=self._estimate_lesson_time(difficulty)
        )

        # Add interactive elements
        interactive_elements = self._add_interactive_elements(topic, difficulty)

        return {
            'lesson': lesson.__dict__,
            'interactive_elements': interactive_elements,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'topic': topic,
                'difficulty': difficulty,
                'target_audience': target_audience,
                'template_used': template['style']
            }
        }

    def _generate_title(self, topic: str, difficulty: str) -> str:
        """Generate lesson title"""
        difficulty_prefix = {
            'beginner': 'Introduction to',
            'intermediate': 'Understanding',
            'advanced': 'Advanced Topics in'
        }

        prefix = difficulty_prefix.get(difficulty.lower(), 'Understanding')
        return f"{prefix} {topic}"

    def _generate_introduction(self, topic: str, context: str, difficulty: str) -> str:
        """Generate lesson introduction"""
        templates = {
            'beginner': f"""
Welcome to this lesson on {topic}! 

This topic is fundamental to cloud computing and forms the basis for many advanced concepts we'll explore later. In this beginner-friendly lesson, we'll start with the basics and build up your understanding step by step.

{context if context else ''}

By the end of this lesson, you'll have a solid foundation in {topic} and be ready to apply these concepts in practical scenarios.
""",
            'intermediate': f"""
In this intermediate lesson on {topic}, we'll dive deeper into the technical aspects and practical applications.

{context if context else ''}

We'll explore the architecture, implementation details, and best practices that professionals use when working with {topic} in real-world cloud environments.
""",
            'advanced': f"""
This advanced lesson explores the complexities and sophisticated aspects of {topic}.

{context if context else ''}

We'll examine performance optimization, scalability considerations, security implications, and advanced implementation patterns used in large-scale production systems.
"""
        }

        return templates.get(difficulty.lower(), templates['intermediate']).strip()

    def _generate_learning_objectives(self, topic: str, difficulty: str) -> List[str]:
        """Generate learning objectives"""
        base_objectives = {
            'beginner': [
                f"Understand the basic concept of {topic}",
                f"Identify common use cases for {topic}",
                f"Recognize the benefits of using {topic}",
                f"Follow a simple example of {topic} implementation"
            ],
            'intermediate': [
                f"Explain the architecture of {topic}",
                f"Configure and deploy basic {topic} solutions",
                f"Compare {topic} with alternative approaches",
                f"Troubleshoot common issues with {topic}"
            ],
            'advanced': [
                f"Design complex systems using {topic}",
                f"Optimize {topic} for performance and cost",
                f"Implement security best practices for {topic}",
                f"Scale {topic} solutions for enterprise use"
            ]
        }

        return base_objectives.get(difficulty.lower(), base_objectives['intermediate'])

    def _extract_key_concepts(self, topic: str, context: str) -> List[str]:
        """Extract key concepts from topic and context"""
        # Common cloud computing concepts
        cloud_concepts = [
            'scalability', 'elasticity', 'availability', 'reliability',
            'fault tolerance', 'load balancing', 'auto-scaling',
            'virtualization', 'containers', 'microservices',
            'serverless', 'infrastructure as code', 'devops'
        ]

        # Topic-specific concepts
        topic_concepts = {
            'aws': ['EC2', 'S3', 'Lambda', 'RDS', 'VPC', 'IAM'],
            'azure': ['Virtual Machines', 'Blob Storage', 'Functions', 'SQL Database', 'Virtual Network'],
            'gcp': ['Compute Engine', 'Cloud Storage', 'Cloud Functions', 'Cloud SQL', 'VPC'],
            'virtualization': ['hypervisor', 'VM', 'host', 'guest', 'resource allocation'],
            'containers': ['docker', 'kubernetes', 'pod', 'node', 'orchestration'],
            'serverless': ['functions', 'events', 'triggers', 'stateless', 'scaling']
        }

        # Combine concepts
        concepts = []

        # Add topic-specific concepts
        for key, values in topic_concepts.items():
            if key in topic.lower():
                concepts.extend(values)

        # Add general cloud concepts
        concepts.extend(cloud_concepts[:5])

        return list(set(concepts))[:8]  # Limit to 8 concepts

    def _generate_content_sections(self, topic: str, template: Dict) -> List[Dict[str, Any]]:
        """Generate content sections"""
        sections = []

        for i, section_title in enumerate(template['sections']):
            section = {
                'title': section_title,
                'content': self._generate_section_content(topic, section_title, template['style']),
                'order': i + 1,
                'key_points': self._generate_key_points(topic, section_title),
                'estimated_reading_time': random.randint(2, 5)  # minutes
            }
            sections.append(section)

        return sections

    def _generate_section_content(self, topic: str, section_title: str, style: str) -> str:
        """Generate content for a section"""
        content_templates = {
            'Introduction': f"""
{section_title} to {topic} begins with understanding its fundamental purpose in cloud computing. This technology enables...

In modern cloud architecture, {topic} plays a crucial role by providing...
""",
            'Core Concepts': f"""
The core concepts of {topic} include several key components:

1. **Component A**: Handles primary functionality by...
2. **Component B**: Manages data processing through...
3. **Component C**: Ensures reliability and availability by...

These components work together to deliver the full capabilities of {topic}.
""",
            'Architecture': f"""
The architecture of {topic} follows a {random.choice(['layered', 'microservices', 'event-driven', 'serverless'])} pattern:

- **Frontend Layer**: Handles user interactions and requests
- **Application Layer**: Processes business logic and operations
- **Data Layer**: Manages storage and data persistence
- **Integration Layer**: Connects with other services and systems

This architecture enables scalability, reliability, and maintainability.
""",
            'Implementation': f"""
Implementing {topic} involves several key steps:

1. **Setup and Configuration**: Initialize the environment and configure settings
2. **Resource Provisioning**: Create and configure necessary resources
3. **Integration**: Connect with existing systems and services
4. **Testing**: Verify functionality and performance
5. **Deployment**: Release to production environment

Best practices include using infrastructure as code and automated testing.
"""
        }

        # Get template or create generic one
        if section_title in content_templates:
            content = content_templates[section_title]
        else:
            content = f"""
{section_title} for {topic} involves important considerations and practices.

In cloud computing, effective {section_title.lower()} requires understanding of...

Key aspects include:
- Aspect 1: Critical for performance
- Aspect 2: Essential for security
- Aspect 3: Important for cost optimization
"""

        # Apply style
        if style == 'simple':
            content = content.replace('**', '').replace('1.', '•').replace('2.', '•').replace('3.', '•')
        elif style == 'expert':
            content += "\n\n**Technical Details:**\n- Protocol: HTTP/2, gRPC\n- Data format: JSON, Protocol Buffers\n- Security: TLS 1.3, OAuth 2.0"

        return content.strip()

    def _generate_key_points(self, topic: str, section_title: str) -> List[str]:
        """Generate key points for a section"""
        key_points = [
            f"Understanding {section_title.lower()} is essential for working with {topic}",
            f"This section covers the fundamental aspects of {section_title.lower()}",
            f"Practical examples will demonstrate real-world applications",
            f"These concepts form the basis for more advanced topics"
        ]

        return key_points[:3]

    def _generate_examples(self, topic: str, count: int = 2) -> List[Dict[str, Any]]:
        """Generate examples for the lesson"""
        examples = []

        example_templates = [
            {
                'title': f'Basic {topic} Implementation',
                'description': f'A simple example showing how to implement {topic}',
                'code': f'# Example: Basic {topic} implementation\nprint("Implementing {topic} in cloud environment")',
                'explanation': 'This example demonstrates the fundamental approach to implementing this technology.'
            },
            {
                'title': f'{topic} Use Case',
                'description': f'A real-world scenario where {topic} provides value',
                'scenario': f'A company needs to scale their application during peak traffic. {topic} enables automatic scaling.',
                'explanation': 'This use case shows practical benefits in production environments.'
            },
            {
                'title': f'{topic} Configuration',
                'description': f'Configuration example for optimal performance',
                'config': '{\n  "setting1": "optimized_value",\n  "setting2": "performance_tuned"\n}',
                'explanation': 'These configuration settings optimize the technology for specific workloads.'
            }
        ]

        for i in range(min(count, len(example_templates))):
            examples.append(example_templates[i])

        return examples

    def _generate_summary(self, topic: str) -> str:
        """Generate lesson summary"""
        return f"""
In this lesson, we've explored {topic} and its importance in cloud computing.

Key takeaways:
1. {topic} provides essential capabilities for modern cloud applications
2. Understanding the architecture and implementation is crucial
3. Best practices ensure optimal performance and reliability
4. Real-world applications demonstrate practical value

Next steps: Practice implementing {topic} in the cloud simulator and explore related advanced topics.
""".strip()

    def _estimate_lesson_time(self, difficulty: str) -> int:
        """Estimate lesson completion time"""
        times = {
            'beginner': 15,
            'intermediate': 30,
            'advanced': 45
        }
        return times.get(difficulty.lower(), 30)

    def _add_interactive_elements(self, topic: str, difficulty: str) -> Dict[str, Any]:
        """Add interactive elements to lesson"""
        return {
            'quiz_questions': self._generate_quiz_questions(topic, difficulty),
            'practice_exercises': self._generate_practice_exercises(topic),
            'diagrams': self._suggest_diagrams(topic),
            'further_reading': self._suggest_further_reading(topic)
        }

    def _generate_quiz_questions(self, topic: str, difficulty: str) -> List[Dict[str, Any]]:
        """Generate quiz questions for the lesson"""
        questions = []

        question_types = ['multiple_choice', 'true_false', 'short_answer']

        for i in range(3):  # Generate 3 questions
            q_type = random.choice(question_types)

            if q_type == 'multiple_choice':
                question = {
                    'type': 'multiple_choice',
                    'question': f'What is a key feature of {topic}?',
                    'options': [
                        'A. High cost and complexity',
                        'B. Manual scaling requirements',
                        'C. Automatic resource provisioning',
                        'D. Limited availability'
                    ],
                    'correct_answer': 'C',
                    'explanation': f'{topic} provides automatic resource provisioning as a key cloud feature.'
                }
            elif q_type == 'true_false':
                question = {
                    'type': 'true_false',
                    'statement': f'{topic} is only available in on-premises environments.',
                    'correct_answer': 'False',
                    'explanation': f'{topic} is primarily a cloud computing technology available across major cloud providers.'
                }
            else:
                question = {
                    'type': 'short_answer',
                    'question': f'Explain one benefit of using {topic} in cloud computing.',
                    'sample_answer': f'One benefit of {topic} is automated scaling which allows applications to handle variable loads efficiently.',
                    'evaluation_criteria': ['mentions scalability', 'discusses automation', 'relates to cloud benefits']
                }

            questions.append(question)

        return questions

    def _generate_practice_exercises(self, topic: str) -> List[Dict[str, Any]]:
        """Generate practice exercises"""
        return [
            {
                'title': f'Implement Basic {topic}',
                'description': f'Create a simple implementation of {topic} using the cloud simulator',
                'steps': [
                    '1. Access the cloud simulator',
                    '2. Create necessary resources',
                    '3. Configure the implementation',
                    '4. Test functionality',
                    '5. Document your approach'
                ],
                'learning_objectives': [
                    'Hands-on experience with implementation',
                    'Understanding configuration options',
                    'Troubleshooting practice'
                ]
            },
            {
                'title': f'{topic} Use Case Analysis',
                'description': f'Analyze a real-world use case for {topic}',
                'task': 'Research and document how a company uses this technology',
                'deliverables': [
                    'Use case description',
                    'Technical architecture',
                    'Benefits achieved',
                    'Lessons learned'
                ]
            }
        ]

    def _suggest_diagrams(self, topic: str) -> List[str]:
        """Suggest diagrams for the lesson"""
        return [
            f'Architecture diagram for {topic}',
            f'{topic} workflow diagram',
            f'Data flow in {topic} implementation',
            f'Scalability diagram for {topic}'
        ]

    def _suggest_further_reading(self, topic: str) -> List[Dict[str, str]]:
        """Suggest further reading materials"""
        return [
            {
                'title': f'Official {topic} Documentation',
                'type': 'documentation',
                'description': 'Complete technical documentation and guides'
            },
            {
                'title': f'Best Practices for {topic}',
                'type': 'article',
                'description': 'Industry best practices and implementation patterns'
            },
            {
                'title': f'{topic} Case Studies',
                'type': 'case_study',
                'description': 'Real-world implementations and lessons learned'
            }
        ]

    def export_lesson(self, lesson_data: Dict[str, Any], format: str = 'json') -> str:
        """Export lesson in specified format"""
        if format.lower() == 'json':
            import json
            return json.dumps(lesson_data, indent=2, default=str)

        elif format.lower() == 'markdown':
            lesson = lesson_data['lesson']

            markdown = f"# {lesson['title']}\n\n"
            markdown += f"**Difficulty:** {lesson['difficulty'].title()} | "
            markdown += f"**Estimated Time:** {lesson['estimated_time']} minutes\n\n"

            markdown += f"## Introduction\n{lesson['introduction']}\n\n"

            markdown += "## Learning Objectives\n"
            for obj in lesson['learning_objectives']:
                markdown += f"- {obj}\n"
            markdown += "\n"

            markdown += "## Key Concepts\n"
            for concept in lesson['key_concepts']:
                markdown += f"- {concept}\n"
            markdown += "\n"

            for section in lesson['content_sections']:
                markdown += f"## {section['title']}\n"
                markdown += f"{section['content']}\n\n"

                if section.get('key_points'):
                    markdown += "**Key Points:**\n"
                    for point in section['key_points']:
                        markdown += f"- {point}\n"
                    markdown += "\n"

            if lesson['examples']:
                markdown += "## Examples\n"
                for example in lesson['examples']:
                    markdown += f"### {example['title']}\n"
                    markdown += f"{example['description']}\n\n"
                    if 'code' in example:
                        markdown += f"```python\n{example['code']}\n```\n\n"
                    markdown += f"*Explanation:* {example['explanation']}\n\n"

            markdown += f"## Summary\n{lesson['summary']}\n"

            return markdown

        else:
            return f"Format {format} not supported"