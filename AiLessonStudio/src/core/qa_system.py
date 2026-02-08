import torch
from transformers import pipeline
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import re


@dataclass
class QAAnswer:
    """Structured question answering response"""
    answer: str
    confidence: float
    sources: List[str]
    supporting_evidence: List[str]
    alternative_answers: List[str]
    related_questions: List[str]


class IntelligentQASystem:
    """Intelligent Question Answering system for cloud computing"""

    def __init__(self, config):
        self.config = config
        self.qa_pipeline = None
        self.retrieval_model = None
        self.context_db = {}

        self._initialize_qa_system()
        print("✅ Intelligent QA System initialized")

    def _initialize_qa_system(self):
        """Initialize QA pipeline and models"""
        try:
            # Load QA pipeline
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.config.models.QA_MODEL,
                tokenizer=self.config.models.QA_MODEL,
                device=-1  # Use CPU for compatibility
            )

            # Initialize retrieval system
            self._init_retrieval_system()

        except Exception as e:
            print(f"⚠️ Error initializing QA system: {e}")
            self.qa_pipeline = None

    def _init_retrieval_system(self):
        """Initialize document retrieval system"""
        # Simple in-memory retrieval for now
        self.context_db = {
            'general': {},
            'textbook': {},
            'concepts': {}
        }

    def answer_question(self, question: str,
                        context: str = None,
                        textbook_content: Dict = None,
                        max_answers: int = 3) -> QAAnswer:
        """Answer questions with multiple strategies"""

        # Strategy 1: Direct QA with provided context
        if context and self.qa_pipeline:
            try:
                qa_result = self.qa_pipeline(
                    question=question,
                    context=context[:1000],  # Limit context length
                    max_answer_len=100,
                    handle_impossible_answer=True
                )

                if qa_result['score'] > 0.1:  # Minimal confidence threshold
                    return QAAnswer(
                        answer=qa_result['answer'],
                        confidence=qa_result['score'],
                        sources=["provided_context"],
                        supporting_evidence=[self._extract_evidence(context, qa_result['answer'])],
                        alternative_answers=self._generate_alternatives(question, context),
                        related_questions=self._generate_related_questions(question)
                    )
            except Exception as e:
                print(f"QA pipeline error: {e}")

        # Strategy 2: Textbook-based answering
        if textbook_content:
            answer = self._answer_from_textbook(question, textbook_content)
            if answer:
                return answer

        # Strategy 3: Concept-based answering
        answer = self._answer_from_concepts(question)
        if answer:
            return answer

        # Strategy 4: Fallback to general knowledge
        return self._generate_general_answer(question)

    def _answer_from_textbook(self, question: str,
                              textbook_content: Dict) -> Optional[QAAnswer]:
        """Answer question using textbook content"""
        if not textbook_content.get('chapters'):
            return None

        # Search for relevant content
        relevant_chapters = []
        for chapter_id, chapter in textbook_content['chapters'].items():
            content = chapter.get('content', '')
            if self._calculate_relevance(question, content) > 0.3:
                relevant_chapters.append((chapter_id, chapter, content))

        if not relevant_chapters:
            return None

        # Sort by relevance
        relevant_chapters.sort(key=lambda x: self._calculate_relevance(question, x[2]),
                               reverse=True)

        # Get best match
        best_chapter_id, best_chapter, best_content = relevant_chapters[0]

        # Extract answer
        answer = self._extract_answer_from_text(question, best_content)

        if answer:
            return QAAnswer(
                answer=answer,
                confidence=0.7,
                sources=[f"Textbook Chapter: {best_chapter.get('title', 'Unknown')}"],
                supporting_evidence=[self._extract_evidence(best_content, answer)],
                alternative_answers=[],
                related_questions=self._extract_related_questions(best_content)
            )

        return None

    def _answer_from_concepts(self, question: str) -> Optional[QAAnswer]:
        """Answer question using concept knowledge"""
        # Cloud computing concept database
        cloud_concepts = {
            'cloud computing': {
                'definition': 'The delivery of computing services over the internet',
                'examples': ['AWS', 'Azure', 'Google Cloud'],
                'characteristics': ['On-demand', 'Scalable', 'Pay-as-you-go']
            },
            'virtualization': {
                'definition': 'Creating virtual versions of hardware/resources',
                'examples': ['VMware', 'Hyper-V', 'KVM'],
                'characteristics': ['Resource sharing', 'Isolation', 'Efficiency']
            },
            'containerization': {
                'definition': 'Packaging applications with dependencies',
                'examples': ['Docker', 'Kubernetes', 'Podman'],
                'characteristics': ['Lightweight', 'Portable', 'Consistent']
            },
            'serverless': {
                'definition': 'Cloud execution model without server management',
                'examples': ['AWS Lambda', 'Azure Functions', 'Google Cloud Functions'],
                'characteristics': ['Event-driven', 'Auto-scaling', 'Pay-per-use']
            }
        }

        # Check for concept mentions
        question_lower = question.lower()
        matched_concepts = []

        for concept, data in cloud_concepts.items():
            if concept in question_lower:
                matched_concepts.append((concept, data))

        if matched_concepts:
            best_concept, concept_data = matched_concepts[0]

            # Generate answer based on concept
            answer = f"{concept_data['definition']}. Examples include: {', '.join(concept_data['examples'][:2])}."

            return QAAnswer(
                answer=answer,
                confidence=0.8,
                sources=[f"Cloud Computing Concept: {best_concept.title()}"],
                supporting_evidence=concept_data['characteristics'],
                alternative_answers=[],
                related_questions=self._generate_concept_questions(best_concept)
            )

        return None

    def _calculate_relevance(self, question: str, text: str) -> float:
        """Calculate relevance between question and text"""
        if not text:
            return 0.0

        # Simple keyword matching
        question_words = set(question.lower().split())
        text_words = set(text.lower().split())

        if not question_words or not text_words:
            return 0.0

        intersection = len(question_words.intersection(text_words))
        union = len(question_words.union(text_words))

        return intersection / union if union > 0 else 0.0

    def _extract_answer_from_text(self, question: str, text: str) -> str:
        """Extract answer from text using pattern matching"""
        # Look for definition patterns
        if 'what is' in question.lower():
            # Try to find definition in text
            definition_patterns = [
                r'is (?:a|an|the) (.+?)[\.\n]',
                r'refers to (.+?)[\.\n]',
                r'means (.+?)[\.\n]',
                r'can be defined as (.+?)[\.\n]'
            ]

            for pattern in definition_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()

        # Look for how-to answers
        elif 'how' in question.lower():
            # Extract procedural information
            steps = re.findall(r'(\d+\. .+?)[\n\.]|(- .+?)[\n\.]', text)
            if steps:
                step_list = [s[0] or s[1] for s in steps[:3]]
                return f"Steps: {'; '.join(step_list)}"

        # Default: return relevant sentence
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            # Return first sentence containing keywords
            question_keywords = set(question.lower().split())
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in question_keywords):
                    return sentence.strip()[:200]

            # Fallback to first sentence
            return sentences[0].strip()[:200]

        return ""

    def _extract_evidence(self, text: str, answer: str) -> str:
        """Extract supporting evidence from text"""
        if not text or not answer:
            return "No specific evidence available"

        # Find context around answer
        answer_lower = answer.lower()
        text_lower = text.lower()

        pos = text_lower.find(answer_lower)
        if pos == -1:
            return text[:150] + "..." if len(text) > 150 else text

        # Extract context around answer
        start = max(0, pos - 100)
        end = min(len(text), pos + len(answer) + 100)

        evidence = text[start:end]
        if start > 0:
            evidence = "..." + evidence
        if end < len(text):
            evidence = evidence + "..."

        return evidence

    def _generate_alternatives(self, question: str, context: str) -> List[str]:
        """Generate alternative answers"""
        alternatives = []

        # Extract different perspectives from context
        sentences = re.split(r'[.!?]+', context)

        for sentence in sentences[:3]:  # First 3 sentences as alternatives
            sentence = sentence.strip()
            if sentence and len(sentence) > 20:
                alternatives.append(sentence[:150] + "..." if len(sentence) > 150 else sentence)

        return alternatives[:2]  # Return top 2 alternatives

    def _generate_related_questions(self, question: str) -> List[str]:
        """Generate related questions"""
        question_lower = question.lower()
        related = []

        # Question transformation patterns
        if 'what is' in question_lower:
            # Transform to "how does" questions
            topic = question_lower.replace('what is', '').strip()
            related.append(f"How does {topic} work?")
            related.append(f"What are the benefits of {topic}?")
            related.append(f"When would you use {topic}?")

        elif 'how' in question_lower:
            # Transform to "what" questions
            related.append(f"What are the requirements for this?")
            related.append(f"What tools are needed for this?")

        elif 'why' in question_lower:
            # Transform to "what if" questions
            related.append(f"What if this doesn't work?")
            related.append(f"What are the alternatives?")

        # Add general cloud computing questions
        general_questions = [
            "How does this relate to cloud computing?",
            "What are the security considerations?",
            "How does this affect scalability?",
            "What are the cost implications?"
        ]

        related.extend(general_questions[:2])

        return list(set(related))[:3]  # Return unique questions, max 3

    def _extract_related_questions(self, text: str) -> List[str]:
        """Extract potential questions from text"""
        questions = []

        # Look for statements that could be turned into questions
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences[:5]:  # First 5 sentences
            sentence = sentence.strip()
            if not sentence or len(sentence.split()) < 5:
                continue

            # Convert statement to question
            words = sentence.split()
            if len(words) > 3:
                # Simple transformation: "X is Y" -> "What is X?"
                if ' is ' in sentence.lower():
                    parts = sentence.lower().split(' is ', 1)
                    if len(parts) == 2:
                        subject = parts[0].split()[-1] if parts[0].split() else "it"
                        questions.append(f"What is {subject}?")

        return questions[:2]

    def _generate_concept_questions(self, concept: str) -> List[str]:
        """Generate questions about a concept"""
        return [
            f"What are the key features of {concept}?",
            f"How is {concept} implemented in practice?",
            f"What are the benefits of using {concept}?",
            f"How does {concept} compare to similar technologies?"
        ][:2]

    def _generate_general_answer(self, question: str) -> QAAnswer:
        """Generate general answer when no specific information is available"""
        # Cloud computing knowledge base
        general_knowledge = {
            'cloud': "Cloud computing provides on-demand access to computing resources over the internet.",
            'aws': "Amazon Web Services (AWS) is a comprehensive cloud platform offering over 200 services.",
            'azure': "Microsoft Azure is a cloud computing service for building, testing, and managing applications.",
            'google cloud': "Google Cloud Platform provides infrastructure and platform services for cloud computing.",
            'virtual machine': "A virtual machine is a software emulation of a physical computer system.",
            'container': "Containers package applications with dependencies for consistent deployment.",
            'kubernetes': "Kubernetes is an open-source system for automating container deployment and management.",
            'serverless': "Serverless computing runs code without provisioning or managing servers.",
            'microservices': "Microservices architecture structures an application as loosely coupled services.",
            'devops': "DevOps combines software development and IT operations for faster delivery."
        }

        question_lower = question.lower()
        best_match = None
        best_score = 0

        for keyword, answer in general_knowledge.items():
            if keyword in question_lower:
                score = len(keyword)  # Simple scoring
                if score > best_score:
                    best_score = score
                    best_match = (keyword, answer)

        if best_match:
            keyword, answer = best_match
            return QAAnswer(
                answer=answer,
                confidence=0.5,
                sources=["General Cloud Computing Knowledge"],
                supporting_evidence=[f"Based on standard definition of {keyword}"],
                alternative_answers=[],
                related_questions=[f"What are the advantages of {keyword}?",
                                   f"How is {keyword} implemented?"]
            )

        # Fallback answer
        return QAAnswer(
            answer="I don't have specific information about that question. Could you rephrase or provide more context about cloud computing?",
            confidence=0.1,
            sources=[],
            supporting_evidence=[],
            alternative_answers=[],
            related_questions=["What is cloud computing?",
                               "What are the main cloud service providers?"]
        )

    def get_confidence_score(self, answer: str, context: str) -> float:
        """Calculate confidence score for an answer"""
        if not answer or not context:
            return 0.0

        # Simple confidence calculation
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        if not answer_words:
            return 0.0

        # Check how many answer words appear in context
        matches = len(answer_words.intersection(context_words))
        total = len(answer_words)

        return matches / total if total > 0 else 0.0

    def validate_answer(self, question: str, answer: str) -> Dict[str, Any]:
        """Validate if answer properly addresses the question"""
        validation = {
            'relevant': False,
            'complete': False,
            'specific': False,
            'score': 0.0,
            'feedback': ""
        }

        if not answer:
            validation['feedback'] = "No answer provided"
            return validation

        # Check relevance
        question_keywords = set(question.lower().split())
        answer_keywords = set(answer.lower().split())

        common_keywords = question_keywords.intersection(answer_keywords)
        relevance_score = len(common_keywords) / max(1, len(question_keywords))

        # Check completeness (answer length)
        completeness_score = min(1.0, len(answer) / 100)

        # Check specificity (avoid vague answers)
        vague_phrases = ['it depends', 'maybe', 'could be', 'possibly', 'generally']
        specificity_score = 1.0
        for phrase in vague_phrases:
            if phrase in answer.lower():
                specificity_score -= 0.2

        # Calculate overall score
        overall_score = (relevance_score * 0.5 +
                         completeness_score * 0.3 +
                         specificity_score * 0.2)

        validation['score'] = overall_score
        validation['relevant'] = relevance_score > 0.3
        validation['complete'] = completeness_score > 0.5
        validation['specific'] = specificity_score > 0.7

        # Generate feedback
        feedback_parts = []
        if relevance_score < 0.3:
            feedback_parts.append("Answer doesn't seem relevant to the question.")
        if completeness_score < 0.5:
            feedback_parts.append("Answer could be more complete.")
        if specificity_score < 0.7:
            feedback_parts.append("Answer could be more specific.")

        if not feedback_parts:
            feedback_parts.append("Good answer that addresses the question.")

        validation['feedback'] = " ".join(feedback_parts)

        return validation