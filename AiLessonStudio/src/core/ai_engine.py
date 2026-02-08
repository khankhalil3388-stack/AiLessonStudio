import torch
from transformers import (
    pipeline,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    GPT2LMHeadModel,
    AutoModelForTokenClassification
)
from sentence_transformers import SentenceTransformer
import spacy
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AIResponse:
    """Structured AI response"""
    answer: str
    confidence: float
    sources: List[str]
    reasoning: Optional[str] = None
    alternatives: List[str] = None


class CompleteAIEngine:
    """Complete AI engine with multiple HuggingFace models"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing AI Engine on {self.device}")

        # Load all models
        self.models = {}
        self.tokenizers = {}

        self._load_all_models()

        # Initialize NLP pipelines
        self._init_nlp_pipelines()

        logger.info("✅ AI Engine initialized successfully")

    def _load_all_models(self):
        """Load all required AI models"""
        try:
            # 1. Question Answering Model
            logger.info("Loading QA model...")
            self.models['qa'] = AutoModelForQuestionAnswering.from_pretrained(
                self.config.models.QA_MODEL
            ).to(self.device)
            self.tokenizers['qa'] = AutoTokenizer.from_pretrained(
                self.config.models.QA_MODEL
            )

            # 2. Text Generation Model (for lessons)
            logger.info("Loading text generation model...")
            self.models['generation'] = GPT2LMHeadModel.from_pretrained(
                self.config.models.TEXT_GENERATION_MODEL
            ).to(self.device)
            self.tokenizers['generation'] = AutoTokenizer.from_pretrained(
                self.config.models.TEXT_GENERATION_MODEL
            )
            self.tokenizers['generation'].pad_token = self.tokenizers['generation'].eos_token

            # 3. Summarization Model
            logger.info("Loading summarization model...")
            self.models['summarization'] = T5ForConditionalGeneration.from_pretrained(
                self.config.models.SUMMARIZATION_MODEL
            ).to(self.device)
            self.tokenizers['summarization'] = AutoTokenizer.from_pretrained(
                self.config.models.SUMMARIZATION_MODEL
            )

            # 4. NER Model
            logger.info("Loading NER model...")
            self.models['ner'] = AutoModelForTokenClassification.from_pretrained(
                self.config.models.NER_MODEL
            ).to(self.device)
            self.tokenizers['ner'] = AutoTokenizer.from_pretrained(
                self.config.models.NER_MODEL
            )

            # 5. Sentence Transformer for embeddings
            logger.info("Loading sentence transformer...")
            self.embedding_model = SentenceTransformer(
                self.config.models.EMBEDDING_MODEL
            )

            # 6. Initialize HuggingFace pipelines for easy use
            self.pipelines = {
                'qa': pipeline(
                    "question-answering",
                    model=self.models['qa'],
                    tokenizer=self.tokenizers['qa'],
                    device=0 if torch.cuda.is_available() else -1
                ),
                'summarization': pipeline(
                    "summarization",
                    model=self.models['summarization'],
                    tokenizer=self.tokenizers['summarization'],
                    device=0 if torch.cuda.is_available() else -1
                ),
                'ner': pipeline(
                    "ner",
                    model=self.models['ner'],
                    tokenizer=self.tokenizers['ner'],
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                ),
                'text_generation': pipeline(
                    "text-generation",
                    model=self.models['generation'],
                    tokenizer=self.tokenizers['generation'],
                    device=0 if torch.cuda.is_available() else -1
                )
            }

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _init_nlp_pipelines(self):
        """Initialize NLP processing pipelines"""
        try:
            self.nlp = spacy.load("en_core_web_lg")
            self.nlp.max_length = 2000000
        except:
            logger.warning("spaCy model not available, using basic tokenization")
            self.nlp = None

    def answer_question(self, question: str, context: str,
                        max_length: int = 512) -> AIResponse:
        """Answer question based on context using AI"""
        try:
            # Use HuggingFace QA pipeline
            result = self.pipelines['qa'](
                question=question,
                context=context[:10000],  # Limit context length
                max_answer_len=max_length,
                handle_impossible_answer=True
            )

            return AIResponse(
                answer=result['answer'],
                confidence=result['score'],
                sources=["context"],
                reasoning=f"Answer extracted from provided context with confidence {result['score']:.2f}"
            )

        except Exception as e:
            logger.error(f"QA error: {e}")
            return AIResponse(
                answer="I couldn't find a specific answer in the provided context.",
                confidence=0.0,
                sources=[],
                reasoning=str(e)
            )

    def generate_lesson_content(self, topic: str, context: str = "",
                                style: str = "academic",
                                length: str = "medium") -> AIResponse:
        """Generate lesson content using AI"""
        try:
            # Create prompt based on style and length
            prompt = self._create_lesson_prompt(topic, context, style, length)

            # Generate content
            generated = self.pipelines['text_generation'](
                prompt,
                max_new_tokens=self._get_token_count(length),
                temperature=self.config.models.TEMPERATURE,
                top_p=self.config.models.TOP_P,
                do_sample=True,
                num_return_sequences=1
            )

            content = generated[0]['generated_text']

            # Clean up content (remove prompt if present)
            if content.startswith(prompt):
                content = content[len(prompt):].strip()

            # Summarize if too long
            if len(content.split()) > 500:
                content = self.summarize_text(content, max_length=300)

            return AIResponse(
                answer=content,
                confidence=0.8,
                sources=["AI generation"],
                reasoning=f"Generated {length} lesson on '{topic}' in {style} style"
            )

        except Exception as e:
            logger.error(f"Lesson generation error: {e}")
            return AIResponse(
                answer=f"# {topic}\n\nContent generation failed. Please try again.",
                confidence=0.0,
                sources=[],
                reasoning=str(e)
            )

    def _create_lesson_prompt(self, topic: str, context: str,
                              style: str, length: str) -> str:
        """Create prompt for lesson generation"""
        style_descriptions = {
            "academic": "Write in an academic, textbook-style tone.",
            "conversational": "Write in a conversational, easy-to-understand tone.",
            "technical": "Write in a technical, detailed manner with specifications.",
            "beginner": "Write for complete beginners with simple explanations."
        }

        length_tokens = {
            "short": 100,
            "medium": 300,
            "long": 500,
            "detailed": 800
        }

        prompt = f"""Generate a {length} lesson on '{topic}' for cloud computing students.

{style_descriptions.get(style, 'Write in an academic tone.')}

Structure the lesson with:
1. Introduction
2. Key concepts
3. Examples
4. Practical applications
5. Summary

{context if context else 'Include relevant cloud computing concepts.'}

Lesson:
"""

        return prompt

    def _get_token_count(self, length: str) -> int:
        """Get token count based on desired length"""
        length_map = {
            "short": 150,
            "medium": 300,
            "long": 500,
            "detailed": 800
        }
        return length_map.get(length, 300)

    def summarize_text(self, text: str,
                       max_length: int = 150,
                       min_length: int = 30) -> str:
        """Summarize text using AI"""
        try:
            # Use HuggingFace summarization pipeline
            result = self.pipelines['summarization'](
                text[:2000],  # Limit input length
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            return result[0]['summary_text']

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            # Fallback to extractive summarization
            return self._extractive_summary(text, max_length)

    def _extractive_summary(self, text: str, max_length: int) -> str:
        """Extractive summarization fallback"""
        if not self.nlp:
            # Simple sentence extraction
            sentences = text.split('. ')
            if len(sentences) <= 3:
                return text

            # Take first and last sentences
            summary = '. '.join([sentences[0], sentences[-1]]) + '.'
            if len(summary) > max_length:
                summary = summary[:max_length - 3] + '...'
            return summary

        # Use spaCy for better extraction
        doc = self.nlp(text[:5000])
        sentences = list(doc.sents)

        if len(sentences) <= 2:
            return text

        # Score sentences by position and length
        scored_sentences = []
        for i, sent in enumerate(sentences):
            score = 0

            # Favor first and last sentences
            if i == 0:
                score += 2
            if i == len(sentences) - 1:
                score += 1

            # Favor medium-length sentences
            sent_length = len(sent.text.split())
            if 10 <= sent_length <= 30:
                score += 1

            scored_sentences.append((score, sent.text))

        # Select top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        selected = [s[1] for s in scored_sentences[:3]]

        summary = ' '.join(selected)
        if len(summary) > max_length:
            summary = summary[:max_length - 3] + '...'

        return summary

    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract key concepts from text using NER"""
        try:
            # Use NER pipeline
            entities = self.pipelines['ner'](text[:2000])

            concepts = []
            for entity in entities:
                if entity['score'] > 0.8:  # High confidence threshold
                    concepts.append({
                        'concept': entity['word'],
                        'type': entity['entity_group'],
                        'confidence': entity['score'],
                        'context': self._get_entity_context(text, entity['start'], entity['end'])
                    })

            return concepts

        except Exception as e:
            logger.error(f"Concept extraction error: {e}")
            return []

    def _get_entity_context(self, text: str, start: int, end: int,
                            context_chars: int = 100) -> str:
        """Get context around an entity"""
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)

        context = text[context_start:context_end]
        if context_start > 0:
            context = '...' + context
        if context_end < len(text):
            context = context + '...'

        return context

    def generate_quiz_questions(self, topic: str, context: str,
                                num_questions: int = 5,
                                question_types: List[str] = None) -> List[Dict]:
        """Generate quiz questions using AI"""
        if question_types is None:
            question_types = ['multiple_choice', 'true_false', 'short_answer']

        try:
            questions = []

            for i in range(num_questions):
                # Select question type
                q_type = question_types[i % len(question_types)]

                # Generate question based on type
                if q_type == 'multiple_choice':
                    question = self._generate_mc_question(topic, context)
                elif q_type == 'true_false':
                    question = self._generate_tf_question(topic, context)
                else:
                    question = self._generate_short_answer_question(topic, context)

                questions.append(question)

            return questions

        except Exception as e:
            logger.error(f"Quiz generation error: {e}")
            return []

    def _generate_mc_question(self, topic: str, context: str) -> Dict:
        """Generate multiple choice question"""
        prompt = f"""Generate a multiple choice question about '{topic}' in cloud computing.

Context: {context[:500]}

Format:
Question: [question]
Options:
A. [option1]
B. [option2]
C. [option3]
D. [option4]
Correct Answer: [letter]

Make sure only one answer is correct.
"""

        try:
            generated = self.pipelines['text_generation'](
                prompt,
                max_new_tokens=200,
                temperature=0.7,
                num_return_sequences=1
            )[0]['generated_text']

            # Parse generated text
            lines = generated.split('\n')
            question_data = {
                'type': 'multiple_choice',
                'question': '',
                'options': [],
                'correct_answer': '',
                'explanation': ''
            }

            for line in lines:
                if line.startswith('Question:'):
                    question_data['question'] = line.replace('Question:', '').strip()
                elif line.strip().startswith(('A.', 'B.', 'C.', 'D.')):
                    question_data['options'].append(line.strip())
                elif line.startswith('Correct Answer:'):
                    question_data['correct_answer'] = line.replace('Correct Answer:', '').strip()

            return question_data

        except:
            # Fallback question
            return {
                'type': 'multiple_choice',
                'question': f'What is a key characteristic of {topic}?',
                'options': [
                    'A. It requires physical hardware',
                    'B. It provides on-demand resources',
                    'C. It is always more expensive',
                    'D. It cannot scale automatically'
                ],
                'correct_answer': 'B',
                'explanation': f'{topic} provides on-demand, scalable resources over the internet.'
            }

    def _generate_tf_question(self, topic: str, context: str) -> Dict:
        """Generate true/false question"""
        prompt = f"""Generate a true/false question about '{topic}' in cloud computing.

Context: {context[:500]}

Format:
Statement: [statement]
Correct Answer: [True/False]
Explanation: [brief explanation]
"""

        try:
            generated = self.pipelines['text_generation'](
                prompt,
                max_new_tokens=100,
                temperature=0.7,
                num_return_sequences=1
            )[0]['generated_text']

            # Parse generated text
            lines = generated.split('\n')
            question_data = {
                'type': 'true_false',
                'statement': '',
                'correct_answer': '',
                'explanation': ''
            }

            for line in lines:
                if line.startswith('Statement:'):
                    question_data['statement'] = line.replace('Statement:', '').strip()
                elif line.startswith('Correct Answer:'):
                    question_data['correct_answer'] = line.replace('Correct Answer:', '').strip()
                elif line.startswith('Explanation:'):
                    question_data['explanation'] = line.replace('Explanation:', '').strip()

            return question_data

        except:
            # Fallback question
            return {
                'type': 'true_false',
                'statement': f'{topic} is a fundamental concept in cloud computing.',
                'correct_answer': 'True',
                'explanation': f'{topic} is indeed a core concept in cloud computing.'
            }

    def _generate_short_answer_question(self, topic: str, context: str) -> Dict:
        """Generate short answer question"""
        return {
            'type': 'short_answer',
            'question': f'Explain {topic} in the context of cloud computing.',
            'expected_answer': f'{topic} is a cloud computing concept that enables...',
            'evaluation_criteria': [
                'Mentions scalability',
                'Discusses on-demand nature',
                'Includes practical examples'
            ]
        }

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0

    def classify_difficulty(self, text: str) -> str:
        """Classify text difficulty level"""
        if not self.nlp:
            return "intermediate"

        try:
            doc = self.nlp(text[:1000])

            # Analyze text characteristics
            num_sentences = len(list(doc.sents))
            num_words = len([token for token in doc if not token.is_punct])

            if num_sentences == 0:
                return "beginner"

            avg_sentence_length = num_words / num_sentences

            # Count complex words (more than 2 syllables or technical terms)
            complex_words = 0
            for token in doc:
                if len(token.text) > 8 or token.pos_ in ['NOUN', 'VERB']:
                    complex_words += 1

            complexity_ratio = complex_words / num_words if num_words > 0 else 0

            # Determine difficulty
            if avg_sentence_length < 15 and complexity_ratio < 0.2:
                return "beginner"
            elif avg_sentence_length < 25 and complexity_ratio < 0.4:
                return "intermediate"
            else:
                return "advanced"

        except Exception as e:
            logger.error(f"Difficulty classification error: {e}")
            return "intermediate"

    @lru_cache(maxsize=100)
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings for text with caching"""
        return self.embedding_model.encode([text])[0]

    def cluster_concepts(self, concepts: List[str]) -> List[List[str]]:
        """Cluster related concepts"""
        if not concepts:
            return []

        try:
            # Get embeddings for all concepts
            embeddings = self.embedding_model.encode(concepts)

            # Simple clustering using cosine similarity
            from sklearn.cluster import DBSCAN

            # Convert to similarity matrix
            similarity_matrix = np.zeros((len(concepts), len(concepts)))
            for i in range(len(concepts)):
                for j in range(len(concepts)):
                    similarity_matrix[i][j] = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )

            # Use DBSCAN for clustering
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='precomputed').fit(1 - similarity_matrix)

            # Group concepts by cluster
            clusters = {}
            for i, label in enumerate(clustering.labels_):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(concepts[i])

            # Filter out noise (label = -1)
            result = [clusters[label] for label in clusters if label != -1]

            return result

        except Exception as e:
            logger.error(f"Clustering error: {e}")
            return [concepts]  # Return single cluster as fallback

    def generate_learning_path(self, topics: List[str],
                               student_level: str = "beginner") -> List[Dict]:
        """Generate personalized learning path"""
        try:
            # Sort topics by complexity
            topic_complexity = []
            for topic in topics:
                difficulty = self.classify_difficulty(topic)
                complexity_score = {"beginner": 1, "intermediate": 2, "advanced": 3}.get(difficulty, 2)
                topic_complexity.append((topic, complexity_score))

            # Sort by complexity
            topic_complexity.sort(key=lambda x: x[1])

            # Create learning path
            learning_path = []
            for i, (topic, complexity) in enumerate(topic_complexity):
                # Adjust based on student level
                if student_level == "beginner" and complexity > 2:
                    continue  # Skip advanced topics for beginners

                learning_path.append({
                    'step': i + 1,
                    'topic': topic,
                    'difficulty': ["beginner", "intermediate", "advanced"][complexity - 1],
                    'estimated_time': complexity * 30,  # minutes
                    'prerequisites': self._get_prerequisites(topic, topics[:i]),
                    'learning_objectives': self._generate_learning_objectives(topic)
                })

            return learning_path

        except Exception as e:
            logger.error(f"Learning path generation error: {e}")
            return []

    def _get_prerequisites(self, topic: str, previous_topics: List[str]) -> List[str]:
        """Get prerequisites for a topic"""
        # Simple heuristic: topics mentioned in previous steps
        return previous_topics[:2]  # First two topics as prerequisites

    def _generate_learning_objectives(self, topic: str) -> List[str]:
        """Generate learning objectives for a topic"""
        objectives = [
            f"Understand the concept of {topic}",
            f"Identify use cases for {topic}",
            f"Configure basic {topic} implementations",
            f"Troubleshoot common {topic} issues"
        ]
        return objectives[:3]