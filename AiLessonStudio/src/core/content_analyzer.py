import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from datetime import datetime


@dataclass
class ContentAnalysis:
    """Content analysis results"""
    readability: float
    complexity: str
    key_terms: List[str]
    structure_score: float
    recommendations: List[str]
    metadata: Dict[str, Any]


class ContentAnalyzer:
    """Advanced content analyzer for educational materials"""

    def __init__(self, config):
        self.config = config

        # Initialize NLP components
        self._initialize_nlp()

        # Technical term database
        self.technical_terms = self._load_technical_terms()

        print("✅ Content Analyzer initialized")

    def _initialize_nlp(self):
        """Initialize NLP components"""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
            print("⚠️ spaCy not available, using basic analysis")

    def _load_technical_terms(self) -> Dict[str, List[str]]:
        """Load technical term database"""
        return {
            'cloud_computing': [
                'iaas', 'paas', 'saas', 'virtualization', 'containerization',
                'serverless', 'microservices', 'orchestration', 'elasticity',
                'scalability', 'availability', 'reliability', 'fault tolerance'
            ],
            'aws': [
                'ec2', 's3', 'lambda', 'rds', 'dynamodb', 'vpc', 'iam',
                'cloudfront', 'route53', 'elastic beanstalk', 'cloudformation'
            ],
            'azure': [
                'virtual machines', 'app service', 'functions', 'sql database',
                'cosmos db', 'virtual network', 'active directory', 'cdn',
                'dns', 'service fabric', 'resource manager'
            ],
            'gcp': [
                'compute engine', 'cloud storage', 'cloud functions',
                'cloud sql', 'bigtable', 'vpc', 'cloud iam', 'cloud cdn',
                'cloud dns', 'kubernetes engine', 'deployment manager'
            ],
            'security': [
                'encryption', 'authentication', 'authorization', 'firewall',
                'vpn', 'ssl/tls', 'compliance', 'auditing', 'monitoring',
                'vulnerability', 'penetration testing', 'security groups'
            ]
        }

    def analyze_text(self, text: str, content_type: str = "lesson") -> ContentAnalysis:
        """Analyze text content for educational quality"""

        if not text:
            return ContentAnalysis(
                readability=0.0,
                complexity='unknown',
                key_terms=[],
                structure_score=0.0,
                recommendations=['No text provided for analysis'],
                metadata={'error': 'Empty text'}
            )

        # 1. Readability analysis
        readability_score = self._calculate_readability(text)

        # 2. Complexity analysis
        complexity_level = self._assess_complexity(text)

        # 3. Key term extraction
        key_terms = self._extract_key_terms(text)

        # 4. Structure analysis
        structure_score = self._analyze_structure(text, content_type)

        # 5. Generate recommendations
        recommendations = self._generate_recommendations(
            readability_score, complexity_level, structure_score, key_terms
        )

        # 6. Collect metadata
        metadata = self._collect_metadata(text, content_type)

        return ContentAnalysis(
            readability=readability_score,
            complexity=complexity_level,
            key_terms=key_terms,
            structure_score=structure_score,
            recommendations=recommendations,
            metadata=metadata
        )

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (0-100)"""
        # Simple Flesch Reading Ease approximation
        sentences = re.split(r'[.!?]+', text)
        words = text.split()

        if len(sentences) == 0 or len(words) == 0:
            return 50.0  # Default score

        avg_sentence_length = len(words) / len(sentences)

        # Count syllables (approximation)
        syllable_count = 0
        for word in words:
            word = word.lower()
            if len(word) <= 3:
                syllable_count += 1
            else:
                # Simple syllable counting
                syllable_count += len(re.findall(r'[aeiouy]+', word))

        avg_syllables = syllable_count / len(words)

        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)

        # Normalize to 0-100
        return max(0.0, min(100.0, score))

    def _assess_complexity(self, text: str) -> str:
        """Assess text complexity level"""
        readability = self._calculate_readability(text)

        if readability >= 70:
            return "beginner"
        elif readability >= 50:
            return "intermediate"
        elif readability >= 30:
            return "advanced"
        else:
            return "expert"

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key technical terms from text"""
        text_lower = text.lower()
        found_terms = []

        # Check for technical terms
        for category, terms in self.technical_terms.items():
            for term in terms:
                if term in text_lower:
                    found_terms.append(term)

        # Use NLP for additional term extraction
        if self.nlp and len(text) < 10000:  # Limit for performance
            try:
                doc = self.nlp(text[:5000])  # Limit text length

                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) <= 3:  # Single or compound nouns
                        term = chunk.text.lower()
                        if term not in found_terms and len(term) > 3:
                            found_terms.append(term)
            except:
                pass

        # Deduplicate and limit
        unique_terms = list(set(found_terms))
        return unique_terms[:15]  # Return top 15 terms

    def _analyze_structure(self, text: str, content_type: str) -> float:
        """Analyze content structure (0-1 score)"""
        score = 0.0
        features_found = 0
        total_features = 5

        # 1. Check for headings/sections
        if re.search(r'(?i)(?:^|\n)(?:#+|==+|--+|chapter|section)', text):
            score += 0.2
            features_found += 1

        # 2. Check for lists
        if re.search(r'(?i)(?:^|\n)[•\-\*]\s|\d+\.\s', text):
            score += 0.2
            features_found += 1

        # 3. Check for examples/code blocks
        if re.search(r'```|example:|for example|e\.g\.', text, re.IGNORECASE):
            score += 0.2
            features_found += 1

        # 4. Check for definitions
        if re.search(r'(?i)define|definition|means|refers to', text):
            score += 0.2
            features_found += 1

        # 5. Check for summary/conclusion
        if re.search(r'(?i)summary|conclusion|key points|takeaways', text):
            score += 0.2
            features_found += 1

        # Calculate final score
        if total_features > 0:
            score = features_found / total_features

        return score

    def _generate_recommendations(self, readability: float, complexity: str,
                                  structure_score: float, key_terms: List[str]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        # Readability recommendations
        if readability < 40:
            recommendations.append("Consider simplifying sentence structure for better readability")
        elif readability > 80:
            recommendations.append("Content is very readable - good for beginners")

        # Structure recommendations
        if structure_score < 0.4:
            recommendations.append("Add more structure with headings, lists, and examples")
        elif structure_score < 0.7:
            recommendations.append("Consider adding more examples and a summary section")

        # Technical content recommendations
        if len(key_terms) < 5:
            recommendations.append("Include more technical terms relevant to cloud computing")
        elif len(key_terms) > 20:
            recommendations.append("Consider explaining technical terms for better understanding")

        # Complexity-specific recommendations
        if complexity == "expert":
            recommendations.append("Add more explanations for complex concepts")
        elif complexity == "beginner":
            recommendations.append("Consider adding some intermediate concepts for challenge")

        return recommendations[:5]  # Limit to 5 recommendations

    def _collect_metadata(self, text: str, content_type: str) -> Dict[str, Any]:
        """Collect metadata about the content"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')

        # Calculate various metrics
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        avg_sentence_length = len(words) / max(1, len(sentences))

        # Technical term density
        technical_terms_count = sum(
            1 for term in self._extract_key_terms(text)
            if any(term in category_terms for category_terms in self.technical_terms.values())
        )
        term_density = technical_terms_count / max(1, len(words))

        # Code/example detection
        has_code = bool(re.search(r'```|\$ |command:|sudo ', text))
        has_examples = bool(re.search(r'example:|for example|e\.g\.', text, re.IGNORECASE))

        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'technical_terms': technical_terms_count,
            'term_density': round(term_density, 4),
            'has_code': has_code,
            'has_examples': has_examples,
            'content_type': content_type,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def compare_contents(self, content1: str, content2: str) -> Dict[str, Any]:
        """Compare two pieces of content"""
        analysis1 = self.analyze_text(content1)
        analysis2 = self.analyze_text(content2)

        # Calculate similarity scores
        readability_diff = abs(analysis1.readability - analysis2.readability)
        structure_diff = abs(analysis1.structure_score - analysis2.structure_score)

        # Term overlap
        terms1 = set(analysis1.key_terms)
        terms2 = set(analysis2.key_terms)
        term_overlap = len(terms1.intersection(terms2)) / max(1, len(terms1.union(terms2)))

        # Complexity comparison
        complexity_levels = ['beginner', 'intermediate', 'advanced', 'expert']
        complexity_diff = abs(
            complexity_levels.index(analysis1.complexity) -
            complexity_levels.index(analysis2.complexity)
        )

        return {
            'similarity_score': round((1 - readability_diff / 100 + term_overlap) / 2, 3),
            'readability_comparison': {
                'content1': round(analysis1.readability, 1),
                'content2': round(analysis2.readability, 1),
                'difference': round(readability_diff, 1)
            },
            'complexity_comparison': {
                'content1': analysis1.complexity,
                'content2': analysis2.complexity,
                'level_difference': complexity_diff
            },
            'term_overlap': round(term_overlap, 3),
            'structure_comparison': {
                'content1': round(analysis1.structure_score, 3),
                'content2': round(analysis2.structure_score, 3),
                'difference': round(structure_diff, 3)
            },
            'recommendations': self._generate_comparison_recommendations(
                analysis1, analysis2, term_overlap, readability_diff
            )
        }

    def _generate_comparison_recommendations(self, analysis1: ContentAnalysis,
                                             analysis2: ContentAnalysis,
                                             term_overlap: float,
                                             readability_diff: float) -> List[str]:
        """Generate recommendations based on content comparison"""
        recommendations = []

        if readability_diff > 30:
            recommendations.append(
                "Contents have significantly different readability levels. "
                "Consider adjusting for consistent audience level."
            )

        if term_overlap < 0.3:
            recommendations.append(
                "Contents cover different technical topics. "
                "Consider adding bridging material if they're meant to be connected."
            )

        if analysis1.complexity != analysis2.complexity:
            recommendations.append(
                f"Complexity levels differ ({analysis1.complexity} vs {analysis2.complexity}). "
                "Consider rebalancing for consistent learning progression."
            )

        return recommendations

    def generate_content_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate a summary of the content"""
        if not text:
            return "No content to summarize"

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= max_sentences:
            return '. '.join(sentences) + '.'

        # Simple extractive summarization
        # Score sentences by position and length
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0

            # Favor first sentences
            if i == 0:
                score += 2
            elif i < 3:
                score += 1

            # Favor sentences with key terms
            key_terms = self._extract_key_terms(sentence)
            score += len(key_terms) * 0.5

            # Favor medium-length sentences
            word_count = len(sentence.split())
            if 10 <= word_count <= 30:
                score += 1

            scored_sentences.append((score, sentence))

        # Select top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        selected = [s[1] for s in scored_sentences[:max_sentences]]

        # Reorder to maintain some original order
        selected.sort(key=lambda x: sentences.index(x))

        return '. '.join(selected) + '.'

    def extract_learning_objectives(self, text: str) -> List[str]:
        """Extract potential learning objectives from content"""
        objectives = []

        # Look for objective patterns
        patterns = [
            r'learn(?:ing)? (?:objective|goal)s?[:\s]+(.+?)(?=\n\n|\Z)',
            r'by the end of (?:this|the).*?you will be able to[:\s]+(.+?)(?=\n\n|\Z)',
            r'understand(?:ing)?[:\s]+(.+?)(?=\n\n|\Z)',
            r'key (?:takeaway|point)s?[:\s]+(.+?)(?=\n\n|\Z)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split into individual objectives
                lines = match.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and len(line.split()) >= 3:
                        objectives.append(line)

        # If no objectives found, generate from key terms
        if not objectives:
            key_terms = self._extract_key_terms(text)
            for term in key_terms[:5]:
                objectives.append(f"Understand the concept of {term}")
                objectives.append(f"Apply {term} in practical scenarios")

        return list(set(objectives))[:5]  # Return unique objectives, max 5

    def assess_content_quality(self, text: str) -> Dict[str, Any]:
        """Comprehensive content quality assessment"""
        analysis = self.analyze_text(text)

        # Calculate overall quality score
        quality_score = (
                min(100, analysis.readability) * 0.3 +
                analysis.structure_score * 100 * 0.3 +
                len(analysis.key_terms) * 5 * 0.2 +
                (5 - len(analysis.recommendations)) * 20 * 0.2
        )

        # Determine quality level
        if quality_score >= 80:
            quality_level = "excellent"
        elif quality_score >= 65:
            quality_level = "good"
        elif quality_score >= 50:
            quality_level = "adequate"
        else:
            quality_level = "needs_improvement"

        return {
            'overall_score': round(quality_score, 1),
            'quality_level': quality_level,
            'readability_score': round(analysis.readability, 1),
            'structure_score': round(analysis.structure_score, 3),
            'technical_depth': len(analysis.key_terms),
            'key_strengths': self._identify_strengths(analysis),
            'improvement_areas': analysis.recommendations,
            'suitable_for': self._determine_audience(analysis)
        }

    def _identify_strengths(self, analysis: ContentAnalysis) -> List[str]:
        """Identify content strengths"""
        strengths = []

        if analysis.readability >= 70:
            strengths.append("High readability for target audience")
        elif analysis.readability >= 50:
            strengths.append("Appropriate readability level")

        if analysis.structure_score >= 0.7:
            strengths.append("Well-structured content")
        elif analysis.structure_score >= 0.5:
            strengths.append("Adequate structure")

        if len(analysis.key_terms) >= 10:
            strengths.append("Good technical depth")
        elif len(analysis.key_terms) >= 5:
            strengths.append("Adequate technical coverage")

        return strengths

    def _determine_audience(self, analysis: ContentAnalysis) -> List[str]:
        """Determine suitable audience levels"""
        audience = []

        if analysis.complexity == "beginner":
            audience.append("Beginner learners")
            if analysis.readability >= 70:
                audience.append("Complete beginners")

        elif analysis.complexity == "intermediate":
            audience.append("Intermediate learners")
            if len(analysis.key_terms) >= 8:
                audience.append("Practitioners seeking depth")

        elif analysis.complexity == "advanced":
            audience.append("Advanced learners")
            if len(analysis.key_terms) >= 12:
                audience.append("Experienced professionals")

        elif analysis.complexity == "expert":
            audience.append("Expert audience")
            audience.append("Specialists in the field")

        return audience