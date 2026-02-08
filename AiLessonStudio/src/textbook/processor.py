import os
import json
import re
import warnings
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pdfplumber
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from docx import Document
import spacy
import numpy as np
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
import networkx as nx
from dataclasses import dataclass
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

warnings.filterwarnings('ignore')


@dataclass
class Chapter:
    """Chapter data structure"""
    number: str
    title: str
    content: str
    start_page: int
    end_page: int
    key_concepts: List[str]
    learning_objectives: List[str]
    summary: str
    difficulty: str


@dataclass
class Concept:
    """Concept data structure"""
    name: str
    definition: str
    examples: List[str]
    related_concepts: List[str]
    importance_score: float
    difficulty_level: str
    textbook_references: List[str]


class TextbookProcessor:
    """Textbook processor for educational content"""

    def __init__(self, config):
        self.config = config

    def process(self, content):
        """Process textbook content"""
        # Delegate to AdvancedTextbookProcessor for backward compatibility
        processor = AdvancedTextbookProcessor(self.config)

        # If content is a file path, load it
        if isinstance(content, str) and os.path.exists(content):
            success = processor.load_textbook(content)
            if success:
                return processor
            else:
                raise ValueError(f"Failed to load textbook from {content}")
        else:
            # Assume content is already extracted text
            processor.full_text = str(content)
            processor._extract_chapters_advanced(str(content))
            processor._perform_advanced_analysis()
            processor._build_knowledge_graph()
            return processor


class AdvancedTextbookProcessor:
    """Advanced textbook processor with AI-powered analysis"""

    def __init__(self, config):
        self.config = config
        self.textbook_id = None
        self.textbook_metadata = {}
        self.chapters: Dict[str, Chapter] = {}
        self.concepts: Dict[str, Concept] = {}
        self.knowledge_graph = nx.Graph()

        # Load NLP models
        self._load_models()

        # Initialize extractors
        self._init_extractors()

        print("✅ Advanced Textbook Processor initialized")

    def _load_models(self):
        """Load all required NLP models"""
        try:
            # Load spaCy for NLP
            self.nlp = spacy.load("en_core_web_lg")
            self.nlp.max_length = 3000000

            # Load transformers models
            print("📥 Loading transformer models...")

            # For NER and concept extraction
            self.ner_pipeline = pipeline(
                "ner",
                model=self.config.models.NER_MODEL,
                aggregation_strategy="simple"
            )

            # For text summarization
            self.summarizer = pipeline(
                "summarization",
                model=self.config.models.SUMMARIZATION_MODEL
            )

            # For embeddings
            self.embedding_model = SentenceTransformer(
                self.config.models.EMBEDDING_MODEL
            )

            # For question answering (loaded on demand)
            self.qa_tokenizer = None
            self.qa_model = None

            print("✅ All models loaded successfully")

        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            print("⚠️ Using fallback methods...")
            self.nlp = None
            self.ner_pipeline = None
            self.summarizer = None
            self.embedding_model = None

    def _init_extractors(self):
        """Initialize specialized extractors"""
        from .diagram_extractor import DiagramExtractor
        from .ocr_handler import OCRHandler
        from .table_extractor import TableExtractor

        self.diagram_extractor = DiagramExtractor(self.config)
        self.ocr_handler = OCRHandler(self.config)
        self.table_extractor = TableExtractor()

    def load_textbook(self, file_path: str, textbook_name: str = None) -> bool:
        """Load and process textbook from file"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                return False

            # Generate textbook ID
            self.textbook_id = hashlib.md5(file_path.read_bytes()).hexdigest()[:16]

            # Extract metadata
            self.textbook_metadata = {
                'id': self.textbook_id,
                'name': textbook_name or file_path.stem,
                'path': str(file_path),
                'size': file_path.stat().st_size,
                'extension': file_path.suffix.lower(),
                'upload_time': datetime.now().isoformat()
            }

            print(f"📖 Processing textbook: {self.textbook_metadata['name']}")

            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                success = self._process_pdf(file_path)
            elif file_path.suffix.lower() == '.docx':
                success = self._process_docx(file_path)
            elif file_path.suffix.lower() == '.txt':
                success = self._process_txt(file_path)
            else:
                print(f"❌ Unsupported file format: {file_path.suffix}")
                return False

            if success:
                # Perform advanced analysis
                self._perform_advanced_analysis()

                # Build knowledge graph
                self._build_knowledge_graph()

                # Save processed data
                self._save_processed_data()

                print(f"✅ Textbook processing complete!")
                print(f"   Chapters: {len(self.chapters)}")
                print(f"   Concepts: {len(self.concepts)}")
                print(f"   Knowledge Graph Nodes: {self.knowledge_graph.number_of_nodes()}")

                return True
            else:
                return False

        except Exception as e:
            print(f"❌ Error loading textbook: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _process_pdf(self, file_path: Path) -> bool:
        """Process PDF file with advanced extraction"""
        try:
            print("📄 Processing PDF with advanced extraction...")

            # Use multiple extraction methods
            extracted_texts = []

            # Method 1: pdfplumber (for text-based PDFs)
            print("  Using pdfplumber for text extraction...")
            try:
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        extracted_texts.append((i, text, "pdfplumber"))

                        # Extract tables
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                # Process table data
                                pass
            except Exception as e:
                print(f"  pdfplumber extraction failed: {e}")

            # Method 2: PyMuPDF (alternative)
            print("  Using PyMuPDF for text extraction...")
            try:
                doc = fitz.open(file_path)
                for i, page in enumerate(doc):
                    text = page.get_text()
                    extracted_texts.append((i, text, "pymupdf"))

                    # Extract images for OCR if needed
                    if self.config.processing.USE_OCR:
                        images = page.get_images()
                        for img in images:
                            # Process images with OCR
                            pass
                doc.close()
            except Exception as e:
                print(f"  PyMuPDF extraction failed: {e}")

            # Combine and deduplicate text
            combined_text = self._combine_extracted_texts(extracted_texts)

            # Process combined text
            self.full_text = combined_text

            # Extract chapters using multiple strategies
            self._extract_chapters_advanced(combined_text)

            # Extract diagrams and figures
            if self.config.processing.EXTRACT_DIAGRAMS:
                self._extract_diagrams(file_path)

            return True

        except Exception as e:
            print(f"❌ PDF processing error: {e}")
            return False

    def _combine_extracted_texts(self, extracted_texts: List[Tuple[int, str, str]]) -> str:
        """Combine and deduplicate texts from multiple extractors"""
        # Group by page
        page_texts = {}
        for page_num, text, source in extracted_texts:
            if page_num not in page_texts:
                page_texts[page_num] = []
            page_texts[page_num].append(text)

        # For each page, choose the best text
        combined_pages = []
        for page_num in sorted(page_texts.keys()):
            texts = page_texts[page_num]
            if texts:
                # Choose the longest non-empty text
                valid_texts = [t for t in texts if t.strip()]
                if valid_texts:
                    best_text = max(valid_texts, key=len)
                    combined_pages.append(f"--- Page {page_num + 1} ---\n{best_text}")

        return "\n\n".join(combined_pages)

    def _extract_chapters_advanced(self, text: str):
        """Advanced chapter extraction with multiple strategies"""
        print("🔍 Extracting chapters with advanced methods...")

        strategies = [
            self._extract_by_numbered_chapters,
            self._extract_by_heading_patterns,
            self._extract_by_ml_patterns,
            self._extract_by_semantic_segmentation
        ]

        best_chapters = None
        best_score = -1

        for strategy in strategies:
            try:
                chapters = strategy(text)
                score = self._evaluate_chapter_quality(chapters)

                if score > best_score and len(chapters) >= 2:
                    best_score = score
                    best_chapters = chapters

            except Exception as e:
                print(f"  Strategy failed: {strategy.__name__}, error: {e}")

        if best_chapters:
            self._process_extracted_chapters(best_chapters)
        else:
            # Fallback: create single chapter
            self._create_fallback_chapter(text)

    def _extract_by_numbered_chapters(self, text: str) -> Dict[str, Dict]:
        """Extract chapters using numbered patterns"""
        chapters = {}

        # Multiple patterns for chapter detection
        patterns = [
            # Chapter X: Title
            r'(?i)(?:^|\n)\s*(?:chapter|chap|ch\.?)\s+(\d+(?:\.\d+)?)[:.\s]+\s*(.+?)(?=\n\s*(?:chapter|chap|ch\.?)\s+\d+|\n\s*\d+\.\s+|\n\s*[A-Z][A-Z\s]{10,}|\Z)',
            # X. Title (numbered sections)
            r'(?i)(?:^|\n)\s*(\d+(?:\.\d+)*)\.\s+(.+?)(?=\n\s*\d+(?:\.\d+)*\.\s+|\n\s*[A-Z][A-Z\s]{10,}|\Z)',
            # CHAPTER X in all caps
            r'(?:^|\n)\s*CHAPTER\s+(\d+(?:\.\d+)?)[:.\s]+\s*(.+?)(?=\n\s*CHAPTER\s+\d+|\Z)',
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.DOTALL | re.MULTILINE))
            if len(matches) >= 3:  # Need at least 3 chapters
                for match in matches:
                    chapter_num = match.group(1)
                    chapter_title = match.group(2).strip()
                    chapter_content = match.group(0)

                    chapter_id = f"chapter_{chapter_num}"
                    chapters[chapter_id] = {
                        'number': chapter_num,
                        'title': chapter_title,
                        'content': chapter_content,
                        'raw_match': match.group(0)
                    }
                break

        return chapters

    def _extract_by_heading_patterns(self, text: str) -> Dict[str, Dict]:
        """Extract chapters using heading patterns"""
        chapters = {}
        lines = text.split('\n')

        current_chapter = None
        chapter_content = []
        chapter_num = 1

        heading_patterns = [
            r'^\s*\d+[\.\)]\s+.+$',  # Numbered headings
            r'^\s*[A-Z][A-Z\s]{5,}$',  # All caps headings
            r'^\s*.+:\s*$',  # Headings ending with colon
            r'^\s*(?:introduction|conclusion|summary|references|bibliography)\s*$',  # Common sections
        ]

        for i, line in enumerate(lines):
            line = line.strip()

            # Check if line is a heading
            is_heading = any(re.match(pattern, line, re.IGNORECASE) for pattern in heading_patterns)

            if is_heading and len(line) < 200:  # Heading should be reasonably short
                # Save previous chapter
                if current_chapter and chapter_content:
                    chapter_id = f"chapter_{chapter_num}"
                    chapters[chapter_id] = {
                        'number': str(chapter_num),
                        'title': current_chapter,
                        'content': '\n'.join(chapter_content),
                        'start_line': i - len(chapter_content)
                    }
                    chapter_num += 1

                # Start new chapter
                current_chapter = line
                chapter_content = []
            elif current_chapter:
                chapter_content.append(line)

        # Add last chapter
        if current_chapter and chapter_content:
            chapter_id = f"chapter_{chapter_num}"
            chapters[chapter_id] = {
                'number': str(chapter_num),
                'title': current_chapter,
                'content': '\n'.join(chapter_content)
            }

        return chapters

    def _extract_by_ml_patterns(self, text: str) -> Dict[str, Dict]:
        """Extract chapters using ML-based pattern recognition"""
        if not self.nlp:
            return {}

        # Use NLP to identify section boundaries
        doc = self.nlp(text[:100000])  # Process first 100k chars

        chapters = {}
        current_section = []
        section_num = 1
        in_section = False

        for sent in doc.sents:
            sent_text = sent.text.strip()

            # Check if sentence looks like a section heading
            is_heading = (
                    len(sent_text.split()) <= 10 and  # Short
                    sent_text[0].isupper() and  # Starts with capital
                    any(word.tag_ in ['NNP', 'NN'] for word in sent) and  # Contains nouns
                    not sent_text.endswith('.')  # Doesn't end with period
            )

            if is_heading:
                if in_section and current_section:
                    chapter_id = f"section_{section_num}"
                    chapters[chapter_id] = {
                        'number': str(section_num),
                        'title': current_section[0] if current_section else f"Section {section_num}",
                        'content': ' '.join(current_section)
                    }
                    section_num += 1
                    current_section = []

                in_section = True
                current_section.append(sent_text)
            elif in_section:
                current_section.append(sent_text)

        # Add final section
        if current_section:
            chapter_id = f"section_{section_num}"
            chapters[chapter_id] = {
                'number': str(section_num),
                'title': current_section[0] if current_section else f"Section {section_num}",
                'content': ' '.join(current_section)
            }

        return chapters

    def _extract_by_semantic_segmentation(self, text: str) -> Dict[str, Dict]:
        """Extract chapters using semantic segmentation"""
        # Split text into segments based on content changes
        sentences = re.split(r'[.!?]+', text)

        chapters = {}
        current_chunk = []
        chunk_num = 1

        # Simple chunking by sentence count
        chunk_size = 50  # sentences per chunk
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:
                current_chunk.append(sentence)

                if len(current_chunk) >= chunk_size:
                    # Create chapter from chunk
                    chapter_id = f"chunk_{chunk_num}"
                    chapters[chapter_id] = {
                        'number': str(chunk_num),
                        'title': f"Section {chunk_num}",
                        'content': ' '.join(current_chunk)
                    }
                    chunk_num += 1
                    current_chunk = []

        # Add final chunk
        if current_chunk:
            chapter_id = f"chunk_{chunk_num}"
            chapters[chapter_id] = {
                'number': str(chunk_num),
                'title': f"Section {chunk_num}",
                'content': ' '.join(current_chunk)
            }

        return chapters

    def _evaluate_chapter_quality(self, chapters: Dict) -> float:
        """Evaluate the quality of extracted chapters"""
        if not chapters:
            return 0.0

        scores = []

        for chapter_id, chapter in chapters.items():
            chapter_score = 0.0

            # Score based on title quality
            title = chapter.get('title', '')
            if title:
                # Good titles are 5-50 chars, not all caps, contain meaningful words
                title_len = len(title)
                if 5 <= title_len <= 50:
                    chapter_score += 0.3
                if not title.isupper():
                    chapter_score += 0.2

            # Score based on content quality
            content = chapter.get('content', '')
            if content:
                content_len = len(content)
                # Ideal chapter length: 1000-5000 chars
                if 1000 <= content_len <= 10000:
                    chapter_score += 0.3
                elif content_len > 500:
                    chapter_score += 0.2

            # Score based on structure
            if 'number' in chapter:
                chapter_score += 0.2

            scores.append(chapter_score)

        return sum(scores) / len(scores) if scores else 0.0

    def _process_extracted_chapters(self, raw_chapters: Dict[str, Dict]):
        """Process extracted chapters into structured format"""
        for chapter_id, raw_data in raw_chapters.items():
            try:
                # Extract key concepts from chapter
                key_concepts = self._extract_key_concepts_from_text(raw_data['content'])

                # Generate learning objectives
                learning_objectives = self._generate_learning_objectives(
                    raw_data['content'],
                    key_concepts
                )

                # Generate summary
                summary = self._summarize_text(raw_data['content'])

                # Create chapter object
                chapter = Chapter(
                    number=raw_data.get('number', '1'),
                    title=raw_data.get('title', f'Chapter {raw_data.get("number", "1")}'),
                    content=raw_data['content'],
                    start_page=raw_data.get('start_page', 0),
                    end_page=raw_data.get('end_page', 0),
                    key_concepts=key_concepts,
                    learning_objectives=learning_objectives,
                    summary=summary,
                    difficulty=self._assess_difficulty(raw_data['content'])
                )

                self.chapters[chapter_id] = chapter

                # Extract concepts from this chapter
                self._extract_concepts_from_chapter(chapter)

            except Exception as e:
                print(f"⚠️ Error processing chapter {chapter_id}: {e}")

    def _extract_key_concepts_from_text(self, text: str, max_concepts: int = 10) -> List[str]:
        """Extract key concepts from text using NLP"""
        concepts = set()

        # Method 1: Use NER pipeline
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(text[:5000])  # Limit text length
                for entity in ner_results:
                    if entity['score'] > 0.8 and entity['word'] not in concepts:
                        concepts.add(entity['word'])
                        if len(concepts) >= max_concepts:
                            break
            except:
                pass

        # Method 2: Use spaCy noun phrases
        if self.nlp and len(concepts) < max_concepts:
            try:
                doc = self.nlp(text[:2000])
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) <= 3:  # Single or compound nouns
                        concepts.add(chunk.text)
                        if len(concepts) >= max_concepts:
                            break
            except:
                pass

        # Method 3: Keyword extraction
        cloud_keywords = [
            'cloud computing', 'aws', 'amazon web services', 'azure',
            'google cloud', 'ec2', 's3', 'lambda', 'vpc', 'rds',
            'iaas', 'paas', 'saas', 'virtualization', 'containers',
            'docker', 'kubernetes', 'serverless', 'microservices',
            'load balancing', 'auto scaling', 'elasticity', 'devops',
            'ci/cd', 'infrastructure as code', 'monitoring', 'security'
        ]

        text_lower = text.lower()
        for keyword in cloud_keywords:
            if keyword in text_lower and len(concepts) < max_concepts:
                concepts.add(keyword.title())

        return list(concepts)[:max_concepts]

    def _generate_learning_objectives(self, text: str, key_concepts: List[str]) -> List[str]:
        """Generate learning objectives from text"""
        objectives = []

        # Template-based generation
        templates = [
            "Understand the concept of {concept}",
            "Learn how to implement {concept}",
            "Compare {concept} with related technologies",
            "Identify use cases for {concept}",
            "Configure and manage {concept}",
            "Troubleshoot common issues with {concept}",
            "Design systems using {concept}",
            "Evaluate the benefits of {concept}"
        ]

        for concept in key_concepts[:5]:  # Use top 5 concepts
            for template in templates[:3]:  # Use first 3 templates
                objective = template.format(concept=concept.lower())
                objectives.append(objective)

        return list(set(objectives))[:5]  # Return unique objectives, max 5

    def _summarize_text(self, text: str, max_length: int = 200) -> str:
        """Summarize text using AI or extractive methods"""
        # If summarizer is available, use it
        if self.summarizer and len(text) > 500:
            try:
                summary = self.summarizer(
                    text[:2000],  # Limit input length
                    max_length=max_length,
                    min_length=50,
                    do_sample=False
                )[0]['summary_text']
                return summary
            except:
                pass

        # Fallback: extractive summarization
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) <= 3:
            return text

        # Simple heuristic: take first and last sentences
        summary_sentences = []
        if len(sentences) > 0:
            summary_sentences.append(sentences[0].strip())
        if len(sentences) > 2:
            summary_sentences.append(sentences[-2].strip())

        return '. '.join(summary_sentences) + '.'

    def _assess_difficulty(self, text: str) -> str:
        """Assess difficulty level of text"""
        # Simple heuristic based on text characteristics
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))

        if sentence_count == 0:
            return "beginner"

        avg_sentence_length = word_count / sentence_count

        if avg_sentence_length < 15:
            return "beginner"
        elif avg_sentence_length < 25:
            return "intermediate"
        else:
            return "advanced"

    def _extract_concepts_from_chapter(self, chapter: Chapter):
        """Extract and structure concepts from chapter"""
        # Combine chapter content with key concepts
        text_for_concepts = f"{chapter.title}\n\n{chapter.content}"

        # Use NLP to extract concept definitions
        if self.nlp:
            try:
                doc = self.nlp(text_for_concepts[:5000])

                # Look for definition patterns
                definition_patterns = [
                    r'(\w+(?:\s+\w+)*)\s+is\s+(?:a|an|the)\s+(.+)',
                    r'(\w+(?:\s+\w+)*)\s+refers to\s+(.+)',
                    r'(\w+(?:\s+\w+)*)\s+means\s+(.+)',
                    r'(\w+(?:\s+\w+)*)\s+can be defined as\s+(.+)'
                ]

                for pattern in definition_patterns:
                    matches = re.finditer(pattern, text_for_concepts, re.IGNORECASE)
                    for match in matches:
                        concept_name = match.group(1).strip()
                        definition = match.group(2).strip()

                        if concept_name and definition and len(concept_name.split()) <= 3:
                            # Create or update concept
                            if concept_name not in self.concepts:
                                self.concepts[concept_name] = Concept(
                                    name=concept_name,
                                    definition=definition,
                                    examples=[],
                                    related_concepts=[],
                                    importance_score=0.5,
                                    difficulty_level=chapter.difficulty,
                                    textbook_references=[chapter.title]
                                )
                            else:
                                # Update existing concept
                                self.concepts[concept_name].textbook_references.append(chapter.title)

            except Exception as e:
                print(f"⚠️ Error extracting concepts: {e}")

    def _perform_advanced_analysis(self):
        """Perform advanced analysis on extracted content"""
        print("🧠 Performing advanced analysis...")

        # Parallel processing for efficiency
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            # Analyze each chapter
            for chapter_id, chapter in self.chapters.items():
                futures.append(
                    executor.submit(self._analyze_chapter_content, chapter_id, chapter)
                )

            # Wait for all analyses to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"⚠️ Chapter analysis failed: {e}")

        # Analyze concept relationships
        self._analyze_concept_relationships()

        print("✅ Advanced analysis complete")

    def _analyze_chapter_content(self, chapter_id: str, chapter: Chapter):
        """Analyze individual chapter content"""
        # Extract examples
        examples = self._extract_examples(chapter.content)
        if examples:
            # Add examples to related concepts
            pass

        # Calculate concept importance
        self._calculate_concept_importance(chapter)

        # Generate quiz questions
        self._generate_chapter_questions(chapter)

    def _analyze_concept_relationships(self):
        """Analyze relationships between concepts"""
        print("  Analyzing concept relationships...")

        # Build similarity matrix using embeddings
        if self.embedding_model:
            concept_names = list(self.concepts.keys())
            if concept_names:
                try:
                    # Generate embeddings for all concepts
                    embeddings = self.embedding_model.encode(
                        [f"What is {name}?" for name in concept_names]
                    )

                    # Calculate similarity and create edges in knowledge graph
                    for i, concept1 in enumerate(concept_names):
                        self.knowledge_graph.add_node(concept1, type='concept')

                        for j, concept2 in enumerate(concept_names):
                            if i != j:
                                similarity = np.dot(embeddings[i], embeddings[j]) / (
                                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                                )

                                if similarity > 0.7:  # High similarity threshold
                                    self.knowledge_graph.add_edge(
                                        concept1, concept2,
                                        weight=similarity,
                                        relationship='related'
                                    )

                                    # Update concept relationships
                                    if concept1 in self.concepts:
                                        if concept2 not in self.concepts[concept1].related_concepts:
                                            self.concepts[concept1].related_concepts.append(concept2)

                except Exception as e:
                    print(f"  Embedding analysis failed: {e}")

    def _build_knowledge_graph(self):
        """Build comprehensive knowledge graph"""
        print("  Building knowledge graph...")

        # Add chapters as nodes
        for chapter_id, chapter in self.chapters.items():
            self.knowledge_graph.add_node(
                chapter_id,
                type='chapter',
                title=chapter.title,
                difficulty=chapter.difficulty
            )

            # Connect chapters to their concepts
            for concept in chapter.key_concepts:
                if concept in self.concepts:
                    self.knowledge_graph.add_edge(
                        chapter_id, concept,
                        relationship='contains',
                        strength=1.0
                    )

        # Add hierarchy relationships
        self._add_hierarchical_relationships()

        print(f"  Knowledge graph built: {self.knowledge_graph.number_of_nodes()} nodes, "
              f"{self.knowledge_graph.number_of_edges()} edges")

    def _add_hierarchical_relationships(self):
        """Add hierarchical relationships to knowledge graph"""
        # Identify parent-child relationships between concepts
        for concept_name, concept in self.concepts.items():
            # Check if concept is a specialization of another
            words = concept_name.lower().split()
            for other_concept in self.concepts:
                if concept_name != other_concept:
                    other_words = other_concept.lower().split()

                    # Check for hierarchical relationships
                    if any(word in other_words for word in words):
                        self.knowledge_graph.add_edge(
                            other_concept, concept_name,
                            relationship='generalizes',
                            strength=0.8
                        )

    def _extract_examples(self, text: str) -> List[str]:
        """Extract examples from text"""
        examples = []

        # Pattern for finding examples
        example_patterns = [
            r'for example[^.!?]*[.!?]',
            r'such as[^.!?]*[.!?]',
            r'for instance[^.!?]*[.!?]',
            r'e\.g\.[^.!?]*[.!?]',
            r'example[^.!?]*[.!?]'
        ]

        for pattern in example_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                example = match.group(0).strip()
                if len(example) > 20 and len(example) < 500:
                    examples.append(example)

        return examples[:5]  # Return top 5 examples

    def _calculate_concept_importance(self, chapter: Chapter):
        """Calculate importance score for concepts in chapter"""
        # Simple heuristic: concepts mentioned multiple times are more important
        for concept in chapter.key_concepts:
            if concept in self.concepts:
                # Count occurrences in chapter
                count = chapter.content.lower().count(concept.lower())

                # Update importance score
                self.concepts[concept].importance_score = min(1.0, count / 10)

    def _generate_chapter_questions(self, chapter: Chapter):
        """Generate assessment questions for chapter"""
        # This is a placeholder - actual implementation would use AI
        # to generate questions based on content
        pass

    def _extract_diagrams(self, pdf_path: Path):
        """Extract diagrams from PDF"""
        if hasattr(self, 'diagram_extractor'):
            try:
                diagrams = self.diagram_extractor.extract(pdf_path)
                print(f"  Extracted {len(diagrams)} diagrams")

                # Store diagram information
                self.diagrams = diagrams

            except Exception as e:
                print(f"  Diagram extraction failed: {e}")

    def _create_fallback_chapter(self, text: str):
        """Create fallback chapter when extraction fails"""
        print("⚠️ Using fallback chapter extraction")

        # Split text into manageable chunks
        chunks = self._split_text_into_chunks(text, chunk_size=5000)

        for i, chunk in enumerate(chunks):
            chapter_id = f"section_{i + 1}"

            # Extract key concepts from chunk
            key_concepts = self._extract_key_concepts_from_text(chunk)

            chapter = Chapter(
                number=str(i + 1),
                title=f"Section {i + 1}",
                content=chunk,
                start_page=0,
                end_page=0,
                key_concepts=key_concepts,
                learning_objectives=self._generate_learning_objectives(chunk, key_concepts),
                summary=self._summarize_text(chunk),
                difficulty=self._assess_difficulty(chunk)
            )

            self.chapters[chapter_id] = chapter

            # Extract concepts
            self._extract_concepts_from_chapter(chapter)

    def _split_text_into_chunks(self, text: str, chunk_size: int = 5000) -> List[str]:
        """Split text into chunks of approximately equal size"""
        chunks = []
        words = text.split()

        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space

            if current_length >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0

        # Add last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _save_processed_data(self):
        """Save all processed data to files"""
        print("💾 Saving processed data...")

        # Create output directory
        output_dir = Path(self.config.TEXTBOOKS_DIR) / self.textbook_id
        output_dir.mkdir(exist_ok=True)

        # Save chapters
        chapters_data = {}
        for chapter_id, chapter in self.chapters.items():
            chapters_data[chapter_id] = {
                'number': chapter.number,
                'title': chapter.title,
                'content': chapter.content[:5000],  # Save excerpt
                'key_concepts': chapter.key_concepts,
                'learning_objectives': chapter.learning_objectives,
                'summary': chapter.summary,
                'difficulty': chapter.difficulty
            }

        # Save concepts
        concepts_data = {}
        for concept_name, concept in self.concepts.items():
            concepts_data[concept_name] = {
                'definition': concept.definition,
                'examples': concept.examples,
                'related_concepts': concept.related_concepts,
                'importance_score': concept.importance_score,
                'difficulty_level': concept.difficulty_level,
                'textbook_references': concept.textbook_references
            }

        # Save knowledge graph
        graph_data = {
            'nodes': list(self.knowledge_graph.nodes(data=True)),
            'edges': list(self.knowledge_graph.edges(data=True))
        }

        # Combine all data
        processed_data = {
            'metadata': self.textbook_metadata,
            'chapters': chapters_data,
            'concepts': concepts_data,
            'knowledge_graph': graph_data,
            'diagrams': getattr(self, 'diagrams', {}),
            'processing_timestamp': datetime.now().isoformat()
        }

        # Save to JSON
        output_file = output_dir / "processed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"✅ Data saved to {output_file}")

    def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search textbook content with AI-powered ranking"""
        results = {
            'chapters': [],
            'concepts': [],
            'exact_matches': [],
            'related_content': []
        }

        query_lower = query.lower()

        # 1. Search in chapters
        for chapter_id, chapter in self.chapters.items():
            # Simple keyword matching
            if (query_lower in chapter.title.lower() or
                    query_lower in chapter.content.lower()):
                # Calculate relevance score
                title_score = 2.0 if query_lower in chapter.title.lower() else 0.0
                content_score = chapter.content.lower().count(query_lower) * 0.1

                results['chapters'].append({
                    'id': chapter_id,
                    'title': chapter.title,
                    'number': chapter.number,
                    'relevance_score': title_score + content_score,
                    'excerpt': self._extract_relevant_excerpt(chapter.content, query),
                    'summary': chapter.summary
                })

        # 2. Search in concepts
        for concept_name, concept in self.concepts.items():
            if query_lower in concept_name.lower() or query_lower in concept.definition.lower():
                results['concepts'].append({
                    'name': concept_name,
                    'definition': concept.definition,
                    'importance': concept.importance_score,
                    'difficulty': concept.difficulty_level
                })

        # 3. Use embeddings for semantic search
        if self.embedding_model and len(query) > 3:
            try:
                query_embedding = self.embedding_model.encode([query])[0]

                # Search in chapter summaries
                for chapter_id, chapter in self.chapters.items():
                    chapter_embedding = self.embedding_model.encode([chapter.summary])[0]
                    similarity = np.dot(query_embedding, chapter_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(chapter_embedding)
                    )

                    if similarity > 0.3:  # Semantic similarity threshold
                        results['related_content'].append({
                            'type': 'semantic_match',
                            'chapter_id': chapter_id,
                            'title': chapter.title,
                            'similarity': float(similarity),
                            'summary': chapter.summary
                        })

            except Exception as e:
                print(f"⚠️ Semantic search failed: {e}")

        # Sort results by relevance
        results['chapters'].sort(key=lambda x: x['relevance_score'], reverse=True)
        results['concepts'].sort(key=lambda x: x['importance'], reverse=True)
        results['related_content'].sort(key=lambda x: x['similarity'], reverse=True)

        # Limit results
        results['chapters'] = results['chapters'][:max_results]
        results['concepts'] = results['concepts'][:max_results]
        results['related_content'] = results['related_content'][:max_results]

        return results

    def _extract_relevant_excerpt(self, text: str, query: str, context_chars: int = 300) -> str:
        """Extract relevant excerpt around query"""
        text_lower = text.lower()
        query_lower = query.lower()

        pos = text_lower.find(query_lower)
        if pos == -1:
            # If query not found, return beginning of text
            return text[:context_chars] + "..."

        start = max(0, pos - context_chars // 2)
        end = min(len(text), pos + len(query) + context_chars // 2)

        excerpt = text[start:end]

        # Add ellipsis if not at beginning/end
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(text):
            excerpt = excerpt + "..."

        return excerpt

    def get_chapter(self, chapter_id: str) -> Optional[Chapter]:
        """Get chapter by ID"""
        return self.chapters.get(chapter_id)

    def get_concept(self, concept_name: str) -> Optional[Concept]:
        """Get concept by name"""
        return self.concepts.get(concept_name)

    def get_related_concepts(self, concept_name: str, max_related: int = 5) -> List[str]:
        """Get concepts related to given concept"""
        if concept_name not in self.knowledge_graph:
            return []

        # Get neighbors in knowledge graph
        neighbors = list(self.knowledge_graph.neighbors(concept_name))

        # Sort by edge weight (relationship strength)
        edges_with_weights = []
        for neighbor in neighbors:
            if self.knowledge_graph.has_edge(concept_name, neighbor):
                weight = self.knowledge_graph[concept_name][neighbor].get('weight', 0)
                edges_with_weights.append((neighbor, weight))

        edges_with_weights.sort(key=lambda x: x[1], reverse=True)

        return [concept for concept, _ in edges_with_weights[:max_related]]

    def generate_prerequisite_path(self, target_concept: str) -> List[str]:
        """Generate learning path with prerequisites"""
        if target_concept not in self.knowledge_graph:
            return [target_concept]

        # Use BFS to find prerequisite chain
        visited = set()
        queue = [(target_concept, [])]
        learning_path = []

        while queue and len(learning_path) < 10:  # Limit path length
            current, path = queue.pop(0)

            if current in visited:
                continue

            visited.add(current)
            learning_path.append(current)

            # Find prerequisites (concepts that generalize this one)
            for neighbor in self.knowledge_graph.neighbors(current):
                edge_data = self.knowledge_graph[current][neighbor]
                if edge_data.get('relationship') == 'generalizes':
                    queue.append((neighbor, path + [current]))

        # Return in learning order (prerequisites first)
        return learning_path[::-1]


# Create alias for backward compatibility
TextbookProcessor = TextbookProcessor  # Already defined, this is just for clarity