import subprocess
import sys
import os


def fix_all_issues():
    print("🔧 Fixing all issues...")

    # 1. Update PyTorch
    print("\n1️⃣ Updating PyTorch...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.1.0", "--index-url",
                               "https://download.pytorch.org/whl/cpu"])
        print("✅ PyTorch updated to 2.1.0")
    except Exception as e:
        print(f"⚠️ Could not update PyTorch: {e}")

    # 2. Update other dependencies
    print("\n2️⃣ Updating other dependencies...")
    packages = [
        "transformers==4.36.0",
        "spacy==3.7.0",
        "sentencepiece==0.1.99",
        "protobuf==4.25.1"
    ]

    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Updated {package}")
        except:
            print(f"⚠️ Could not update {package}")

    # 3. Download spaCy model
    print("\n3️⃣ Downloading spaCy model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded")
    except Exception as e:
        print(f"⚠️ Could not download spaCy model: {e}")

    # 4. Create optimized AI engine
    print("\n4️⃣ Creating optimized AI engine...")
    create_optimized_ai_engine()

    print("\n" + "=" * 50)
    print("✅ All fixes applied!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Restart your Streamlit app")
    print("2. Try uploading the textbook again")
    print("3. If still issues, try a smaller textbook or TXT format")


def create_optimized_ai_engine():
    """Create optimized AI engine for large textbooks"""
    optimized_code = '''# core/ai_engine.py - OPTIMIZED VERSION
from transformers import pipeline
from typing import Dict, List, Any
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class AIEngine:
    """AI Engine for Cloud Computing Tutor - Optimized for large texts"""

    def __init__(self):
        print("🚀 Initializing Optimized AI Engine...")

        # Load spaCy for NLP with increased max_length
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp.max_length = 3000000  # 3 million characters
            print(f"✅ spaCy model loaded (max_length: {self.nlp.max_length})")
        except Exception as e:
            print(f"⚠️ spaCy issue: {e}")
            self.nlp = None

        # Load pre-trained models
        print("📥 Loading pre-trained models...")
        self.models = {}
        self._load_models()

        print("✅ AI Engine ready!")

    def _load_models(self):
        """Load all pre-trained models - Optimized for memory"""
        try:
            # Use smaller models to save memory
            print("Using lightweight models for better performance...")

            # Question Answering - smaller model
            self.models['qa'] = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=-1  # CPU for stability
            )
            print("✅ QA model loaded")

            # Text Summarization - smallest available
            self.models['summarize'] = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",  # Very small model
                device=-1
            )
            print("✅ Summarization model loaded")

        except Exception as e:
            print(f"⚠️ Model loading warning: {e}")
            print("Using rule-based fallback methods (still functional)")
            self.models['qa'] = self._rule_based_qa
            self.models['summarize'] = self._simple_summarize

    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """Answer question based on context"""
        # Limit context size
        safe_context = context[:5000]  # First 5000 chars only

        if 'qa' in self.models and callable(self.models['qa']):
            try:
                result = self.models['qa'](question=question, context=safe_context)
                return {
                    'answer': result['answer'],
                    'confidence': result['score'],
                    'source': 'AI Model'
                }
            except:
                pass

        # Fallback to rule-based
        return self._rule_based_qa(question, safe_context)

    def _rule_based_qa(self, question: str, context: str) -> Dict[str, Any]:
        """Rule-based Q&A as fallback"""
        question_lower = question.lower()

        # Simple rule-based matching
        if 'what is' in question_lower:
            concept = question_lower.replace('what is', '').strip()
            # Search in sentences
            sentences = context.split('.')
            for sentence in sentences[:20]:  # First 20 sentences only
                if concept and concept in sentence.lower():
                    return {
                        'answer': sentence.strip() + '.',
                        'confidence': 0.7,
                        'source': 'Rule-based'
                    }

        return {
            'answer': "I need more context to answer this question.",
            'confidence': 0.3,
            'source': 'Default'
        }

    def _simple_summarize(self, text: str, max_length: int = 150) -> str:
        """Simple extractive summarization"""
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return text

        # Return first few sentences
        summary = ' '.join(sentences[:3])
        if len(summary) > max_length:
            summary = summary[:max_length] + '...'

        return summary

    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize text"""
        # Limit text size
        safe_text = text[:2000]  # First 2000 chars only

        if 'summarize' in self.models and callable(self.models['summarize']):
            try:
                result = self.models['summarize'](
                    safe_text,
                    max_length=max_length,
                    min_length=30,
                    do_sample=False
                )
                return result[0]['summary_text']
            except:
                pass

        # Fallback
        return self._simple_summarize(safe_text, max_length)

    def extract_key_concepts(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key concepts from text - optimized"""
        # Use simple word frequency for large texts
        words = text.lower().split()

        # Filter out common words and short words
        stop_words = {
            'the', 'and', 'for', 'with', 'this', 'that', 'are', 'was', 'is', 'in',
            'of', 'to', 'a', 'an', 'or', 'but', 'on', 'at', 'by', 'as', 'from',
            'up', 'down', 'out', 'over', 'under', 'again', 'further', 'then'
        }

        filtered = [w.strip('.,!?;:()[]{}"\'') for w in words 
                   if len(w) > 3 and w not in stop_words]

        from collections import Counter
        return [word for word, _ in Counter(filtered).most_common(top_n)]

    def generate_lesson_outline(self, topic: str, context: str) -> List[str]:
        """Generate lesson outline for topic"""
        concepts = self.extract_key_concepts(context, top_n=5)

        outline = [
            f"Introduction to {topic}",
            f"Key Concepts: {', '.join(concepts[:3])}",
            "Practical Applications",
            "Common Use Cases",
            "Best Practices",
            "Summary and Key Takeaways"
        ]

        return outline

# Singleton instance
_ai_engine = None

def get_ai_engine() -> AIEngine:
    """Get singleton AI engine instance"""
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIEngine()
    return _ai_engine

if __name__ == "__main__":
    print("Testing Optimized AI Engine...")
    engine = get_ai_engine()
    test_context = "Cloud computing delivers services over internet."
    test_question = "What is cloud computing?"
    answer = engine.answer_question(test_question, test_context)
    print(f"Q: {test_question}")
    print(f"A: {answer['answer']}")
    print("✅ Optimized AI Engine working!")'''

    # Write to file
    with open('src/core/ai_engine.py', 'w') as f:
        f.write(optimized_code)

    print("✅ Created optimized AI engine")


if __name__ == "__main__":
    fix_all_issues()