import os
from dataclasses import dataclass
from typing import Dict, Any
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    # HuggingFace model names
    QA_MODEL: str = "distilbert-base-cased-distilled-squad"
    SUMMARIZATION_MODEL: str = "t5-small"
    TEXT_GENERATION_MODEL: str = "gpt2"
    NER_MODEL: str = "dslim/bert-base-NER"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Model parameters
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    MODEL_CACHE_DIR: str = "data/models"

    # Advanced model settings
    USE_FP16: bool = False  # Use half precision for faster inference
    DEVICE: str = "auto"  # auto, cpu, cuda, mps


@dataclass
class ProcessingConfig:
    """Configuration for text processing"""
    MAX_TEXT_LENGTH: int = 1000000
    CHUNK_SIZE: int = 1000
    OVERLAP: int = 100
    MIN_SENTENCE_LENGTH: int = 20

    # OCR settings
    USE_OCR: bool = True
    OCR_LANGUAGE: str = "eng"

    # Diagram extraction
    EXTRACT_DIAGRAMS: bool = True
    DIAGRAM_MIN_SIZE: int = 100  # pixels


@dataclass
class LearningConfig:
    """Configuration for adaptive learning"""
    INITIAL_DIFFICULTY: str = "intermediate"
    MASTERY_THRESHOLD: float = 0.7
    REVIEW_INTERVAL_DAYS: int = 7

    # Personalization
    USE_CLUSTERING: bool = True
    NUM_LEARNING_STYLES: int = 3

    # Assessment
    QUIZ_QUESTION_COUNT: int = 5
    ASSESSMENT_INTERVAL: int = 3  # lessons


@dataclass
class CloudSimulationConfig:
    """Configuration for cloud simulations"""
    SIMULATE_AWS: bool = True
    SIMULATE_AZURE: bool = True
    SIMULATE_GCP: bool = True

    # Resource limits
    MAX_INSTANCES: int = 10
    MAX_STORAGE_GB: int = 100
    MAX_LAMBDA_FUNCTIONS: int = 20

    # Simulation realism
    USE_REALISTIC_DELAYS: bool = True
    INCLUDE_ERROR_SIMULATION: bool = True


class AppConfig:
    """Main application configuration"""

    def __init__(self):
        self.base_dir = Path(__file__).parent

        # Load environment variables
        self.load_env()

        # Initialize configurations
        self.models = ModelConfig()
        self.processing = ProcessingConfig()
        self.learning = LearningConfig()
        self.cloud = CloudSimulationConfig()

        # Application settings
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        self.PORT = int(os.getenv("PORT", 8501))
        self.HOST = os.getenv("HOST", "0.0.0.0")

        # Paths
        self.DATA_DIR = self.base_dir / "data"
        self.MODELS_DIR = self.DATA_DIR / "models"
        self.TEXTBOOKS_DIR = self.DATA_DIR / "textbooks"
        self.EXPORTS_DIR = self.DATA_DIR / "exports"
        self.STATIC_DIR = self.base_dir / "static"

        # Create directories
        self.setup_directories()

        # Database
        self.DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{self.DATA_DIR}/lessonstudio.db")

        # Security
        self.SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
        self.ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "your-encryption-key-32-bytes")

        # Cache
        self.CACHE_TYPE = os.getenv("CACHE_TYPE", "simple")
        self.CACHE_TIMEOUT = int(os.getenv("CACHE_TIMEOUT", 300))

        # Override from environment variables if present
        self._override_from_env()

    def load_env(self):
        """Load environment variables from .env file"""
        env_path = self.base_dir / ".env"
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)

    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.TEXTBOOKS_DIR,
            self.EXPORTS_DIR,
            self.STATIC_DIR / "css",
            self.STATIC_DIR / "js",
            self.STATIC_DIR / "templates",
            self.DATA_DIR / "progress",
            self.DATA_DIR / "cache"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _override_from_env(self):
        """Override config from environment variables"""
        # Model configurations from env
        if os.getenv("QA_MODEL"):
            self.models.QA_MODEL = os.getenv("QA_MODEL")
        if os.getenv("SUMMARIZATION_MODEL"):
            self.models.SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL")
        if os.getenv("TEXT_GENERATION_MODEL"):
            self.models.TEXT_GENERATION_MODEL = os.getenv("TEXT_GENERATION_MODEL")
        if os.getenv("NER_MODEL"):
            self.models.NER_MODEL = os.getenv("NER_MODEL")
        if os.getenv("EMBEDDING_MODEL"):
            self.models.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

        # Cloud simulation settings
        if os.getenv("SIMULATE_AWS"):
            self.cloud.SIMULATE_AWS = os.getenv("SIMULATE_AWS").lower() == "true"
        if os.getenv("SIMULATE_AZURE"):
            self.cloud.SIMULATE_AZURE = os.getenv("SIMULATE_AZURE").lower() == "true"
        if os.getenv("SIMULATE_GCP"):
            self.cloud.SIMULATE_GCP = os.getenv("SIMULATE_GCP").lower() == "true"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "models": self.models.__dict__,
            "processing": self.processing.__dict__,
            "learning": self.learning.__dict__,
            "cloud": self.cloud.__dict__,
            "app": {
                "debug": self.DEBUG,
                "port": self.PORT,
                "host": self.HOST,
                "database_url": self.DATABASE_URL,
                "models_dir": str(self.MODELS_DIR)
            }
        }

    def save(self, path: Path = None):
        """Save configuration to file"""
        if path is None:
            path = self.DATA_DIR / "config.json"

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path):
        """Load configuration from file"""
        with open(path, 'r') as f:
            data = json.load(f)

        config = cls()
        # Update configuration from file
        if "models" in data:
            for key, value in data["models"].items():
                if hasattr(config.models, key):
                    setattr(config.models, key, value)

        if "processing" in data:
            for key, value in data["processing"].items():
                if hasattr(config.processing, key):
                    setattr(config.processing, key, value)

        if "learning" in data:
            for key, value in data["learning"].items():
                if hasattr(config.learning, key):
                    setattr(config.learning, key, value)

        if "cloud" in data:
            for key, value in data["cloud"].items():
                if hasattr(config.cloud, key):
                    setattr(config.cloud, key, value)

        return config


# Global configuration instance
config = AppConfig()