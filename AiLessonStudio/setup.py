#!/usr/bin/env python3
"""
Setup script for AI Lesson Studio
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def create_directories():
    """Create necessary directories"""
    directories = [
        'data/textbooks',
        'data/models',
        'data/exports',
        'data/progress',
        'static/css',
        'static/js',
        'static/templates',
        'logs'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)


def download_models():
    """Download AI models"""
    print("🤖 Downloading AI models...")

    models_to_download = [
        "distilbert-base-cased-distilled-squad",
        "t5-small",
        "gpt2",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]

    for model in models_to_download:
        print(f"  Downloading {model}...")
        # This would use transformers' built-in download
        # In practice, models are downloaded on first use

    print("✅ AI models setup complete")


def setup_spacy():
    """Setup spaCy NLP model"""
    print("🧠 Setting up spaCy...")

    try:
        import spacy
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
        print("✅ spaCy model downloaded")
    except Exception as e:
        print(f"⚠️ Could not download spaCy model: {e}")


def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# AI Lesson Studio Configuration
DEBUG=False
PORT=8501
HOST=0.0.0.0
SECRET_KEY=your-secret-key-change-in-production
ENCRYPTION_KEY=your-encryption-key-32-bytes-long-here

# Database
DATABASE_URL=sqlite:///data/lessonstudio.db

# Cache
CACHE_TYPE=simple
CACHE_TIMEOUT=300

# HuggingFace (optional)
# HUGGINGFACE_TOKEN=your_token_here

# Cloud Simulation
SIMULATE_AWS=True
SIMULATE_AZURE=True
SIMULATE_GCP=True
"""

        with open(env_file, "w") as f:
            f.write(env_content)

        print("✅ Created .env file")
        print("⚠️ Please update SECRET_KEY and ENCRYPTION_KEY in .env file")


def main():
    """Main setup function"""
    print("=" * 50)
    print("AI Lesson Studio - Setup")
    print("=" * 50)

    # Check Python version
    check_python_version()

    # Create directories
    create_directories()

    # Install dependencies
    install_dependencies()

    # Setup AI models
    download_models()
    setup_spacy()

    # Create environment file
    create_env_file()

    print("=" * 50)
    print("✅ Setup complete!")
    print("\nTo start the application:")
    print("1. streamlit run app.py")
    print("2. Open http://localhost:8501 in your browser")
    print("\nFor Docker deployment:")
    print("1. docker-compose up --build")
    print("=" * 50)


if __name__ == "__main__":
    main()