from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(200), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    preferences = Column(JSON)
    progress_data = Column(JSON)


class Textbook(Base):
    __tablename__ = 'textbooks'

    id = Column(String(50), primary_key=True)
    name = Column(String(200))
    file_path = Column(String(500))
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    metadata = Column(JSON)
    chapters = Column(JSON)
    concepts = Column(JSON)


class LearningSession(Base):
    __tablename__ = 'learning_sessions'

    id = Column(String(50), primary_key=True)
    user_id = Column(String(50))
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    duration_minutes = Column(Float)
    topics_covered = Column(JSON)
    assessment_results = Column(JSON)


class AssessmentResult(Base):
    __tablename__ = 'assessment_results'

    id = Column(String(50), primary_key=True)
    user_id = Column(String(50))
    assessment_id = Column(String(50))
    score = Column(Float)
    total_score = Column(Float)
    percentage = Column(Float)
    completed_at = Column(DateTime, default=datetime.utcnow)
    details = Column(JSON)


class GeneratedContent(Base):
    __tablename__ = 'generated_content'

    id = Column(String(50), primary_key=True)
    content_type = Column(String(50))  # lesson, diagram, quiz
    topic = Column(String(200))
    generated_at = Column(DateTime, default=datetime.utcnow)
    content_data = Column(JSON)
    metadata = Column(JSON)