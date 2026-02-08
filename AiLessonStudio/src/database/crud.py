from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
from sqlalchemy import func

from .models import User, Textbook, LearningSession, AssessmentResult, GeneratedContent


class CRUDOperations:
    """Database CRUD operations"""

    def __init__(self, session: Session):
        self.session = session

    # User operations
    def create_user(self, username: str, email: str = None,
                    preferences: Dict = None) -> Optional[User]:
        """Create a new user"""
        try:
            user = User(
                username=username,
                email=email,
                preferences=preferences or {},
                created_at=datetime.utcnow()
            )
            self.session.add(user)
            self.session.commit()
            return user
        except SQLAlchemyError as e:
            self.session.rollback()
            print(f"Error creating user: {e}")
            return None

    def get_user(self, user_id: int = None, username: str = None) -> Optional[User]:
        """Get user by ID or username"""
        try:
            if user_id:
                return self.session.query(User).filter(User.id == user_id).first()
            elif username:
                return self.session.query(User).filter(User.username == username).first()
            return None
        except SQLAlchemyError as e:
            print(f"Error getting user: {e}")
            return None

    def update_user_progress(self, user_id: int, progress_data: Dict) -> bool:
        """Update user progress data"""
        try:
            user = self.get_user(user_id=user_id)
            if user:
                user.progress_data = progress_data
                self.session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            self.session.rollback()
            print(f"Error updating user progress: {e}")
            return False

    # Textbook operations
    def create_textbook(self, textbook_id: str, name: str, file_path: str,
                        metadata: Dict = None, chapters: Dict = None,
                        concepts: Dict = None) -> Optional[Textbook]:
        """Create textbook record"""
        try:
            textbook = Textbook(
                id=textbook_id,
                name=name,
                file_path=file_path,
                uploaded_at=datetime.utcnow(),
                processed_at=datetime.utcnow(),
                metadata=metadata or {},
                chapters=chapters or {},
                concepts=concepts or {}
            )
            self.session.add(textbook)
            self.session.commit()
            return textbook
        except SQLAlchemyError as e:
            self.session.rollback()
            print(f"Error creating textbook: {e}")
            return None

    def get_textbook(self, textbook_id: str) -> Optional[Textbook]:
        """Get textbook by ID"""
        try:
            return self.session.query(Textbook).filter(Textbook.id == textbook_id).first()
        except SQLAlchemyError as e:
            print(f"Error getting textbook: {e}")
            return None

    def get_all_textbooks(self) -> List[Textbook]:
        """Get all textbooks"""
        try:
            return self.session.query(Textbook).all()
        except SQLAlchemyError as e:
            print(f"Error getting textbooks: {e}")
            return []

    # Learning session operations
    def create_learning_session(self, user_id: str, topics: List[str] = None,
                                duration_minutes: float = 0) -> Optional[LearningSession]:
        """Create learning session record"""
        try:
            session = LearningSession(
                id=f"session_{datetime.utcnow().timestamp()}",
                user_id=user_id,
                started_at=datetime.utcnow(),
                duration_minutes=duration_minutes,
                topics_covered=topics or [],
                assessment_results={}
            )
            self.session.add(session)
            self.session.commit()
            return session
        except SQLAlchemyError as e:
            self.session.rollback()
            print(f"Error creating learning session: {e}")
            return None

    def end_learning_session(self, session_id: str,
                             assessment_results: Dict = None) -> bool:
        """End learning session"""
        try:
            session = self.session.query(LearningSession).filter(
                LearningSession.id == session_id
            ).first()

            if session:
                session.ended_at = datetime.utcnow()
                if assessment_results:
                    session.assessment_results = assessment_results
                self.session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            self.session.rollback()
            print(f"Error ending learning session: {e}")
            return False

    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[LearningSession]:
        """Get user's learning sessions"""
        try:
            return self.session.query(LearningSession).filter(
                LearningSession.user_id == user_id
            ).order_by(LearningSession.started_at.desc()).limit(limit).all()
        except SQLAlchemyError as e:
            print(f"Error getting user sessions: {e}")
            return []

    # Assessment operations
    def create_assessment_result(self, user_id: str, assessment_id: str,
                                 score: float, total_score: float,
                                 percentage: float, details: Dict) -> Optional[AssessmentResult]:
        """Create assessment result"""
        try:
            result = AssessmentResult(
                id=f"assessment_{datetime.utcnow().timestamp()}",
                user_id=user_id,
                assessment_id=assessment_id,
                score=score,
                total_score=total_score,
                percentage=percentage,
                completed_at=datetime.utcnow(),
                details=details
            )
            self.session.add(result)
            self.session.commit()
            return result
        except SQLAlchemyError as e:
            self.session.rollback()
            print(f"Error creating assessment result: {e}")
            return None

    def get_user_assessments(self, user_id: str, limit: int = 20) -> List[AssessmentResult]:
        """Get user's assessment results"""
        try:
            return self.session.query(AssessmentResult).filter(
                AssessmentResult.user_id == user_id
            ).order_by(AssessmentResult.completed_at.desc()).limit(limit).all()
        except SQLAlchemyError as e:
            print(f"Error getting user assessments: {e}")
            return []

    # Generated content operations
    def save_generated_content(self, content_type: str, topic: str,
                               content_data: Dict, metadata: Dict = None) -> Optional[GeneratedContent]:
        """Save generated content"""
        try:
            content = GeneratedContent(
                id=f"{content_type}_{datetime.utcnow().timestamp()}",
                content_type=content_type,
                topic=topic,
                generated_at=datetime.utcnow(),
                content_data=content_data,
                metadata=metadata or {}
            )
            self.session.add(content)
            self.session.commit()
            return content
        except SQLAlchemyError as e:
            self.session.rollback()
            print(f"Error saving generated content: {e}")
            return None

    def get_generated_content(self, content_type: str = None,
                              topic: str = None, limit: int = 10) -> List[GeneratedContent]:
        """Get generated content with filters"""
        try:
            query = self.session.query(GeneratedContent)

            if content_type:
                query = query.filter(GeneratedContent.content_type == content_type)

            if topic:
                query = query.filter(GeneratedContent.topic.ilike(f"%{topic}%"))

            return query.order_by(GeneratedContent.generated_at.desc()).limit(limit).all()
        except SQLAlchemyError as e:
            print(f"Error getting generated content: {e}")
            return []

    # Analytics operations
    def get_user_learning_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user learning statistics"""
        try:
            # Total sessions
            total_sessions = self.session.query(LearningSession).filter(
                LearningSession.user_id == user_id
            ).count()

            # Total learning time
            total_time = self.session.query(
                func.sum(LearningSession.duration_minutes)
            ).filter(LearningSession.user_id == user_id).scalar() or 0

            # Average assessment score
            avg_score = self.session.query(
                func.avg(AssessmentResult.percentage)
            ).filter(AssessmentResult.user_id == user_id).scalar() or 0

            # Recent activity
            recent_sessions = self.get_user_sessions(user_id, limit=5)

            return {
                'total_sessions': total_sessions,
                'total_learning_hours': round(total_time / 60, 1),
                'average_score': round(avg_score, 1),
                'recent_sessions': [
                    {
                        'id': session.id,
                        'started_at': session.started_at.isoformat() if session.started_at else None,
                        'duration': session.duration_minutes,
                        'topics': session.topics_covered
                    }
                    for session in recent_sessions
                ]
            }
        except SQLAlchemyError as e:
            print(f"Error getting user stats: {e}")
            return {}

    # Utility operations
    def backup_data(self, backup_path: str) -> bool:
        """Backup database data to JSON"""
        try:
            data = {
                'users': [],
                'textbooks': [],
                'sessions': [],
                'assessments': [],
                'content': [],
                'backup_time': datetime.utcnow().isoformat()
            }

            # Backup users
            users = self.session.query(User).all()
            for user in users:
                data['users'].append({
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'created_at': user.created_at.isoformat() if user.created_at else None,
                    'preferences': user.preferences
                })

            # Backup textbooks
            textbooks = self.session.query(Textbook).all()
            for textbook in textbooks:
                data['textbooks'].append({
                    'id': textbook.id,
                    'name': textbook.name,
                    'uploaded_at': textbook.uploaded_at.isoformat() if textbook.uploaded_at else None,
                    'metadata': textbook.metadata
                })

            # Save to file
            with open(backup_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            return True
        except Exception as e:
            print(f"Error backing up data: {e}")
            return False

    def cleanup_old_data(self, days_old: int = 30) -> int:
        """Clean up old data"""
        try:
            cutoff_date = datetime.utcnow().timedelta(days=-days_old)

            # Delete old sessions
            old_sessions = self.session.query(LearningSession).filter(
                LearningSession.started_at < cutoff_date
            ).delete()

            # Delete old assessment results
            old_assessments = self.session.query(AssessmentResult).filter(
                AssessmentResult.completed_at < cutoff_date
            ).delete()

            self.session.commit()

            return old_sessions + old_assessments
        except SQLAlchemyError as e:
            self.session.rollback()
            print(f"Error cleaning up data: {e}")
            return 0