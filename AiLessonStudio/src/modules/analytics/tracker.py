import json
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


@dataclass
class LearningEvent:
    """Learning event data structure"""
    event_id: str
    user_id: str
    event_type: str  # 'lesson_view', 'quiz_attempt', 'code_execution', 'diagram_view'
    concept_id: str
    timestamp: datetime
    duration_seconds: float
    success_rate: float
    difficulty: str
    metadata: Dict[str, Any]


@dataclass
class ConceptMastery:
    """Concept mastery tracking"""
    concept_id: str
    user_id: str
    mastery_score: float  # 0.0 to 1.0
    attempts: int
    correct_attempts: int
    last_practiced: datetime
    learning_curve: List[float]
    predicted_next_score: float


class ProgressTracker:
    """Advanced progress tracking with ML analytics"""

    def __init__(self, config):
        self.config = config
        self.user_progress: Dict[str, Dict[str, ConceptMastery]] = {}
        self.learning_events: List[LearningEvent] = []
        self.learning_patterns: Dict[str, Any] = {}

        # Load existing data
        self._load_data()

        print("✅ Progress Tracker initialized")

    def _load_data(self):
        """Load existing progress data"""
        progress_file = self.config.DATA_DIR / "progress" / "mastery_data.pkl"
        if progress_file.exists():
            try:
                with open(progress_file, 'rb') as f:
                    data = pickle.load(f)
                    self.user_progress = data.get('user_progress', {})
                    self.learning_events = data.get('learning_events', [])
                print(f"📊 Loaded progress data for {len(self.user_progress)} users")
            except:
                print("⚠️ Could not load progress data, starting fresh")

    def _save_data(self):
        """Save progress data"""
        progress_file = self.config.DATA_DIR / "progress" / "mastery_data.pkl"
        progress_file.parent.mkdir(exist_ok=True)

        data = {
            'user_progress': self.user_progress,
            'learning_events': self.learning_events,
            'last_saved': datetime.now()
        }

        with open(progress_file, 'wb') as f:
            pickle.dump(data, f)

    def record_event(self, user_id: str, event_type: str,
                     concept_id: str, duration: float = 0.0,
                     success_rate: float = 0.0, difficulty: str = "medium",
                     metadata: Dict[str, Any] = None) -> str:
        """Record a learning event"""
        event_id = f"event_{datetime.now().timestamp()}_{len(self.learning_events)}"

        event = LearningEvent(
            event_id=event_id,
            user_id=user_id,
            event_type=event_type,
            concept_id=concept_id,
            timestamp=datetime.now(),
            duration_seconds=duration,
            success_rate=success_rate,
            difficulty=difficulty,
            metadata=metadata or {}
        )

        self.learning_events.append(event)

        # Update concept mastery
        self._update_concept_mastery(user_id, concept_id, success_rate)

        # Analyze learning patterns
        self._analyze_learning_patterns(user_id)

        # Save data periodically
        if len(self.learning_events) % 10 == 0:
            self._save_data()

        return event_id

    def _update_concept_mastery(self, user_id: str, concept_id: str,
                                success_rate: float):
        """Update mastery score for a concept"""
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}

        if concept_id not in self.user_progress[user_id]:
            # Initialize new concept mastery
            mastery = ConceptMastery(
                concept_id=concept_id,
                user_id=user_id,
                mastery_score=success_rate,
                attempts=1,
                correct_attempts=1 if success_rate >= 0.7 else 0,
                last_practiced=datetime.now(),
                learning_curve=[success_rate],
                predicted_next_score=success_rate
            )
            self.user_progress[user_id][concept_id] = mastery
        else:
            # Update existing mastery
            mastery = self.user_progress[user_id][concept_id]

            # Update attempts
            mastery.attempts += 1
            if success_rate >= 0.7:
                mastery.correct_attempts += 1

            # Update mastery score with weighted average
            # More recent attempts have higher weight
            weights = [0.3, 0.4, 0.5]  # Increasing weights for recent attempts
            recent_attempts = mastery.learning_curve[-3:] + [success_rate]
            weighted_scores = []

            for i, score in enumerate(recent_attempts[-3:]):
                weight = weights[i] if i < len(weights) else 0.5
                weighted_scores.append(score * weight)

            new_score = sum(weighted_scores) / sum(weights[:len(weighted_scores)])
            mastery.mastery_score = max(0.0, min(1.0, new_score))

            # Update learning curve
            mastery.learning_curve.append(success_rate)

            # Predict next score using simple linear regression
            if len(mastery.learning_curve) >= 3:
                x = np.arange(len(mastery.learning_curve[-5:]))
                y = np.array(mastery.learning_curve[-5:])
                if len(x) > 1:
                    z = np.polyfit(x, y, 1)
                    mastery.predicted_next_score = z[0] * (len(x)) + z[1]

            mastery.last_practiced = datetime.now()

    def _analyze_learning_patterns(self, user_id: str):
        """Analyze user learning patterns"""
        if user_id not in self.learning_patterns:
            self.learning_patterns[user_id] = {
                'preferred_time': [],
                'effective_duration': [],
                'best_difficulty': {},
                'concept_connections': {},
                'learning_style': None
            }

        # Get user's recent events
        user_events = [e for e in self.learning_events if e.user_id == user_id]

        if not user_events:
            return

        # Analyze time patterns
        for event in user_events[-20:]:
            hour = event.timestamp.hour
            self.learning_patterns[user_id]['preferred_time'].append(hour)

            # Duration effectiveness
            if event.duration_seconds > 0:
                effectiveness = event.success_rate / max(1, event.duration_seconds / 60)
                self.learning_patterns[user_id]['effective_duration'].append(
                    (event.duration_seconds, effectiveness)
                )

        # Find optimal learning time
        if self.learning_patterns[user_id]['preferred_time']:
            hours = self.learning_patterns[user_id]['preferred_time']
            hour_counts = pd.Series(hours).value_counts()
            if not hour_counts.empty:
                optimal_hour = hour_counts.idxmax()
                self.learning_patterns[user_id]['optimal_time'] = optimal_hour

    def get_concept_mastery(self, user_id: str, concept_id: str) -> Optional[ConceptMastery]:
        """Get mastery for specific concept"""
        if user_id in self.user_progress:
            return self.user_progress[user_id].get(concept_id)
        return None

    def get_overall_progress(self, user_id: str) -> Dict[str, Any]:
        """Get overall learning progress"""
        if user_id not in self.user_progress:
            return {
                'total_concepts': 0,
                'mastered_concepts': 0,
                'average_mastery': 0.0,
                'total_time': 0.0,
                'recent_activity': []
            }

        user_mastery = self.user_progress[user_id]

        # Calculate metrics
        total_concepts = len(user_mastery)
        mastered_concepts = sum(1 for m in user_mastery.values()
                                if m.mastery_score >= 0.7)
        average_mastery = np.mean([m.mastery_score for m in user_mastery.values()])

        # Calculate total learning time
        user_events = [e for e in self.learning_events if e.user_id == user_id]
        total_time = sum(e.duration_seconds for e in user_events)

        # Get recent activity
        recent_events = sorted(user_events, key=lambda x: x.timestamp, reverse=True)[:10]
        recent_activity = [
            {
                'concept': e.concept_id,
                'type': e.event_type,
                'time': e.timestamp.isoformat(),
                'success': e.success_rate,
                'duration': e.duration_seconds
            }
            for e in recent_events
        ]

        return {
            'total_concepts': total_concepts,
            'mastered_concepts': mastered_concepts,
            'average_mastery': float(average_mastery),
            'mastery_percentage': (mastered_concepts / max(1, total_concepts)) * 100,
            'total_time': total_time,
            'recent_activity': recent_activity,
            'learning_patterns': self.learning_patterns.get(user_id, {})
        }

    def get_learning_recommendations(self, user_id: str,
                                     available_concepts: List[str],
                                     max_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Get personalized learning recommendations"""
        recommendations = []

        # Get user's current mastery
        user_mastery = self.user_progress.get(user_id, {})

        # Categorize concepts
        mastered = []
        learning = []
        not_started = []

        for concept in available_concepts:
            mastery = user_mastery.get(concept)
            if mastery:
                if mastery.mastery_score >= 0.7:
                    mastered.append((concept, mastery))
                elif mastery.mastery_score >= 0.3:
                    learning.append((concept, mastery))
                else:
                    not_started.append((concept, mastery))
            else:
                not_started.append((concept, None))

        # Generate recommendations

        # 1. Recommend concepts that need reinforcement
        if learning:
            # Sort by mastery score (lowest first)
            learning.sort(key=lambda x: x[1].mastery_score if x[1] else 1.0)
            for concept, mastery in learning[:2]:
                days_since = (datetime.now() - mastery.last_practiced).days
                if days_since >= 2:  # Needs reinforcement
                    recommendations.append({
                        'concept': concept,
                        'type': 'reinforcement',
                        'priority': 'high',
                        'reason': f'Last practiced {days_since} days ago',
                        'mastery': mastery.mastery_score if mastery else 0.0
                    })

        # 2. Recommend new concepts related to mastered ones
        if mastered and not_started:
            # Simple: recommend 2 new concepts
            for concept, _ in not_started[:2]:
                recommendations.append({
                    'concept': concept,
                    'type': 'new_concept',
                    'priority': 'medium',
                    'reason': 'New concept to expand knowledge',
                    'mastery': 0.0
                })

        # 3. Recommend challenging concepts for advanced learners
        if len(mastered) >= 5 and not_started:
            # Recommend advanced concepts
            for concept, _ in not_started[-2:]:  # Last ones might be advanced
                recommendations.append({
                    'concept': concept,
                    'type': 'challenge',
                    'priority': 'low',
                    'reason': 'Advanced concept to challenge mastery',
                    'mastery': 0.0
                })

        # Limit recommendations
        return recommendations[:max_recommendations]

    def generate_progress_report(self, user_id: str,
                                 start_date: datetime = None,
                                 end_date: datetime = None) -> Dict[str, Any]:
        """Generate detailed progress report"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        # Filter events by date
        user_events = [
            e for e in self.learning_events
            if e.user_id == user_id and start_date <= e.timestamp <= end_date
        ]

        if not user_events:
            return {'message': 'No learning activity in the selected period'}

        # Calculate daily activity
        daily_activity = {}
        for event in user_events:
            date_str = event.timestamp.date().isoformat()
            if date_str not in daily_activity:
                daily_activity[date_str] = {
                    'events': 0,
                    'duration': 0.0,
                    'success_rate': 0.0,
                    'concepts': set()
                }

            daily = daily_activity[date_str]
            daily['events'] += 1
            daily['duration'] += event.duration_seconds
            daily['success_rate'] += event.success_rate
            daily['concepts'].add(event.concept_id)

        # Calculate averages
        for date_str in daily_activity:
            daily = daily_activity[date_str]
            if daily['events'] > 0:
                daily['success_rate'] /= daily['events']
            daily['concepts'] = list(daily['concepts'])

        # Calculate concept progress
        concept_progress = {}
        for event in user_events:
            if event.concept_id not in concept_progress:
                concept_progress[event.concept_id] = {
                    'attempts': 0,
                    'total_success': 0.0,
                    'last_attempt': None,
                    'first_attempt': None
                }

            concept = concept_progress[event.concept_id]
            concept['attempts'] += 1
            concept['total_success'] += event.success_rate
            if not concept['last_attempt'] or event.timestamp > concept['last_attempt']:
                concept['last_attempt'] = event.timestamp
            if not concept['first_attempt'] or event.timestamp < concept['first_attempt']:
                concept['first_attempt'] = event.timestamp

        # Calculate mastery for each concept
        for concept_id, data in concept_progress.items():
            data['average_success'] = data['total_success'] / data['attempts']
            if data['average_success'] >= 0.8:
                data['mastery'] = 'expert'
            elif data['average_success'] >= 0.6:
                data['mastery'] = 'proficient'
            elif data['average_success'] >= 0.4:
                data['mastery'] = 'familiar'
            else:
                data['mastery'] = 'beginner'

        # Overall statistics
        total_events = len(user_events)
        total_duration = sum(e.duration_seconds for e in user_events)
        avg_success = np.mean([e.success_rate for e in user_events])
        unique_concepts = len(concept_progress)

        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'overall': {
                'total_events': total_events,
                'total_duration_hours': total_duration / 3600,
                'average_success_rate': float(avg_success),
                'unique_concepts': unique_concepts,
                'daily_average_events': total_events / max(1, len(daily_activity))
            },
            'daily_activity': daily_activity,
            'concept_progress': concept_progress,
            'recommendations': self.get_learning_recommendations(user_id, list(concept_progress.keys()))
        }

    def identify_learning_gaps(self, user_id: str) -> List[Dict[str, Any]]:
        """Identify learning gaps and weak areas"""
        gaps = []

        if user_id not in self.user_progress:
            return gaps

        user_mastery = self.user_progress[user_id]

        # Find concepts with low mastery
        for concept_id, mastery in user_mastery.items():
            if mastery.mastery_score < 0.5 and mastery.attempts >= 3:
                # Significant gap - multiple attempts but low mastery
                gap = {
                    'concept': concept_id,
                    'mastery_score': mastery.mastery_score,
                    'attempts': mastery.attempts,
                    'last_practiced': mastery.last_practiced.isoformat(),
                    'severity': 'high' if mastery.mastery_score < 0.3 else 'medium',
                    'suggested_action': 'focused_practice',
                    'predicted_improvement': mastery.predicted_next_score - mastery.mastery_score
                }
                gaps.append(gap)

        # Sort by severity and recency
        gaps.sort(key=lambda x: (
            {'high': 0, 'medium': 1, 'low': 2}[x['severity']],
            x['mastery_score']
        ))

        return gaps

    def export_progress_data(self, user_id: str, format: str = 'json') -> str:
        """Export progress data in specified format"""
        progress = self.get_overall_progress(user_id)

        if format.lower() == 'json':
            import json
            return json.dumps(progress, indent=2, default=str)

        elif format.lower() == 'csv':
            import pandas as pd
            # Create DataFrame from progress data
            df_data = []
            if user_id in self.user_progress:
                for concept_id, mastery in self.user_progress[user_id].items():
                    df_data.append({
                        'concept': concept_id,
                        'mastery_score': mastery.mastery_score,
                        'attempts': mastery.attempts,
                        'correct_attempts': mastery.correct_attempts,
                        'last_practiced': mastery.last_practiced,
                        'predicted_next_score': mastery.predicted_next_score
                    })

            if df_data:
                df = pd.DataFrame(df_data)
                return df.to_csv(index=False)
            else:
                return "concept,mastery_score,attempts,correct_attempts,last_practiced,predicted_next_score\n"

        else:
            return f"Format {format} not supported"