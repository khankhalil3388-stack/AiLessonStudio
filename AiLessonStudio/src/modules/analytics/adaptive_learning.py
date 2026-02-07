import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


@dataclass
class LearnerProfile:
    """Comprehensive learner profile"""
    learner_id: str
    learning_style: str  # 'visual', 'auditory', 'kinesthetic', 'reading'
    preferred_difficulty: str
    pace: str  # 'slow', 'medium', 'fast'
    knowledge_level: str
    engagement_pattern: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    last_updated: datetime


@dataclass
class AdaptiveLesson:
    """Adaptive lesson configuration"""
    lesson_id: str
    base_topic: str
    difficulty_level: str
    content_type: str  # 'text', 'video', 'interactive', 'diagram'
    pacing: str
    assessment_strategy: str
    personalization_factors: Dict[str, Any]


class AdaptiveLearningSystem:
    """ML-powered adaptive learning system"""

    def __init__(self, config):
        self.config = config
        self.learner_profiles: Dict[str, LearnerProfile] = {}
        self.learning_models = {}
        self.recommendation_cache = {}

        # Initialize ML models
        self._init_ml_models()

        # Load existing profiles
        self._load_profiles()

        print("✅ Adaptive Learning System initialized")

    def _init_ml_models(self):
        """Initialize machine learning models"""
        # Model for predicting engagement
        self.engagement_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        # Model for predicting difficulty preference
        self.difficulty_model = RandomForestClassifier(
            n_estimators=50,
            random_state=42
        )

        # Label encoders
        self.style_encoder = LabelEncoder()
        self.difficulty_encoder = LabelEncoder()

        # Initialize with default values
        self.style_encoder.fit(['visual', 'auditory', 'kinesthetic', 'reading'])
        self.difficulty_encoder.fit(['beginner', 'intermediate', 'advanced'])

    def _load_profiles(self):
        """Load learner profiles from storage"""
        profiles_file = self.config.DATA_DIR / "progress" / "learner_profiles.json"
        if profiles_file.exists():
            try:
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)

                for learner_id, data in profiles_data.items():
                    profile = LearnerProfile(
                        learner_id=learner_id,
                        learning_style=data['learning_style'],
                        preferred_difficulty=data['preferred_difficulty'],
                        pace=data['pace'],
                        knowledge_level=data['knowledge_level'],
                        engagement_pattern=data['engagement_pattern'],
                        strengths=data['strengths'],
                        weaknesses=data['weaknesses'],
                        last_updated=datetime.fromisoformat(data['last_updated'])
                    )
                    self.learner_profiles[learner_id] = profile

                print(f"📊 Loaded {len(self.learner_profiles)} learner profiles")
            except Exception as e:
                print(f"⚠️ Error loading profiles: {e}")

    def _save_profiles(self):
        """Save learner profiles to storage"""
        profiles_file = self.config.DATA_DIR / "progress" / "learner_profiles.json"
        profiles_file.parent.mkdir(exist_ok=True)

        profiles_data = {}
        for learner_id, profile in self.learner_profiles.items():
            profiles_data[learner_id] = {
                'learning_style': profile.learning_style,
                'preferred_difficulty': profile.preferred_difficulty,
                'pace': profile.pace,
                'knowledge_level': profile.knowledge_level,
                'engagement_pattern': profile.engagement_pattern,
                'strengths': profile.strengths,
                'weaknesses': profile.weaknesses,
                'last_updated': profile.last_updated.isoformat()
            }

        with open(profiles_file, 'w') as f:
            json.dump(profiles_data, f, indent=2)

    def analyze_learning_style(self, events: List[Dict[str, Any]]) -> str:
        """Analyze learning style from interaction patterns"""
        if not events:
            return 'visual'  # Default

        # Count interaction types
        type_counts = {'text': 0, 'video': 0, 'interactive': 0, 'diagram': 0}
        durations = {'text': 0, 'video': 0, 'interactive': 0, 'diagram': 0}

        for event in events[-50:]:  # Last 50 events
            event_type = event.get('type', 'text')
            duration = event.get('duration', 0)

            if event_type in type_counts:
                type_counts[event_type] += 1
                durations[event_type] += duration

        # Calculate engagement scores
        engagement_scores = {}
        for style in type_counts:
            if durations[style] > 0:
                # Engagement = count * average duration
                avg_duration = durations[style] / max(1, type_counts[style])
                engagement_scores[style] = type_counts[style] * avg_duration

        # Determine dominant style
        if engagement_scores:
            dominant_style = max(engagement_scores.items(), key=lambda x: x[1])[0]

            # Map to learning styles
            style_mapping = {
                'text': 'reading',
                'video': 'auditory',
                'interactive': 'kinesthetic',
                'diagram': 'visual'
            }

            return style_mapping.get(dominant_style, 'visual')

        return 'visual'

    def update_learner_profile(self, learner_id: str,
                               events: List[Dict[str, Any]],
                               progress_data: Dict[str, Any]) -> LearnerProfile:
        """Update or create learner profile"""
        # Analyze learning style
        learning_style = self.analyze_learning_style(events)

        # Analyze pace from time between sessions
        pace = self._analyze_pace(events)

        # Analyze knowledge level from success rates
        knowledge_level = self._analyze_knowledge_level(progress_data)

        # Analyze preferred difficulty
        preferred_difficulty = self._analyze_difficulty_preference(events)

        # Analyze engagement patterns
        engagement_pattern = self._analyze_engagement_pattern(events)

        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(progress_data)

        # Create or update profile
        profile = LearnerProfile(
            learner_id=learner_id,
            learning_style=learning_style,
            preferred_difficulty=preferred_difficulty,
            pace=pace,
            knowledge_level=knowledge_level,
            engagement_pattern=engagement_pattern,
            strengths=strengths,
            weaknesses=weaknesses,
            last_updated=datetime.now()
        )

        self.learner_profiles[learner_id] = profile
        self._save_profiles()

        return profile

    def _analyze_pace(self, events: List[Dict[str, Any]]) -> str:
        """Analyze learning pace from events"""
        if len(events) < 3:
            return 'medium'

        # Calculate time between sessions
        timestamps = sorted([e.get('timestamp') for e in events if 'timestamp' in e])
        if len(timestamps) < 2:
            return 'medium'

        intervals = []
        for i in range(1, len(timestamps)):
            try:
                dt1 = datetime.fromisoformat(timestamps[i - 1])
                dt2 = datetime.fromisoformat(timestamps[i])
                interval_hours = (dt2 - dt1).total_seconds() / 3600
                intervals.append(interval_hours)
            except:
                continue

        if not intervals:
            return 'medium'

        avg_interval = np.mean(intervals)

        if avg_interval < 24:  # Less than 1 day
            return 'fast'
        elif avg_interval < 72:  # 1-3 days
            return 'medium'
        else:
            return 'slow'

    def _analyze_knowledge_level(self, progress_data: Dict[str, Any]) -> str:
        """Analyze overall knowledge level"""
        avg_mastery = progress_data.get('average_mastery', 0.0)

        if avg_mastery >= 0.8:
            return 'advanced'
        elif avg_mastery >= 0.5:
            return 'intermediate'
        else:
            return 'beginner'

    def _analyze_difficulty_preference(self, events: List[Dict[str, Any]]) -> str:
        """Analyze preferred difficulty level"""
        if not events:
            return 'intermediate'

        # Count successful attempts by difficulty
        success_by_difficulty = {'beginner': 0, 'intermediate': 0, 'advanced': 0}
        total_by_difficulty = {'beginner': 0, 'intermediate': 0, 'advanced': 0}

        for event in events[-30:]:  # Last 30 events
            difficulty = event.get('difficulty', 'intermediate')
            success_rate = event.get('success_rate', 0.0)

            if difficulty in total_by_difficulty:
                total_by_difficulty[difficulty] += 1
                if success_rate >= 0.7:
                    success_by_difficulty[difficulty] += 1

        # Calculate success rates
        success_rates = {}
        for diff in total_by_difficulty:
            if total_by_difficulty[diff] > 0:
                success_rates[diff] = success_by_difficulty[diff] / total_by_difficulty[diff]

        if not success_rates:
            return 'intermediate'

        # Find difficulty with highest success rate
        preferred = max(success_rates.items(), key=lambda x: x[1])[0]

        # Avoid recommending difficulty with very few attempts
        if total_by_difficulty[preferred] < 3:
            # Try second best
            sorted_diffs = sorted(success_rates.items(), key=lambda x: x[1], reverse=True)
            for diff, rate in sorted_diffs:
                if total_by_difficulty[diff] >= 3:
                    return diff

        return preferred

    def _analyze_engagement_pattern(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze engagement patterns"""
        patterns = {
            'morning_engagement': 0.0,  # 6am-12pm
            'afternoon_engagement': 0.0,  # 12pm-6pm
            'evening_engagement': 0.0,  # 6pm-12am
            'night_engagement': 0.0,  # 12am-6am
            'weekday_engagement': 0.0,
            'weekend_engagement': 0.0
        }

        if not events:
            return patterns

        for event in events[-100:]:  # Last 100 events
            if 'timestamp' not in event:
                continue

            try:
                dt = datetime.fromisoformat(event['timestamp'])
                hour = dt.hour
                weekday = dt.weekday() < 5  # Monday-Friday

                # Time of day
                if 6 <= hour < 12:
                    patterns['morning_engagement'] += 1
                elif 12 <= hour < 18:
                    patterns['afternoon_engagement'] += 1
                elif 18 <= hour < 24:
                    patterns['evening_engagement'] += 1
                else:
                    patterns['night_engagement'] += 1

                # Weekday vs weekend
                if weekday:
                    patterns['weekday_engagement'] += 1
                else:
                    patterns['weekend_engagement'] += 1

            except:
                continue

        # Normalize
        total_events = len([e for e in events[-100:] if 'timestamp' in e])
        if total_events > 0:
            for key in patterns:
                patterns[key] /= total_events

        return patterns

    def _identify_strengths_weaknesses(self, progress_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Identify learner strengths and weaknesses"""
        strengths = []
        weaknesses = []

        if 'concept_progress' not in progress_data:
            return strengths, weaknesses

        concept_progress = progress_data['concept_progress']

        for concept_id, data in concept_progress.items():
            if data.get('average_success', 0) >= 0.8:
                strengths.append(concept_id)
            elif data.get('average_success', 0) < 0.4 and data.get('attempts', 0) >= 3:
                weaknesses.append(concept_id)

        # Limit to top 5 each
        strengths = strengths[:5]
        weaknesses = weaknesses[:5]

        return strengths, weaknesses

    def generate_adaptive_lesson(self, learner_id: str, topic: str,
                                 base_difficulty: str = None) -> AdaptiveLesson:
        """Generate adaptive lesson based on learner profile"""
        # Get or create learner profile
        profile = self.learner_profiles.get(learner_id)
        if not profile:
            # Create default profile
            profile = LearnerProfile(
                learner_id=learner_id,
                learning_style='visual',
                preferred_difficulty='intermediate',
                pace='medium',
                knowledge_level='beginner',
                engagement_pattern={},
                strengths=[],
                weaknesses=[],
                last_updated=datetime.now()
            )

        # Determine difficulty level
        if base_difficulty:
            difficulty = base_difficulty
        else:
            difficulty = profile.preferred_difficulty

        # Determine content type based on learning style
        content_mapping = {
            'visual': ['diagram', 'interactive', 'video'],
            'auditory': ['video', 'interactive', 'text'],
            'kinesthetic': ['interactive', 'diagram', 'video'],
            'reading': ['text', 'diagram', 'interactive']
        }
        content_type = content_mapping.get(profile.learning_style, ['text'])[0]

        # Determine pacing
        pacing = profile.pace

        # Determine assessment strategy
        if profile.knowledge_level == 'beginner':
            assessment_strategy = 'formative_frequent'
        elif profile.knowledge_level == 'intermediate':
            assessment_strategy = 'mixed_strategy'
        else:
            assessment_strategy = 'summative_challenging'

        # Create adaptive lesson
        lesson_id = f"adaptive_{topic}_{learner_id}_{datetime.now().timestamp()}"

        lesson = AdaptiveLesson(
            lesson_id=lesson_id,
            base_topic=topic,
            difficulty_level=difficulty,
            content_type=content_type,
            pacing=pacing,
            assessment_strategy=assessment_strategy,
            personalization_factors={
                'learning_style': profile.learning_style,
                'preferred_difficulty': profile.preferred_difficulty,
                'pace': profile.pace,
                'knowledge_level': profile.knowledge_level,
                'engagement_pattern': profile.engagement_pattern,
                'strengths': profile.strengths,
                'weaknesses': profile.weaknesses
            }
        )

        return lesson

    def adjust_difficulty(self, learner_id: str, current_difficulty: str,
                          success_rate: float, time_spent: float) -> str:
        """Dynamically adjust difficulty based on performance"""
        profile = self.learner_profiles.get(learner_id)
        if not profile:
            return current_difficulty

        # Difficulty adjustment logic
        difficulty_levels = ['beginner', 'intermediate', 'advanced']
        current_index = difficulty_levels.index(current_difficulty)

        # Calculate adjustment
        if success_rate >= 0.8 and time_spent < 300:  # Excellent performance, fast
            # Increase difficulty
            if current_index < len(difficulty_levels) - 1:
                return difficulty_levels[current_index + 1]

        elif success_rate <= 0.4 or time_spent > 600:  # Poor performance or too slow
            # Decrease difficulty
            if current_index > 0:
                return difficulty_levels[current_index - 1]

        return current_difficulty

    def get_personalization_suggestions(self, learner_id: str) -> Dict[str, Any]:
        """Get personalized learning suggestions"""
        profile = self.learner_profiles.get(learner_id)
        if not profile:
            return {}

        suggestions = {
            'optimal_learning_time': self._get_optimal_time(profile),
            'recommended_content_types': self._get_recommended_content_types(profile),
            'suggested_study_duration': self._get_study_duration(profile),
            'learning_strategies': self._get_learning_strategies(profile),
            'attention_reminders': self._get_attention_reminders(profile)
        }

        return suggestions

    def _get_optimal_time(self, profile: LearnerProfile) -> str:
        """Determine optimal learning time"""
        engagement = profile.engagement_pattern

        if not engagement:
            return 'Afternoon (2-4 PM)'

        # Find peak engagement time
        time_slots = {
            'morning_engagement': 'Morning (8-10 AM)',
            'afternoon_engagement': 'Afternoon (2-4 PM)',
            'evening_engagement': 'Evening (7-9 PM)',
            'night_engagement': 'Night (10-12 PM)'
        }

        peak_time = max(
            [(k, v) for k, v in engagement.items() if k in time_slots],
            key=lambda x: x[1],
            default=('afternoon_engagement', 0)
        )[0]

        return time_slots.get(peak_time, 'Afternoon (2-4 PM)')

    def _get_recommended_content_types(self, profile: LearnerProfile) -> List[str]:
        """Get recommended content types based on learning style"""
        style_recommendations = {
            'visual': ['Diagrams', 'Infographics', 'Flowcharts', 'Videos with visuals'],
            'auditory': ['Video lectures', 'Podcasts', 'Audio explanations', 'Discussions'],
            'kinesthetic': ['Interactive labs', 'Hands-on exercises', 'Simulations', 'Code practice'],
            'reading': ['Text articles', 'E-books', 'Documentation', 'Case studies']
        }

        return style_recommendations.get(profile.learning_style, ['Mixed content'])

    def _get_study_duration(self, profile: LearnerProfile) -> str:
        """Get recommended study duration"""
        pace_mapping = {
            'slow': '25-30 minutes per session',
            'medium': '45-50 minutes per session',
            'fast': '60-75 minutes per session'
        }

        return pace_mapping.get(profile.pace, '45-50 minutes per session')

    def _get_learning_strategies(self, profile: LearnerProfile) -> List[str]:
        """Get personalized learning strategies"""
        strategies = []

        # Based on learning style
        if profile.learning_style == 'visual':
            strategies.extend([
                'Use mind maps for complex topics',
                'Create visual summaries of each lesson',
                'Watch explanatory videos before reading'
            ])
        elif profile.learning_style == 'auditory':
            strategies.extend([
                'Listen to explanations while doing other tasks',
                'Record yourself explaining concepts',
                'Participate in study groups'
            ])
        elif profile.learning_style == 'kinesthetic':
            strategies.extend([
                'Practice with hands-on labs',
                'Take frequent breaks for movement',
                'Use physical models or drawings'
            ])
        else:  # reading
            strategies.extend([
                'Take detailed notes while reading',
                'Summarize each section in your own words',
                'Create flashcards for key concepts'
            ])

        # Based on pace
        if profile.pace == 'fast':
            strategies.append('Try spaced repetition for long-term retention')
        elif profile.pace == 'slow':
            strategies.append('Break complex topics into smaller chunks')

        return strategies[:5]

    def _get_attention_reminders(self, profile: LearnerProfile) -> List[str]:
        """Get attention and focus reminders"""
        reminders = []

        if 'evening_engagement' in profile.engagement_pattern:
            if profile.engagement_pattern['evening_engagement'] > 0.6:
                reminders.append('Consider studying in natural light during daytime')

        if 'weekend_engagement' in profile.engagement_pattern:
            if profile.engagement_pattern['weekend_engagement'] < 0.3:
                reminders.append('Try short study sessions on weekends for consistency')

        return reminders

    def cluster_learners(self, learner_ids: List[str]) -> Dict[str, List[str]]:
        """Cluster learners based on profiles"""
        if len(learner_ids) < 3:
            return {'cluster_0': learner_ids}

        # Extract features from profiles
        features = []
        valid_learners = []

        for learner_id in learner_ids:
            profile = self.learner_profiles.get(learner_id)
            if profile:
                # Create feature vector
                feature = [
                    {'visual': 0, 'auditory': 1, 'kinesthetic': 2, 'reading': 3}[profile.learning_style],
                    {'beginner': 0, 'intermediate': 1, 'advanced': 2}[profile.knowledge_level],
                    {'slow': 0, 'medium': 1, 'fast': 2}[profile.pace]
                ]

                # Add engagement patterns
                engagement = profile.engagement_pattern
                feature.extend([
                    engagement.get('morning_engagement', 0),
                    engagement.get('afternoon_engagement', 0),
                    engagement.get('evening_engagement', 0),
                    engagement.get('weekday_engagement', 0)
                ])

                features.append(feature)
                valid_learners.append(learner_id)

        if len(features) < 3:
            return {'cluster_0': valid_learners}

        # Perform clustering
        from sklearn.cluster import KMeans

        n_clusters = min(3, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)

        # Group learners by cluster
        clustered = {}
        for learner_id, cluster in zip(valid_learners, clusters):
            cluster_key = f'cluster_{cluster}'
            if cluster_key not in clustered:
                clustered[cluster_key] = []
            clustered[cluster_key].append(learner_id)

        return clustered