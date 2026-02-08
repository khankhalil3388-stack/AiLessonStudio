import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import random


@dataclass
class Recommendation:
    """Learning recommendation"""
    type: str  # 'lesson', 'quiz', 'practice', 'review'
    topic: str
    reason: str
    priority: str  # 'high', 'medium', 'low'
    estimated_time: int  # minutes
    confidence: float
    metadata: Dict[str, Any]


class RecommenderSystem:
    """Intelligent recommendation system for personalized learning"""

    def __init__(self, config):
        self.config = config
        self.user_profiles = {}
        self.content_features = {}
        self.collaborative_filter = {}

        self._initialize_content_features()
        print("✅ Recommender System initialized")

    def _initialize_content_features(self):
        """Initialize content feature database"""
        # Cloud computing topics with features
        self.content_features = {
            'cloud_computing_basics': {
                'difficulty': 'beginner',
                'duration': 30,
                'topics': ['introduction', 'benefits', 'service_models'],
                'prerequisites': [],
                'style': ['conceptual', 'visual']
            },
            'virtualization': {
                'difficulty': 'beginner',
                'duration': 45,
                'topics': ['hypervisors', 'vms', 'resource_allocation'],
                'prerequisites': ['cloud_computing_basics'],
                'style': ['technical', 'hands-on']
            },
            'containerization': {
                'difficulty': 'intermediate',
                'duration': 60,
                'topics': ['docker', 'images', 'containers'],
                'prerequisites': ['virtualization'],
                'style': ['technical', 'practical']
            },
            'kubernetes': {
                'difficulty': 'intermediate',
                'duration': 90,
                'topics': ['pods', 'services', 'deployments'],
                'prerequisites': ['containerization'],
                'style': ['technical', 'complex']
            },
            'serverless': {
                'difficulty': 'intermediate',
                'duration': 45,
                'topics': ['functions', 'events', 'scaling'],
                'prerequisites': ['cloud_computing_basics'],
                'style': ['conceptual', 'practical']
            },
            'cloud_security': {
                'difficulty': 'advanced',
                'duration': 60,
                'topics': ['iam', 'encryption', 'compliance'],
                'prerequisites': ['cloud_computing_basics'],
                'style': ['technical', 'conceptual']
            },
            'cloud_architecture': {
                'difficulty': 'advanced',
                'duration': 75,
                'topics': ['design_patterns', 'scalability', 'reliability'],
                'prerequisites': ['cloud_computing_basics', 'serverless'],
                'style': ['conceptual', 'technical']
            }
        }

    def get_recommendations(self, user_id: str,
                            user_profile: Dict[str, Any],
                            recent_activity: List[Dict],
                            available_topics: List[str],
                            max_recommendations: int = 5) -> List[Recommendation]:
        """Get personalized learning recommendations"""
        recommendations = []

        # 1. Knowledge gap analysis
        gap_recommendations = self._analyze_knowledge_gaps(
            user_profile, recent_activity, available_topics
        )
        recommendations.extend(gap_recommendations)

        # 2. Learning path continuation
        path_recommendations = self._continue_learning_path(
            user_profile, recent_activity, available_topics
        )
        recommendations.extend(path_recommendations)

        # 3. Skill reinforcement
        reinforcement_recommendations = self._recommend_reinforcement(
            recent_activity, available_topics
        )
        recommendations.extend(reinforcement_recommendations)

        # 4. Explore new areas
        exploration_recommendations = self._recommend_exploration(
            user_profile, available_topics
        )
        recommendations.extend(exploration_recommendations)

        # Sort by priority and remove duplicates
        recommendations = self._deduplicate_recommendations(recommendations)
        recommendations.sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x.priority])

        return recommendations[:max_recommendations]

    def _analyze_knowledge_gaps(self, user_profile: Dict[str, Any],
                                recent_activity: List[Dict],
                                available_topics: List[str]) -> List[Recommendation]:
        """Analyze and recommend based on knowledge gaps"""
        recommendations = []

        # Get user's mastery scores
        mastery_scores = user_profile.get('mastery_scores', {})

        # Find topics with low mastery
        for topic, score in mastery_scores.items():
            if score < 0.5 and topic in available_topics:
                # Check if recently practiced
                recently_practiced = any(
                    act.get('topic') == topic
                    for act in recent_activity[-5:]  # Last 5 activities
                )

                if not recently_practiced:
                    recommendations.append(
                        Recommendation(
                            type='review',
                            topic=topic,
                            reason=f'Low mastery score ({score:.0%})',
                            priority='high' if score < 0.3 else 'medium',
                            estimated_time=30,
                            confidence=0.8,
                            metadata={'mastery_score': score}
                        )
                    )

        return recommendations

    def _continue_learning_path(self, user_profile: Dict[str, Any],
                                recent_activity: List[Dict],
                                available_topics: List[str]) -> List[Recommendation]:
        """Recommend next steps in learning path"""
        recommendations = []

        # Get recently learned topics
        recent_topics = [
            act.get('topic') for act in recent_activity[-3:]
            if act.get('topic') and act.get('success_rate', 0) > 0.7
        ]

        if not recent_topics:
            return recommendations

        # Find topics that build on recent learning
        for recent_topic in recent_topics:
            if recent_topic in self.content_features:
                # Find prerequisites for other topics
                for topic, features in self.content_features.items():
                    if topic in available_topics and topic != recent_topic:
                        prerequisites = features.get('prerequisites', [])
                        if recent_topic in prerequisites:
                            # Check if user is ready for this topic
                            ready = self._check_topic_readiness(
                                topic, user_profile, recent_activity
                            )

                            if ready:
                                recommendations.append(
                                    Recommendation(
                                        type='lesson',
                                        topic=topic,
                                        reason=f'Builds on your knowledge of {recent_topic}',
                                        priority='medium',
                                        estimated_time=features.get('duration', 45),
                                        confidence=0.7,
                                        metadata={
                                            'builds_on': recent_topic,
                                            'difficulty': features.get('difficulty', 'intermediate')
                                        }
                                    )
                                )

        return recommendations

    def _recommend_reinforcement(self, recent_activity: List[Dict],
                                 available_topics: List[str]) -> List[Recommendation]:
        """Recommend reinforcement for recently learned topics"""
        recommendations = []

        # Group activities by topic
        topic_activities = defaultdict(list)
        for activity in recent_activity:
            topic = activity.get('topic')
            if topic:
                topic_activities[topic].append(activity)

        # Find topics that need reinforcement
        for topic, activities in topic_activities.items():
            if topic not in available_topics:
                continue

            # Calculate average success rate
            success_rates = [act.get('success_rate', 0) for act in activities]
            avg_success = np.mean(success_rates) if success_rates else 0

            # Check time since last practice
            if activities:
                last_activity = max(activities, key=lambda x: x.get('timestamp', ''))
                last_time = datetime.fromisoformat(last_activity.get('timestamp', datetime.now().isoformat()))
                days_since = (datetime.now() - last_time).days

                # Recommend reinforcement if needed
                if avg_success < 0.8 or days_since > 7:
                    recommendations.append(
                        Recommendation(
                            type='practice',
                            topic=topic,
                            reason=f'Reinforce learning ({avg_success:.0%} success, {days_since} days ago)',
                            priority='medium' if avg_success < 0.7 else 'low',
                            estimated_time=20,
                            confidence=0.6,
                            metadata={
                                'avg_success': avg_success,
                                'days_since': days_since,
                                'activity_count': len(activities)
                            }
                        )
                    )

        return recommendations

    def _recommend_exploration(self, user_profile: Dict[str, Any],
                               available_topics: List[str]) -> List[Recommendation]:
        """Recommend new areas to explore"""
        recommendations = []

        # Get user's current level
        user_level = user_profile.get('knowledge_level', 'beginner')

        # Find topics at appropriate level that haven't been explored
        explored_topics = user_profile.get('explored_topics', [])

        for topic, features in self.content_features.items():
            if (topic in available_topics and
                    topic not in explored_topics and
                    features.get('difficulty', 'intermediate') in self._get_appropriate_levels(user_level)):

                # Check if prerequisites are met
                prerequisites = features.get('prerequisites', [])
                prerequisites_met = all(
                    prereq in explored_topics or
                    user_profile.get('mastery_scores', {}).get(prereq, 0) > 0.7
                    for prereq in prerequisites
                )

                if prerequisites_met or not prerequisites:
                    recommendations.append(
                        Recommendation(
                            type='lesson',
                            topic=topic,
                            reason=f'New topic at your level ({features.get("difficulty")})',
                            priority='low',
                            estimated_time=features.get('duration', 45),
                            confidence=0.5,
                            metadata={
                                'difficulty': features.get('difficulty'),
                                'prerequisites': prerequisites
                            }
                        )
                    )

        return recommendations

    def _check_topic_readiness(self, topic: str, user_profile: Dict[str, Any],
                               recent_activity: List[Dict]) -> bool:
        """Check if user is ready for a topic"""
        if topic not in self.content_features:
            return False

        features = self.content_features[topic]
        prerequisites = features.get('prerequisites', [])

        # Check mastery of prerequisites
        mastery_scores = user_profile.get('mastery_scores', {})
        for prereq in prerequisites:
            if mastery_scores.get(prereq, 0) < 0.5:
                return False

        # Check recent activity on related topics
        related_topics = features.get('topics', [])
        recent_related = any(
            act.get('topic') in related_topics
            for act in recent_activity[-5:]
        )

        return recent_related or not prerequisites

    def _get_appropriate_levels(self, user_level: str) -> List[str]:
        """Get appropriate difficulty levels for user"""
        level_mapping = {
            'beginner': ['beginner'],
            'intermediate': ['beginner', 'intermediate'],
            'advanced': ['beginner', 'intermediate', 'advanced']
        }
        return level_mapping.get(user_level, ['beginner', 'intermediate'])

    def _deduplicate_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Remove duplicate recommendations"""
        seen = set()
        unique = []

        for rec in recommendations:
            key = (rec.type, rec.topic)
            if key not in seen:
                seen.add(key)
                unique.append(rec)

        return unique

    def generate_learning_path(self, start_topic: str, goal_topic: str,
                               user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate learning path from start to goal"""
        path = []

        # Simple BFS for learning path
        visited = set()
        queue = [(start_topic, [start_topic])]

        while queue:
            current, current_path = queue.pop(0)

            if current == goal_topic:
                path = current_path
                break

            if current in visited:
                continue

            visited.add(current)

            # Get next topics
            if current in self.content_features:
                # Find topics where this is a prerequisite
                for topic, features in self.content_features.items():
                    if current in features.get('prerequisites', []):
                        queue.append((topic, current_path + [topic]))

        # Convert to learning steps
        learning_steps = []
        for i, topic in enumerate(path):
            if topic in self.content_features:
                features = self.content_features[topic]

                step = {
                    'step': i + 1,
                    'topic': topic,
                    'type': 'lesson',
                    'estimated_time': features.get('duration', 45),
                    'difficulty': features.get('difficulty', 'intermediate'),
                    'prerequisites': features.get('prerequisites', []),
                    'learning_objectives': [
                        f"Understand {topic.replace('_', ' ')}",
                        f"Apply {topic.replace('_', ' ')} concepts",
                        f"Practice {topic.replace('_', ' ')} implementation"
                    ]
                }

                # Add practice step after each topic
                if i < len(path) - 1:
                    learning_steps.append(step)

                    practice_step = {
                        'step': i + 1.5,
                        'topic': topic,
                        'type': 'practice',
                        'estimated_time': 30,
                        'difficulty': features.get('difficulty', 'intermediate'),
                        'prerequisites': [],
                        'learning_objectives': [
                            f"Reinforce {topic.replace('_', ' ')} knowledge",
                            f"Apply concepts in exercises",
                            f"Test understanding"
                        ]
                    }
                    learning_steps.append(practice_step)
                else:
                    learning_steps.append(step)

        return learning_steps

    def adapt_recommendations_based_on_feedback(self, user_id: str,
                                                recommendation: Recommendation,
                                                feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt recommendations based on user feedback"""
        adaptation = {
            'adjusted': False,
            'new_recommendation': None,
            'reason': ''
        }

        feedback_score = feedback.get('score', 0)  # 0-1 scale
        feedback_difficulty = feedback.get('difficulty', 'appropriate')
        feedback_interest = feedback.get('interest', 'neutral')

        # Analyze feedback
        if feedback_score < 0.3:
            # User struggled, recommend easier material
            adaptation['adjusted'] = True
            adaptation['reason'] = 'User found material too difficult'

            # Find easier version of same topic
            current_features = self.content_features.get(recommendation.topic, {})
            current_difficulty = current_features.get('difficulty', 'intermediate')

            # Look for easier prerequisites
            prerequisites = current_features.get('prerequisites', [])
            for prereq in prerequisites:
                prereq_features = self.content_features.get(prereq, {})
                if prereq_features.get('difficulty', 'beginner') < current_difficulty:
                    adaptation['new_recommendation'] = Recommendation(
                        type='review',
                        topic=prereq,
                        reason='Build stronger foundation',
                        priority='high',
                        estimated_time=prereq_features.get('duration', 45),
                        confidence=0.9,
                        metadata={'original_topic': recommendation.topic}
                    )
                    break

        elif feedback_difficulty == 'too_easy':
            # User found it too easy, recommend more challenging material
            adaptation['adjusted'] = True
            adaptation['reason'] = 'User found material too easy'

            # Find more advanced topics
            current_features = self.content_features.get(recommendation.topic, {})

            # Look for topics that require this as prerequisite
            for topic, features in self.content_features.items():
                if recommendation.topic in features.get('prerequisites', []):
                    adaptation['new_recommendation'] = Recommendation(
                        type='lesson',
                        topic=topic,
                        reason='Challenge yourself with advanced material',
                        priority='medium',
                        estimated_time=features.get('duration', 60),
                        confidence=0.7,
                        metadata={'builds_on': recommendation.topic}
                    )
                    break

        elif feedback_interest in ['high', 'very_high']:
            # User interested, recommend related topics
            adaptation['adjusted'] = True
            adaptation['reason'] = 'User showed high interest'

            # Find related topics
            related_topics = self._find_related_topics(recommendation.topic)
            if related_topics:
                next_topic = random.choice(related_topics)
                next_features = self.content_features.get(next_topic, {})

                adaptation['new_recommendation'] = Recommendation(
                    type='lesson',
                    topic=next_topic,
                    reason='Explore related topic of interest',
                    priority='medium',
                    estimated_time=next_features.get('duration', 45),
                    confidence=0.6,
                    metadata={'related_to': recommendation.topic}
                )

        return adaptation

    def _find_related_topics(self, topic: str) -> List[str]:
        """Find topics related to given topic"""
        related = []

        if topic in self.content_features:
            features = self.content_features[topic]

            # Topics that share prerequisites
            for other_topic, other_features in self.content_features.items():
                if other_topic != topic:
                    # Check shared topics
                    shared_topics = set(features.get('topics', [])).intersection(
                        set(other_features.get('topics', []))
                    )

                    if shared_topics:
                        related.append(other_topic)

        return related[:3]  # Return top 3 related topics