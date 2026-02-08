import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import json


@dataclass
class LearningInsight:
    """Learning insight with evidence and recommendations"""
    type: str  # 'strength', 'weakness', 'pattern', 'opportunity'
    title: str
    description: str
    confidence: float
    evidence: List[str]
    impact: str  # 'high', 'medium', 'low'
    recommendations: List[str]


class InsightsGenerator:
    """Generate actionable insights from learning analytics"""

    def __init__(self, config):
        self.config = config
        self.insight_templates = self._load_insight_templates()
        print("✅ Insights Generator initialized")

    def _load_insight_templates(self) -> Dict[str, Any]:
        """Load insight templates"""
        return {
            'performance': {
                'excellent': {
                    'title': 'Consistent High Performance',
                    'description': 'You consistently perform well across different topics and assessment types.',
                    'recommendations': [
                        'Challenge yourself with more advanced topics',
                        'Consider helping peers who are struggling',
                        'Explore real-world applications of your knowledge'
                    ]
                },
                'improving': {
                    'title': 'Steady Improvement Detected',
                    'description': 'Your performance is showing consistent improvement over time.',
                    'recommendations': [
                        'Maintain your current study routine',
                        'Set specific goals for next milestones',
                        'Track your progress weekly'
                    ]
                },
                'variable': {
                    'title': 'Variable Performance Patterns',
                    'description': 'Your performance varies significantly across different topics or times.',
                    'recommendations': [
                        'Identify patterns in when you perform best',
                        'Review topics where performance is lower',
                        'Adjust study schedule based on insights'
                    ]
                }
            },
            'engagement': {
                'high': {
                    'title': 'High Engagement Level',
                    'description': 'You show strong engagement with consistent learning activities.',
                    'recommendations': [
                        'Continue with your current engagement level',
                        'Share your learning strategies with others',
                        'Consider taking on leadership roles in study groups'
                    ]
                },
                'consistent': {
                    'title': 'Consistent Learning Habits',
                    'description': 'You maintain regular learning sessions with good consistency.',
                    'recommendations': [
                        'Maintain your learning schedule',
                        'Try varying your learning methods',
                        'Set specific time-bound goals'
                    ]
                },
                'declining': {
                    'title': 'Declining Engagement Detected',
                    'description': 'Your learning engagement has decreased recently.',
                    'recommendations': [
                        'Revisit your learning goals',
                        'Try new learning formats or topics',
                        'Take short breaks to avoid burnout'
                    ]
                }
            },
            'learning_style': {
                'visual': {
                    'title': 'Visual Learning Preference',
                    'description': 'You learn best through diagrams, charts, and visual explanations.',
                    'recommendations': [
                        'Focus on visual learning materials',
                        'Create mind maps for complex topics',
                        'Use color coding in your notes'
                    ]
                },
                'practical': {
                    'title': 'Hands-On Learning Preference',
                    'description': 'You learn best through practical exercises and implementation.',
                    'recommendations': [
                        'Spend more time on hands-on labs',
                        'Build practical projects',
                        'Try implementing concepts in cloud simulator'
                    ]
                },
                'conceptual': {
                    'title': 'Conceptual Understanding Strength',
                    'description': 'You excel at understanding theoretical concepts and principles.',
                    'recommendations': [
                        'Focus on understanding underlying principles',
                        'Connect concepts to real-world scenarios',
                        'Teach concepts to reinforce understanding'
                    ]
                }
            }
        }

    def generate_insights(self, analytics_data: Dict[str, Any],
                          recent_activity: List[Dict],
                          user_profile: Dict[str, Any]) -> List[LearningInsight]:
        """Generate comprehensive learning insights"""
        insights = []

        # 1. Performance insights
        performance_insights = self._analyze_performance(
            analytics_data.get('performance', {}),
            recent_activity
        )
        insights.extend(performance_insights)

        # 2. Engagement insights
        engagement_insights = self._analyze_engagement(
            analytics_data.get('engagement', {}),
            recent_activity
        )
        insights.extend(engagement_insights)

        # 3. Learning style insights
        style_insights = self._analyze_learning_style(
            analytics_data.get('learning_style', {}),
            user_profile
        )
        insights.extend(style_insights)

        # 4. Knowledge gap insights
        gap_insights = self._analyze_knowledge_gaps(
            analytics_data.get('knowledge_gaps', {}),
            user_profile
        )
        insights.extend(gap_insights)

        # 5. Time management insights
        time_insights = self._analyze_time_management(
            analytics_data.get('time_metrics', {}),
            recent_activity
        )
        insights.extend(time_insights)

        # Sort by impact and confidence
        insights.sort(key=lambda x: (
            {'high': 0, 'medium': 1, 'low': 2}[x.impact],
            -x.confidence  # Higher confidence first
        ))

        return insights[:10]  # Return top 10 insights

    def _analyze_performance(self, performance_data: Dict,
                             recent_activity: List[Dict]) -> List[LearningInsight]:
        """Analyze performance patterns"""
        insights = []

        if not performance_data:
            return insights

        # Calculate performance metrics
        scores = performance_data.get('scores', [])
        if not scores:
            return insights

        avg_score = np.mean(scores)
        score_std = np.std(scores)

        # Determine performance pattern
        if avg_score >= 80 and score_std < 15:
            # Excellent and consistent
            template = self.insight_templates['performance']['excellent']
            confidence = 0.8

            evidence = [
                f'Average score: {avg_score:.1f}%',
                f'Consistency: Score variation is low (±{score_std:.1f}%)',
                f'Recent assessments: {len([s for s in scores[-3:] if s >= 75])}/3 above 75%'
            ]

        elif len(scores) >= 3 and scores[-1] > scores[0] + 10:
            # Improving
            template = self.insight_templates['performance']['improving']
            confidence = 0.7

            improvement = scores[-1] - scores[0]
            evidence = [
                f'Score improvement: +{improvement:.1f}% over time',
                f'Current score: {scores[-1]:.1f}%',
                f'Started at: {scores[0]:.1f}%'
            ]

        elif score_std > 20:
            # Variable
            template = self.insight_templates['performance']['variable']
            confidence = 0.6

            evidence = [
                f'High variability: Scores range from {min(scores):.1f}% to {max(scores):.1f}%',
                f'Standard deviation: {score_std:.1f}%',
                f'Average score: {avg_score:.1f}%'
            ]

        else:
            return insights

        insight = LearningInsight(
            type='strength' if avg_score >= 70 else 'weakness',
            title=template['title'],
            description=template['description'],
            confidence=confidence,
            evidence=evidence,
            impact='high' if avg_score >= 80 else 'medium',
            recommendations=template['recommendations']
        )

        insights.append(insight)

        # Topic-specific performance insights
        topic_scores = performance_data.get('topic_scores', {})
        if topic_scores:
            best_topic = max(topic_scores.items(), key=lambda x: x[1])
            worst_topic = min(topic_scores.items(), key=lambda x: x[1])

            if best_topic[1] >= 80:
                insights.append(LearningInsight(
                    type='strength',
                    title=f'Expertise in {best_topic[0].replace("_", " ").title()}',
                    description=f'You show excellent understanding of {best_topic[0].replace("_", " ")} with {best_topic[1]:.1f}% average score.',
                    confidence=0.8,
                    evidence=[f'Average score: {best_topic[1]:.1f}%'],
                    impact='medium',
                    recommendations=[
                        f'Share your knowledge about {best_topic[0].replace("_", " ")}',
                        f'Explore advanced applications of {best_topic[0].replace("_", " ")}',
                        f'Help others learn {best_topic[0].replace("_", " ")}'
                    ]
                ))

            if worst_topic[1] < 60:
                insights.append(LearningInsight(
                    type='weakness',
                    title=f'Opportunity in {worst_topic[0].replace("_", " ").title()}',
                    description=f'You have opportunity to improve in {worst_topic[0].replace("_", " ")} with {worst_topic[1]:.1f}% average score.',
                    confidence=0.7,
                    evidence=[f'Average score: {worst_topic[1]:.1f}%'],
                    impact='high',
                    recommendations=[
                        f'Review fundamentals of {worst_topic[0].replace("_", " ")}',
                        f'Practice more exercises on {worst_topic[0].replace("_", " ")}',
                        f'Seek additional resources for {worst_topic[0].replace("_", " ")}'
                    ]
                ))

        return insights

    def _analyze_engagement(self, engagement_data: Dict,
                            recent_activity: List[Dict]) -> List[LearningInsight]:
        """Analyze engagement patterns"""
        insights = []

        if not recent_activity:
            return insights

        # Calculate engagement metrics
        total_activities = len(recent_activity)
        days_active = len(set(
            datetime.fromisoformat(act['timestamp']).date()
            for act in recent_activity
            if 'timestamp' in act
        ))

        avg_daily_activities = total_activities / max(1, days_active)

        # Check engagement trend
        if len(recent_activity) >= 7:
            recent_week = recent_activity[-7:]
            first_half = recent_week[:3]
            second_half = recent_week[4:]

            recent_avg = len(second_half) / 3 if len(second_half) > 0 else 0
            earlier_avg = len(first_half) / 3 if len(first_half) > 0 else 0

            engagement_trend = 'increasing' if recent_avg > earlier_avg else 'decreasing'
        else:
            engagement_trend = 'stable'

        # Determine engagement level
        if avg_daily_activities >= 3:
            template = self.insight_templates['engagement']['high']
            confidence = 0.8
            impact = 'high'

            evidence = [
                f'Daily activities: {avg_daily_activities:.1f}',
                f'Days active: {days_active}',
                f'Total activities: {total_activities}'
            ]

        elif avg_daily_activities >= 1:
            template = self.insight_templates['engagement']['consistent']
            confidence = 0.7
            impact = 'medium'

            evidence = [
                f'Daily activities: {avg_daily_activities:.1f}',
                f'Consistent learning habit detected',
                f'Total activities: {total_activities}'
            ]

        elif engagement_trend == 'decreasing':
            template = self.insight_templates['engagement']['declining']
            confidence = 0.6
            impact = 'high'

            evidence = [
                f'Engagement trend: Decreasing',
                f'Recent activities: {len(recent_activity[-3:])} in last 3 days',
                f'Total activities: {total_activities}'
            ]

        else:
            return insights

        insight = LearningInsight(
            type='pattern',
            title=template['title'],
            description=template['description'],
            confidence=confidence,
            evidence=evidence,
            impact=impact,
            recommendations=template['recommendations']
        )

        insights.append(insight)

        # Time of day analysis
        if recent_activity:
            activity_times = []
            for act in recent_activity:
                if 'timestamp' in act:
                    try:
                        hour = datetime.fromisoformat(act['timestamp']).hour
                        activity_times.append(hour)
                    except:
                        pass

            if activity_times:
                hour_counts = pd.Series(activity_times).value_counts()
                if not hour_counts.empty:
                    peak_hour = hour_counts.idxmax()
                    peak_count = hour_counts.max()

                    if peak_count >= len(activity_times) * 0.3:  # 30% at same hour
                        insights.append(LearningInsight(
                            type='pattern',
                            title=f'Preferred Learning Time: {peak_hour}:00',
                            description=f'You tend to learn most actively around {peak_hour}:00.',
                            confidence=0.7,
                            evidence=[
                                f'{peak_count} activities at {peak_hour}:00',
                                f'{peak_count / len(activity_times) * 100:.1f}% of activities at this time'
                            ],
                            impact='medium',
                            recommendations=[
                                f'Schedule important learning sessions around {peak_hour}:00',
                                f'Use this time for challenging topics',
                                f'Plan breaks around other times'
                            ]
                        ))

        return insights

    def _analyze_learning_style(self, style_data: Dict,
                                user_profile: Dict[str, Any]) -> List[LearningInsight]:
        """Analyze learning style preferences"""
        insights = []

        if not style_data:
            return insights

        # Analyze content type preferences
        content_preferences = style_data.get('content_preferences', {})
        if content_preferences:
            preferred_type = max(content_preferences.items(), key=lambda x: x[1])

            if preferred_type[0] == 'visual' and preferred_type[1] > 0.6:
                template = self.insight_templates['learning_style']['visual']
                confidence = 0.8

                evidence = [
                    f'Visual content preference: {preferred_type[1] * 100:.1f}%',
                    'High engagement with diagrams and charts',
                    'Better performance on visual-based assessments'
                ]

            elif preferred_type[0] == 'practical' and preferred_type[1] > 0.6:
                template = self.insight_templates['learning_style']['practical']
                confidence = 0.8

                evidence = [
                    f'Practical content preference: {preferred_type[1] * 100:.1f}%',
                    'High engagement with hands-on exercises',
                    'Better retention with practical applications'
                ]

            elif preferred_type[0] == 'conceptual' and preferred_type[1] > 0.6:
                template = self.insight_templates['learning_style']['conceptual']
                confidence = 0.7

                evidence = [
                    f'Conceptual content preference: {preferred_type[1] * 100:.1f}%',
                    'Strong performance on theory-based assessments',
                    'Good understanding of underlying principles'
                ]
            else:
                return insights

            insight = LearningInsight(
                type='strength',
                title=template['title'],
                description=template['description'],
                confidence=confidence,
                evidence=evidence,
                impact='medium',
                recommendations=template['recommendations']
            )

            insights.append(insight)

        # Learning pace analysis
        pace_data = style_data.get('learning_pace', {})
        if pace_data:
            avg_pace = pace_data.get('average', 0)
            if avg_pace > 0:
                if avg_pace < 30:  # Fast learner
                    insights.append(LearningInsight(
                        type='strength',
                        title='Fast Learning Pace',
                        description='You process and understand new concepts quickly.',
                        confidence=0.7,
                        evidence=[
                            f'Average learning pace: {avg_pace:.1f} minutes per concept',
                            'Quick grasp of new material',
                            'Efficient study sessions'
                        ],
                        impact='medium',
                        recommendations=[
                            'Take on more challenging topics',
                            'Help explain concepts to peers',
                            'Consider accelerated learning paths'
                        ]
                    ))
                elif avg_pace > 60:  # Thorough learner
                    insights.append(LearningInsight(
                        type='pattern',
                        title='Thorough Learning Approach',
                        description='You take time to deeply understand concepts.',
                        confidence=0.7,
                        evidence=[
                            f'Average learning pace: {avg_pace:.1f} minutes per concept',
                            'Deep understanding of material',
                            'Comprehensive knowledge retention'
                        ],
                        impact='medium',
                        recommendations=[
                            'Focus on understanding rather than speed',
                            'Create detailed notes and summaries',
                            'Review material periodically for retention'
                        ]
                    ))

        return insights

    def _analyze_knowledge_gaps(self, gap_data: Dict,
                                user_profile: Dict[str, Any]) -> List[LearningInsight]:
        """Analyze knowledge gaps and opportunities"""
        insights = []

        if not gap_data:
            return insights

        gaps = gap_data.get('identified_gaps', [])
        if not gaps:
            return insights

        # Categorize gaps
        beginner_gaps = [g for g in gaps if g.get('difficulty') == 'beginner']
        intermediate_gaps = [g for g in gaps if g.get('difficulty') == 'intermediate']
        advanced_gaps = [g for g in gaps if g.get('difficulty') == 'advanced']

        # Generate insights based on gap types
        if beginner_gaps:
            insights.append(LearningInsight(
                type='opportunity',
                title='Foundation Building Opportunity',
                description=f'You have {len(beginner_gaps)} foundational concepts to strengthen.',
                confidence=0.8,
                evidence=[
                    f'Beginner-level gaps: {len(beginner_gaps)}',
                    'These form the basis for advanced topics',
                    'Strengthening these will improve overall understanding'
                ],
                impact='high',
                recommendations=[
                    'Review basic concepts systematically',
                    'Practice fundamental exercises',
                    'Build strong foundation before advancing'
                ]
            ))

        if intermediate_gaps:
            insights.append(LearningInsight(
                type='opportunity',
                title='Skill Development Opportunity',
                description=f'You have {len(intermediate_gaps)} intermediate skills to develop.',
                confidence=0.7,
                evidence=[
                    f'Intermediate-level gaps: {len(intermediate_gaps)}',
                    'These are practical application skills',
                    'Important for real-world implementation'
                ],
                impact='medium',
                recommendations=[
                    'Focus on practical applications',
                    'Work on hands-on projects',
                    'Practice implementation scenarios'
                ]
            ))

        if advanced_gaps:
            insights.append(LearningInsight(
                type='opportunity',
                title='Expertise Development Opportunity',
                description=f'You have {len(advanced_gaps)} advanced topics to master.',
                confidence=0.6,
                evidence=[
                    f'Advanced-level gaps: {len(advanced_gaps)}',
                    'These represent specialized knowledge',
                    'Mastery leads to expert-level understanding'
                ],
                impact='medium',
                recommendations=[
                    'Study specialized materials',
                    'Work on complex projects',
                    'Seek expert guidance if needed'
                ]
            ))

        # Gap patterns
        if gaps:
            gap_topics = [g.get('topic', '') for g in gaps]
            topic_categories = defaultdict(list)

            for gap in gaps:
                topic = gap.get('topic', '')
                if 'security' in topic.lower():
                    topic_categories['security'].append(gap)
                elif 'scalability' in topic.lower():
                    topic_categories['scalability'].append(gap)
                elif 'performance' in topic.lower():
                    topic_categories['performance'].append(gap)

            for category, category_gaps in topic_categories.items():
                if len(category_gaps) >= 2:
                    insights.append(LearningInsight(
                        type='pattern',
                        title=f'{category.title()} Knowledge Gap Pattern',
                        description=f'You have multiple gaps in {category} related topics.',
                        confidence=0.7,
                        evidence=[
                            f'{len(category_gaps)} gaps in {category}',
                            'Pattern suggests specific area needs focus',
                            'Connected concepts may be affected'
                        ],
                        impact='medium',
                        recommendations=[
                            f'Study {category} as a focused topic',
                            f'Look for connections between {category} concepts',
                            f'Practice {category} scenarios specifically'
                        ]
                    ))

        return insights

    def _analyze_time_management(self, time_data: Dict,
                                 recent_activity: List[Dict]) -> List[LearningInsight]:
        """Analyze time management patterns"""
        insights = []

        if not recent_activity:
            return insights

        # Calculate time metrics
        session_durations = []
        session_times = []

        for act in recent_activity:
            if 'duration' in act and act['duration']:
                session_durations.append(act['duration'])

            if 'timestamp' in act:
                try:
                    hour = datetime.fromisoformat(act['timestamp']).hour
                    session_times.append(hour)
                except:
                    pass

        if not session_durations:
            return insights

        # Analyze session length patterns
        avg_duration = np.mean(session_durations)
        duration_std = np.std(session_durations)

        if avg_duration < 15:
            insights.append(LearningInsight(
                type='pattern',
                title='Short, Frequent Learning Sessions',
                description='You prefer short learning sessions, which can be effective for retention.',
                confidence=0.7,
                evidence=[
                    f'Average session: {avg_duration:.1f} minutes',
                    'Frequent short sessions detected',
                    'Good for spaced repetition'
                ],
                impact='medium',
                recommendations=[
                    'Continue with spaced learning approach',
                    'Use short sessions for review and practice',
                    'Combine with occasional longer deep-dive sessions'
                ]
            ))
        elif avg_duration > 45:
            insights.append(LearningInsight(
                type='pattern',
                title='Deep Focus Learning Sessions',
                description='You engage in longer, focused learning sessions.',
                confidence=0.7,
                evidence=[
                    f'Average session: {avg_duration:.1f} minutes',
                    'Extended focus periods',
                    'Good for complex topics'
                ],
                impact='medium',
                recommendations=[
                    'Schedule breaks during long sessions',
                    'Use longer sessions for complex material',
                    'Combine with active recall techniques'
                ]
            ))

        # Session consistency
        if duration_std > avg_duration * 0.5:  # High variability
            insights.append(LearningInsight(
                type='pattern',
                title='Variable Session Lengths',
                description='Your learning session lengths vary significantly.',
                confidence=0.6,
                evidence=[
                    f'Average session: {avg_duration:.1f} minutes',
                    f'Variability: ±{duration_std:.1f} minutes',
                    'Session length depends on topic or time available'
                ],
                impact='low',
                recommendations=[
                    'Try to establish more consistent session lengths',
                    'Match session length to topic complexity',
                    'Track what session lengths work best for you'
                ]
            ))

        # Time of day efficiency
        if session_times:
            morning_sessions = [t for t in session_times if 6 <= t < 12]
            afternoon_sessions = [t for t in session_times if 12 <= t < 18]
            evening_sessions = [t for t in session_times if 18 <= t < 24]

            session_counts = {
                'Morning': len(morning_sessions),
                'Afternoon': len(afternoon_sessions),
                'Evening': len(evening_sessions)
            }

            preferred_time = max(session_counts.items(), key=lambda x: x[1])

            if preferred_time[1] >= len(session_times) * 0.4:  # 40% at preferred time
                insights.append(LearningInsight(
                    type='pattern',
                    title=f'Preferred Learning Time: {preferred_time[0]}',
                    description=f'You learn most actively during the {preferred_time[0].lower()}.',
                    confidence=0.7,
                    evidence=[
                        f'{preferred_time[1]} sessions in {preferred_time[0]}',
                        f'{preferred_time[1] / len(session_times) * 100:.1f}% of sessions'
                    ],
                    impact='medium',
                    recommendations=[
                        f'Schedule important learning in {preferred_time[0]}',
                        f'Use {preferred_time[0]} for challenging topics',
                        f'Plan lighter activities for other times'
                    ]
                ))

        return insights

    def generate_summary_report(self, insights: List[LearningInsight]) -> Dict[str, Any]:
        """Generate summary report from insights"""
        if not insights:
            return {'error': 'No insights available'}

        # Categorize insights
        strength_insights = [i for i in insights if i.type == 'strength']
        weakness_insights = [i for i in insights if i.type == 'weakness']
        pattern_insights = [i for i in insights if i.type == 'pattern']
        opportunity_insights = [i for i in insights if i.type == 'opportunity']

        # Calculate overall metrics
        total_insights = len(insights)
        high_impact = len([i for i in insights if i.impact == 'high'])
        medium_impact = len([i for i in insights if i.impact == 'medium'])
        high_confidence = len([i for i in insights if i.confidence >= 0.7])

        # Generate summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_insights': total_insights,
            'insight_distribution': {
                'strengths': len(strength_insights),
                'weaknesses': len(weakness_insights),
                'patterns': len(pattern_insights),
                'opportunities': len(opportunity_insights)
            },
            'impact_levels': {
                'high': high_impact,
                'medium': medium_impact,
                'low': total_insights - high_impact - medium_impact
            },
            'confidence_levels': {
                'high': high_confidence,
                'medium': total_insights - high_confidence
            },
            'key_strengths': [
                {
                    'title': i.title,
                    'confidence': i.confidence,
                    'impact': i.impact
                }
                for i in strength_insights[:3]
            ],
            'key_opportunities': [
                {
                    'title': i.title,
                    'confidence': i.confidence,
                    'impact': i.impact
                }
                for i in opportunity_insights[:3]
            ],
            'action_items': self._generate_action_items(insights),
            'overall_assessment': self._generate_overall_assessment(insights)
        }

        return summary

    def _generate_action_items(self, insights: List[LearningInsight]) -> List[Dict[str, Any]]:
        """Generate actionable items from insights"""
        action_items = []

        # High impact insights first
        high_impact = [i for i in insights if i.impact == 'high']

        for insight in high_impact[:3]:
            for rec in insight.recommendations[:2]:  # Top 2 recommendations
                action_items.append({
                    'action': rec,
                    'source': insight.title,
                    'priority': 'high',
                    'estimated_time': '30-60 minutes',
                    'category': insight.type
                })

        # Medium impact insights
        medium_impact = [i for i in insights if i.impact == 'medium']

        for insight in medium_impact[:2]:
            for rec in insight.recommendations[:1]:  # Top recommendation
                action_items.append({
                    'action': rec,
                    'source': insight.title,
                    'priority': 'medium',
                    'estimated_time': '20-40 minutes',
                    'category': insight.type
                })

        return action_items[:5]  # Return top 5 action items

    def _generate_overall_assessment(self, insights: List[LearningInsight]) -> Dict[str, Any]:
        """Generate overall assessment"""
        if not insights:
            return {'level': 'beginning', 'description': 'Starting learning journey'}

        # Calculate scores
        strength_score = len([i for i in insights if i.type == 'strength' and i.confidence >= 0.7])
        opportunity_score = len([i for i in insights if i.type in ['weakness', 'opportunity'] and i.impact == 'high'])

        total_strengths = len([i for i in insights if i.type == 'strength'])
        total_opportunities = len([i for i in insights if i.type in ['weakness', 'opportunity']])

        # Determine learning stage
        if total_strengths >= 3 and opportunity_score == 0:
            stage = 'advanced'
            description = 'Strong foundation with few knowledge gaps'
        elif total_strengths >= 2 and opportunity_score <= 1:
            stage = 'intermediate'
            description = 'Good progress with some areas for improvement'
        elif total_strengths >= 1:
            stage = 'developing'
            description = 'Making progress with clear learning path'
        else:
            stage = 'beginning'
            description = 'Starting learning journey with many opportunities'

        return {
            'stage': stage,
            'description': description,
            'strengths_count': total_strengths,
            'opportunities_count': total_opportunities,
            'recommendation': self._get_stage_recommendation(stage)
        }

    def _get_stage_recommendation(self, stage: str) -> str:
        """Get recommendation based on learning stage"""
        recommendations = {
            'beginning': 'Focus on building strong fundamentals and consistent learning habits.',
            'developing': 'Work on strengthening weak areas while expanding knowledge base.',
            'intermediate': 'Challenge yourself with complex topics and practical applications.',
            'advanced': 'Pursue specialized knowledge and consider mentoring others.'
        }
        return recommendations.get(stage, 'Continue with your current learning approach.')