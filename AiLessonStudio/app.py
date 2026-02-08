import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import torch
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import configuration
from config import config

# Import core components
from src.core.ai_engine import CompleteAIEngine
from src.textbook.processor import AdvancedTextbookProcessor
from src.modules.code_executor.cloud_simulator import CloudSimulator
from src.modules.diagrams.generator import DiagramGenerator
from src.modules.assessments.evaluator import AssessmentEvaluator
from src.modules.assessments.quiz_generator import QuizGenerator
from src.modules.analytics.tracker import ProgressTracker
from src.modules.analytics.adaptive_learning import AdaptiveLearningSystem

# Import the missing modules from your additional code
from src.core.lesson_generator import LessonGenerator
from src.core.qa_system import IntelligentQASystem
from src.core.content_analyzer import ContentAnalyzer
from src.modules.assessments.feedback_system import FeedbackSystem
from src.modules.analytics.recommender import RecommenderSystem
from src.modules.analytics.insights import InsightsGenerator

# Page configuration
st.set_page_config(
    page_title="AI Lesson Studio - Complete Cloud Learning Platform",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ailessonstudio',
        'Report a bug': "https://github.com/ailessonstudio/issues",
        'About': "AI Lesson Studio v2.0 - Complete Cloud Computing Education Platform"
    }
)


# Initialize session state with all systems
def init_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.textbook_processor = None
        st.session_state.ai_engine = None
        st.session_state.cloud_simulator = None
        st.session_state.progress_tracker = ProgressTracker(config)
        st.session_state.adaptive_learning = AdaptiveLearningSystem(config)

        # Initialize the new systems from your code
        st.session_state.lesson_generator = LessonGenerator(config)
        st.session_state.qa_system = IntelligentQASystem(config)
        st.session_state.content_analyzer = ContentAnalyzer(config)
        st.session_state.recommender = RecommenderSystem(config)
        st.session_state.insights_generator = InsightsGenerator(config)
        st.session_state.feedback_system = FeedbackSystem(config)

        # Initialize QuizGenerator (already imported)
        st.session_state.quiz_generator = QuizGenerator(config)

        st.session_state.current_user = None
        st.session_state.textbook_loaded = False
        st.session_state.current_lesson = None
        st.session_state.learning_path = []
        st.session_state.code_outputs = {}
        st.session_state.quiz_responses = {}
        st.session_state.diagrams_generated = {}
        st.session_state.current_quiz = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        st.session_state.quiz_result = None
        st.session_state.custom_nodes = []
        st.session_state.custom_edges = []

        # Add additional state variables for new features
        st.session_state.lessons_generated = []
        st.session_state.current_qa_session = None
        st.session_state.recommendations = []
        st.session_state.learning_insights = None


# Initialize
init_session_state()


# Custom CSS - use the same CSS from the second code block
def load_custom_css():
    """Load custom CSS styles"""
    custom_css = """
    <style>
    /* Main styles */
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .section-header {
        font-size: 2rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
    }

    .subsection-header {
        font-size: 1.5rem;
        color: #4F46E5;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    /* Cards */
    .cloud-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }

    .cloud-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .success-card {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10B981;
    }

    .warning-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #F59E0B;
    }

    .error-card {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid #EF4444;
    }

    /* Code boxes */
    .code-box {
        background-color: #1E293B;
        color: #E2E8F0;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
        border: 1px solid #475569;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
    }

    section[data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
        width: 100%;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #F1F5F9;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: #94A3B8;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #64748B;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


load_custom_css()

# Sidebar - Use the sidebar code from the SECOND (shorter) code block
with st.sidebar:
    st.title("☁️ AI Lesson Studio")
    st.markdown("---")

    # User Profile
    with st.expander("👤 User Profile", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("🎓")
        with col2:
            st.session_state.current_user = st.selectbox(
                "Select Learner",
                ["Student", "Instructor", "Administrator"],
                key="user_role"
            )

    # Textbook Management
    st.subheader("📚 Textbook Management")

    uploaded_file = st.file_uploader(
        "Upload Cloud Computing Textbook",
        type=['pdf', 'txt', 'docx'],
        help="Upload your cloud computing textbook (PDF, DOCX, or TXT)"
    )

    if uploaded_file is not None:
        # Save uploaded file
        os.makedirs(config.TEXTBOOKS_DIR, exist_ok=True)
        file_path = config.TEXTBOOKS_DIR / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process textbook
        with st.spinner("📖 Processing textbook with AI..."):
            try:
                # Initialize processor
                processor = AdvancedTextbookProcessor(config)

                # Load textbook
                success = processor.load_textbook(
                    str(file_path),
                    textbook_name=uploaded_file.name
                )

                if success:
                    st.session_state.textbook_processor = processor
                    st.session_state.textbook_loaded = True

                    # Initialize AI engine with processor
                    st.session_state.ai_engine = CompleteAIEngine(config)

                    st.success(f"✅ Textbook loaded successfully!")

                    # Show textbook info
                    with st.expander("📊 Textbook Analysis"):
                        st.write(f"**Chapters:** {len(processor.chapters)}")
                        st.write(f"**Concepts:** {len(processor.concepts)}")
                        st.write(f"**Knowledge Graph:** {processor.knowledge_graph.number_of_nodes()} nodes")

                else:
                    st.error("❌ Failed to process textbook")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.markdown("---")

    # Navigation
    st.subheader("🧭 Navigation")
    page = st.radio(
        "Go to:",
        [
            "🏠 Dashboard",
            "📖 Textbook Explorer",
            "🎓 AI Lesson Generator",
            "💻 Cloud Simulator",
            "🖼️ Interactive Diagrams",
            "📝 Assessments & Quizzes",
            "📊 Learning Analytics",
            "⚙️ System Settings"
        ],
        key="navigation"
    )

    st.markdown("---")

    # System Status
    st.subheader("⚙️ System Status")

    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if st.session_state.textbook_loaded:
            st.success("📚 Loaded")
        else:
            st.warning("📚 Not Loaded")

        if st.session_state.ai_engine:
            st.success("🤖 AI Ready")
        else:
            st.warning("🤖 AI Offline")

    with status_col2:
        st.info("💾 Local")
        st.info("🆓 Free")

    # Quick Actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")

    if st.button("🔄 Clear Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if st.button("📥 Export Data"):
        st.info("Export functionality coming soon!")

# Main content based on selected page
if page == "🏠 Dashboard":
    # Dashboard Header
    st.markdown('<div class="main-header">AI Lesson Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Complete Cloud Computing Education Platform</div>', unsafe_allow_html=True)

    # Dashboard Stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.markdown("### 📚 Textbooks")
        st.markdown("### 0" if not st.session_state.textbook_loaded else f"### 1")
        st.markdown("Loaded")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.markdown("### 🎓 Lessons")
        lessons_count = len(st.session_state.lessons_generated) if hasattr(st.session_state, 'lessons_generated') else 0
        st.markdown(f"### {lessons_count}")
        st.markdown("Generated")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.markdown("### 💻 Labs")
        st.markdown("### 5+")
        st.markdown("Available")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Progress")
        progress = st.session_state.progress_tracker.get_overall_progress() if st.session_state.progress_tracker else 0
        st.markdown(f"### {progress:.0f}%")
        st.markdown("Completed")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Quick Start Section
    st.markdown('<div class="subsection-header">🚀 Quick Start</div>', unsafe_allow_html=True)

    quick_col1, quick_col2 = st.columns(2)

    with quick_col1:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.markdown("### 1. Upload Textbook")
        st.markdown("""
        • PDF, DOCX, or TXT format
        • Any cloud computing textbook
        • AI-powered processing
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.markdown("### 3. Practice with Labs")
        st.markdown("""
        • Simulated AWS/Azure/GCP
        • No cost, no risk
        • Realistic scenarios
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with quick_col2:
        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.markdown("### 2. Generate Lessons")
        st.markdown("""
        • AI-powered content
        • Personalized learning
        • Interactive diagrams
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cloud-card">', unsafe_allow_html=True)
        st.markdown("### 4. Track Progress")
        st.markdown("""
        • Learning analytics
        • Adaptive difficulty
        • Achievement tracking
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Feature Highlights
    st.markdown('<div class="subsection-header">✨ Key Features</div>', unsafe_allow_html=True)

    features_tabs = st.tabs([
        "🤖 AI-Powered",
        "☁️ Cloud Simulation",
        "📊 Analytics",
        "🎨 Interactive"
    ])

    with features_tabs[0]:
        st.markdown("""
        ### Advanced AI Integration
        • **HuggingFace Transformers** for intelligent Q&A
        • **spaCy NLP** for textbook analysis
        • **GPT-2/T5** for content generation
        • **BERT** for concept extraction
        • **Intelligent Q&A System** for personalized queries
        • **Lesson Generator** for adaptive content
        """)

    with features_tabs[1]:
        st.markdown("""
        ### Complete Cloud Simulation
        • **AWS/Azure/GCP** simulation
        • **Realistic CLI/API** experience
        • **Cost tracking** and monitoring
        • **Error simulation** for learning
        """)

    with features_tabs[2]:
        st.markdown("""
        ### Learning Analytics
        • **Progress tracking** per concept
        • **Adaptive learning paths**
        • **Performance insights**
        • **Recommendation engine**
        • **Insights generator** for detailed analysis
        """)

    with features_tabs[3]:
        st.markdown("""
        ### Interactive Content
        • **Mermaid.js diagrams**
        • **Plotly visualizations**
        • **Code execution sandbox**
        • **Real-time feedback**
        • **Interactive assessments**
        """)

    # Recent Activity (if available)
    if st.session_state.lessons_generated:
        st.markdown("---")
        st.markdown('<div class="subsection-header">📝 Recent Activity</div>', unsafe_allow_html=True)

        for i, lesson in enumerate(st.session_state.lessons_generated[-3:]):  # Show last 3 lessons
            with st.expander(f"🎓 Lesson {i + 1}: {lesson.get('title', 'Untitled')}"):
                st.write(lesson.get('summary', 'No summary available'))

    # Get Started Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Get Started with Cloud Computing", use_container_width=True):
            st.session_state.navigation = "📖 Textbook Explorer"
            st.rerun()

elif page == "📖 Textbook Explorer":
    st.markdown('<div class="main-header">Textbook Explorer</div>', unsafe_allow_html=True)

    if not st.session_state.textbook_loaded:
        st.warning("Please upload a textbook first from the sidebar.")

        with st.expander("Sample Cloud Computing Topics"):
            st.markdown("""
                ### Core Cloud Concepts
                - **IaaS, PaaS, SaaS**
                - **Public, Private, Hybrid Cloud**
                - **Virtualization & Containerization**

                ### Cloud Providers
                - **AWS Services** (EC2, S3, Lambda)
                - **Azure Services** (VM, Blob Storage, Functions)
                - **GCP Services** (Compute Engine, Cloud Storage)

                ### Advanced Topics
                - **Serverless Computing**
                - **Microservices Architecture**
                - **Cloud Security & Compliance**
                """)
        # Remove the return statement or use st.stop()
        st.stop()  # This will stop execution for this page
    else:
        processor = st.session_state.textbook_processor

        # Textbook Overview
        st.markdown('<div class="section-header">Textbook Overview</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Chapters", len(processor.chapters))
        with col2:
            st.metric("Concepts", len(processor.concepts))
        with col3:
            st.metric("Knowledge Nodes", processor.knowledge_graph.number_of_nodes())

    # Chapter Explorer
    st.markdown('<div class="section-header">Chapter Explorer</div>', unsafe_allow_html=True)

    chapters_tab, concepts_tab, graph_tab = st.tabs(["📑 Chapters", "🔑 Key Concepts", "🧠 Knowledge Graph"])

    with chapters_tab:
        selected_chapter = st.selectbox(
            "Select a Chapter",
            list(processor.chapters.keys()),
            key="chapter_select"
        )

        if selected_chapter:
            chapter_data = processor.chapters[selected_chapter]

            st.markdown(f"### {selected_chapter}")
            st.write(chapter_data.get('summary', 'No summary available'))

            # Show key concepts in this chapter
            if 'concepts' in chapter_data:
                st.markdown("#### Key Concepts:")
                for concept in chapter_data['concepts'][:10]:  # Show first 10
                    st.markdown(f"- {concept}")

    with concepts_tab:
        search_term = st.text_input("🔍 Search Concepts", placeholder="e.g., virtualization, serverless, containers")

        all_concepts = list(processor.concepts.keys())

        if search_term:
            filtered_concepts = [c for c in all_concepts if search_term.lower() in c.lower()]
        else:
            filtered_concepts = all_concepts[:50]  # Show first 50

        for concept in filtered_concepts:
            with st.expander(f"**{concept}**"):
                concept_data = processor.concepts[concept]
                st.write(concept_data.get('description', 'No description available'))

                if 'related_concepts' in concept_data:
                    st.markdown("**Related Concepts:**")
                    for related in concept_data['related_concepts'][:5]:
                        st.markdown(f"- {related}")

    with graph_tab:
        st.markdown("### Interactive Knowledge Graph")
        st.info("The knowledge graph shows relationships between concepts in the textbook.")

        # Simple graph visualization
        if hasattr(processor, 'knowledge_graph'):
            graph_info = {
                "Nodes": processor.knowledge_graph.number_of_nodes(),
                "Edges": processor.knowledge_graph.number_of_edges(),
                "Density": f"{processor.knowledge_graph.number_of_edges() / (processor.knowledge_graph.number_of_nodes() * (processor.knowledge_graph.number_of_nodes() - 1)):.4f}"
            }

            for key, value in graph_info.items():
                col1, col2 = st.columns([1, 3])
                col1.write(f"**{key}:**")
                col2.write(value)

            # Show sample of nodes
            st.markdown("#### Sample Concepts in Graph:")
            sample_nodes = list(processor.knowledge_graph.nodes())[:20]
            for node in sample_nodes:
                st.markdown(f"- {node}")

elif page == "🎓 AI Lesson Generator":
    st.markdown('<div class="main-header">🎓 AI Lesson Generator</div>', unsafe_allow_html=True)

    if not st.session_state.textbook_loaded:
        st.warning("📚 Please upload a textbook first from the sidebar.")
        st.stop()  # Use st.stop() instead of return
    else:
        lesson_tab, qa_tab, analyze_tab = st.tabs(["🎯 Generate Lessons", "❓ Intelligent Q&A", "📊 Content Analysis"])

    with lesson_tab:
        st.markdown('<div class="section-header">Generate AI-Powered Lessons</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            topic = st.text_input(
                "Lesson Topic",
                placeholder="e.g., AWS Lambda, Docker Containers, Cloud Security",
                key="lesson_topic"
            )

        with col2:
            difficulty = st.selectbox(
                "Difficulty Level",
                ["Beginner", "Intermediate", "Advanced"],
                key="lesson_difficulty"
            )

        # Lesson settings
        settings_col1, settings_col2, settings_col3 = st.columns(3)

        with settings_col1:
            include_diagrams = st.checkbox("Include Diagrams", value=True)

        with settings_col2:
            include_code = st.checkbox("Include Code Examples", value=True)

        with settings_col3:
            include_quiz = st.checkbox("Include Quiz Questions", value=True)

        if st.button("✨ Generate Lesson", type="primary", use_container_width=True):
            if not topic:
                st.error("Please enter a topic for the lesson.")
            else:
                with st.spinner("🤖 Generating AI-powered lesson..."):
                    try:
                        # Generate lesson using the LessonGenerator
                        lesson = st.session_state.lesson_generator.generate_lesson(
                            topic=topic,
                            difficulty=difficulty,
                            include_diagrams=include_diagrams,
                            include_code=include_code,
                            include_quiz=include_quiz,
                            textbook_processor=st.session_state.textbook_processor
                        )

                        # Store lesson in session state
                        if 'lessons_generated' not in st.session_state:
                            st.session_state.lessons_generated = []
                        st.session_state.lessons_generated.append(lesson)
                        st.session_state.current_lesson = lesson

                        st.success("✅ Lesson generated successfully!")

                        # Display the lesson
                        st.markdown("---")
                        st.markdown(f'<div class="section-header">{lesson.get("title", "Generated Lesson")}</div>',
                                    unsafe_allow_html=True)

                        # Learning objectives
                        if 'objectives' in lesson:
                            st.markdown("### 📋 Learning Objectives")
                            for obj in lesson['objectives']:
                                st.markdown(f"- {obj}")

                        # Content
                        if 'content' in lesson:
                            st.markdown("### 📖 Lesson Content")
                            st.write(lesson['content'])

                        # Diagrams
                        if include_diagrams and 'diagrams' in lesson and lesson['diagrams']:
                            st.markdown("### 🖼️ Diagrams")
                            for diagram in lesson['diagrams']:
                                st.code(diagram, language="mermaid")

                        # Code examples
                        if include_code and 'code_examples' in lesson and lesson['code_examples']:
                            st.markdown("### 💻 Code Examples")
                            for code in lesson['code_examples']:
                                st.code(code, language="python")

                        # Quiz
                        if include_quiz and 'quiz_questions' in lesson and lesson['quiz_questions']:
                            st.markdown("### 📝 Practice Questions")
                            for i, question in enumerate(lesson['quiz_questions'], 1):
                                with st.expander(f"Question {i}"):
                                    st.write(question.get('question', ''))
                                    if 'options' in question:
                                        for opt in question['options']:
                                            st.write(f"- {opt}")

                    except Exception as e:
                        st.error(f"Error generating lesson: {str(e)}")

    with qa_tab:
        st.markdown('<div class="section-header">Intelligent Q&A System</div>', unsafe_allow_html=True)

        question = st.text_area(
            "Ask a question about cloud computing:",
            placeholder="e.g., What is the difference between IaaS and PaaS? How does serverless computing work?",
            height=100
        )

        if st.button("🤖 Get Answer", key="qa_button"):
            if question:
                with st.spinner("Thinking..."):
                    try:
                        answer = st.session_state.qa_system.answer_question(
                            question=question,
                            context=st.session_state.textbook_processor.get_context_for_question(
                                question) if st.session_state.textbook_loaded else None
                        )

                        st.markdown('<div class="cloud-card success-card">', unsafe_allow_html=True)
                        st.markdown("### Answer:")
                        st.write(answer)
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Store QA session
                        if 'qa_session' not in st.session_state:
                            st.session_state.qa_session = []
                        st.session_state.qa_session.append({
                            "question": question,
                            "answer": answer,
                            "timestamp": datetime.now()
                        })

                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")
            else:
                st.warning("Please enter a question.")

    with analyze_tab:
        st.markdown('<div class="section-header">Content Analysis</div>', unsafe_allow_html=True)

        content_to_analyze = st.text_area(
            "Enter content to analyze:",
            placeholder="Paste cloud computing content here...",
            height=200
        )

        if st.button("🔍 Analyze Content", key="analyze_button"):
            if content_to_analyze:
                with st.spinner("Analyzing content..."):
                    try:
                        analysis = st.session_state.content_analyzer.analyze_content(content_to_analyze)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### 📊 Analysis Results")
                            if 'keywords' in analysis:
                                st.markdown("**Key Concepts:**")
                                for keyword in analysis['keywords'][:10]:
                                    st.markdown(f"- {keyword}")

                        with col2:
                            if 'complexity' in analysis:
                                st.metric("Complexity Score", f"{analysis['complexity']:.2f}/10")

                            if 'recommendations' in analysis:
                                st.markdown("**Recommendations:**")
                                for rec in analysis['recommendations']:
                                    st.markdown(f"- {rec}")

                    except Exception as e:
                        st.error(f"Error analyzing content: {str(e)}")
            else:
                st.warning("Please enter content to analyze.")

elif page == "💻 Cloud Simulator":
    st.markdown('<div class="main-header">💻 Cloud Simulator</div>', unsafe_allow_html=True)

    if 'cloud_simulator' not in st.session_state or st.session_state.cloud_simulator is None:
        st.session_state.cloud_simulator = CloudSimulator(config)

    simulator = st.session_state.cloud_simulator

    # Cloud Provider Selection
    st.markdown('<div class="section-header">Cloud Environment Setup</div>', unsafe_allow_html=True)

    provider_col1, provider_col2, provider_col3 = st.columns(3)

    with provider_col1:
        provider = st.selectbox(
            "Cloud Provider",
            ["AWS", "Azure", "GCP", "Multi-Cloud"],
            key="cloud_provider"
        )

    with provider_col2:
        region = st.selectbox(
            "Region",
            ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
            key="cloud_region"
        )

    with provider_col3:
        environment = st.selectbox(
            "Environment Type",
            ["Development", "Testing", "Production", "Learning"],
            key="cloud_env"
        )

    # Simulation Tabs
    sim_tabs = st.tabs(["🚀 Launch Resources", "📁 Storage Simulator", "⚡ Compute Simulator", "🔧 Advanced Tools"])

    with sim_tabs[0]:
        st.markdown("### Launch Cloud Resources")

        resource_type = st.selectbox(
            "Resource Type",
            ["EC2 Instance (AWS)", "VM Instance (Azure)", "Compute Engine (GCP)", "Lambda Function",
             "Container Service"],
            key="resource_type"
        )

        config_col1, config_col2 = st.columns(2)

        with config_col1:
            instance_size = st.selectbox(
                "Instance Size",
                ["t2.micro", "t2.small", "t2.medium", "m5.large"],
                key="instance_size"
            )

        with config_col2:
            operating_system = st.selectbox(
                "OS",
                ["Amazon Linux", "Ubuntu", "Windows Server", "RHEL"],
                key="instance_os"
            )

        if st.button("🚀 Launch Instance", type="primary"):
            with st.spinner(f"Launching {resource_type}..."):
                try:
                    result = simulator.launch_instance(
                        provider=provider,
                        instance_type=instance_size,
                        os_type=operating_system,
                        region=region
                    )

                    st.success(f"✅ {resource_type} launched successfully!")

                    # Show instance details
                    with st.expander("Instance Details", expanded=True):
                        st.json(result)

                    # Store in session state
                    if 'simulated_resources' not in st.session_state:
                        st.session_state.simulated_resources = []
                    st.session_state.simulated_resources.append(result)

                except Exception as e:
                    st.error(f"Error launching instance: {str(e)}")

    with sim_tabs[1]:
        st.markdown("### Cloud Storage Simulator")

        storage_type = st.selectbox(
            "Storage Type",
            ["S3 Bucket (AWS)", "Blob Storage (Azure)", "Cloud Storage (GCP)", "EFS", "RDS Database"],
            key="storage_type"
        )

        bucket_name = st.text_input("Storage Name", "my-cloud-bucket")

        if st.button("📁 Create Storage", type="primary"):
            with st.spinner(f"Creating {storage_type}..."):
                try:
                    result = simulator.create_storage(
                        storage_type=storage_type,
                        name=bucket_name,
                        region=region
                    )

                    st.success(f"✅ {storage_type} created successfully!")

                    # File upload simulation
                    uploaded_file = st.file_uploader("Upload a test file", type=['txt', 'json', 'csv'])

                    if uploaded_file is not None:
                        file_content = uploaded_file.read()
                        st.info(f"File '{uploaded_file.name}' ready for upload to {bucket_name}")

                        if st.button("📤 Upload File"):
                            with st.spinner("Uploading file..."):
                                upload_result = simulator.upload_to_storage(
                                    storage_name=bucket_name,
                                    filename=uploaded_file.name,
                                    content=file_content
                                )
                                st.success(f"✅ File uploaded successfully! Key: {upload_result.get('key')}")

                except Exception as e:
                    st.error(f"Error creating storage: {str(e)}")

    with sim_tabs[2]:
        st.markdown("### Compute Operations")

        operation = st.selectbox(
            "Operation",
            ["Run Serverless Function", "Deploy Container", "Scale Resources", "Monitor Performance"],
            key="compute_op"
        )

        if operation == "Run Serverless Function":
            function_code = st.text_area(
                "Function Code (Python)",
                """def lambda_handler(event, context):
    # Your serverless function code here
    print("Hello from Lambda!")
    return {
        'statusCode': 200,
        'body': 'Function executed successfully'
    }""",
                height=200
            )

            if st.button("⚡ Execute Function"):
                with st.spinner("Executing function..."):
                    try:
                        result = simulator.execute_function(
                            code=function_code,
                            runtime="python3.9"
                        )

                        st.success("✅ Function executed!")

                        with st.expander("Execution Results", expanded=True):
                            st.code(result.get('output', ''), language="python")
                            st.metric("Execution Time", f"{result.get('duration', 0):.3f}s")
                            st.metric("Memory Used", f"{result.get('memory_used', 0)} MB")

                    except Exception as e:
                        st.error(f"Error executing function: {str(e)}")

    with sim_tabs[3]:
        st.markdown("### Advanced Cloud Tools")

        tool = st.selectbox(
            "Select Tool",
            ["Cost Calculator", "Security Scanner", "Network Configurator", "Auto Scaling"],
            key="cloud_tool"
        )

        if tool == "Cost Calculator":
            st.markdown("#### Monthly Cost Estimation")

            col1, col2, col3 = st.columns(3)

            with col1:
                instances = st.number_input("Number of Instances", min_value=1, max_value=100, value=2)

            with col2:
                hours_per_day = st.number_input("Hours/Day", min_value=1, max_value=24, value=12)

            with col3:
                storage_gb = st.number_input("Storage (GB)", min_value=1, max_value=10000, value=100)

            if st.button("💰 Calculate Cost"):
                monthly_cost = instances * hours_per_day * 30 * 0.0116  # Example rate
                storage_cost = storage_gb * 0.023  # Example storage rate
                total_cost = monthly_cost + storage_cost

                st.markdown(f"""
                ### Estimated Monthly Cost: ${total_cost:.2f}

                **Breakdown:**
                - Compute: ${monthly_cost:.2f}
                - Storage: ${storage_cost:.2f}
                - Data Transfer: $5.00 (estimated)

                *Based on {provider} pricing in {region}*
                """)

        # Active Resources Panel
        st.markdown("---")
        st.markdown("### Active Resources")

        if 'simulated_resources' in st.session_state and st.session_state.simulated_resources:
            for i, resource in enumerate(st.session_state.simulated_resources):
                with st.expander(f"Resource {i + 1}: {resource.get('type', 'Unknown')}"):
                    st.json(resource)
        else:
            st.info("No active resources. Launch resources from the 'Launch Resources' tab.")

elif page == "🖼️ Interactive Diagrams":
    st.markdown('<div class="main-header">🖼️ Interactive Diagrams</div>', unsafe_allow_html=True)

    if 'diagram_generator' not in st.session_state:
        st.session_state.diagram_generator = DiagramGenerator(config)

    diagram_gen = st.session_state.diagram_generator

    diagram_tabs = st.tabs(["📊 Architecture Diagrams", "🔗 Flow Charts", "📈 Data Visualizations", "🎨 Custom Diagrams"])

    with diagram_tabs[0]:
        st.markdown("### Cloud Architecture Diagrams")

        architecture_type = st.selectbox(
            "Architecture Type",
            [
                "3-Tier Web Application",
                "Microservices Architecture",
                "Serverless Backend",
                "Data Lake Architecture",
                "High Availability Setup"
            ],
            key="arch_type"
        )

        provider = st.selectbox(
            "Cloud Provider",
            ["AWS", "Azure", "GCP"],
            key="diagram_provider"
        )

        if st.button("🖼️ Generate Architecture Diagram", type="primary"):
            with st.spinner("Generating diagram..."):
                try:
                    diagram_code = diagram_gen.generate_cloud_architecture(
                        arch_type=architecture_type,
                        provider=provider
                    )

                    st.success("✅ Diagram generated!")

                    # Display the Mermaid diagram
                    st.markdown("### Diagram Preview")
                    st.code(diagram_code, language="mermaid")

                    # Save diagram to session state
                    diagram_key = f"{architecture_type}_{provider}"
                    st.session_state.diagrams_generated[diagram_key] = diagram_code

                    # Explanation
                    with st.expander("📖 Architecture Explanation"):
                        explanation = diagram_gen.explain_architecture(diagram_code)
                        st.write(explanation)

                except Exception as e:
                    st.error(f"Error generating diagram: {str(e)}")

    with diagram_tabs[1]:
        st.markdown("### Process Flow Charts")

        process_type = st.selectbox(
            "Process Type",
            [
                "API Request Flow",
                "CI/CD Pipeline",
                "Data Processing Pipeline",
                "Authentication Flow",
                "Disaster Recovery"
            ],
            key="process_type"
        )

        if st.button("🔗 Generate Flow Chart", type="primary"):
            with st.spinner("Generating flow chart..."):
                try:
                    flow_chart = diagram_gen.generate_flow_chart(process_type)

                    st.success("✅ Flow chart generated!")
                    st.code(flow_chart, language="mermaid")

                except Exception as e:
                    st.error(f"Error generating flow chart: {str(e)}")

    with diagram_tabs[2]:
        st.markdown("### Data Visualizations")

        # Generate sample cloud data for visualization
        if st.button("📈 Generate Sample Visualization", key="viz_button"):
            try:
                # Create sample data
                import pandas as pd
                import numpy as np

                dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
                data = pd.DataFrame({
                    'Date': dates,
                    'AWS_Usage': np.random.randint(50, 200, len(dates)),
                    'Azure_Usage': np.random.randint(30, 150, len(dates)),
                    'GCP_Usage': np.random.randint(20, 100, len(dates)),
                    'Cost_USD': np.random.uniform(100, 500, len(dates)).cumsum()
                })

                # Create tabs for different visualizations
                viz_tabs = st.tabs(["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot"])

                with viz_tabs[0]:
                    fig = px.line(data, x='Date', y=['AWS_Usage', 'Azure_Usage', 'GCP_Usage'],
                                  title='Cloud Service Usage Over Time',
                                  labels={'value': 'Usage Units', 'variable': 'Cloud Provider'})
                    st.plotly_chart(fig, use_container_width=True)

                with viz_tabs[1]:
                    monthly_data = data.copy()
                    monthly_data['Month'] = monthly_data['Date'].dt.month
                    monthly_agg = monthly_data.groupby('Month')[
                        ['AWS_Usage', 'Azure_Usage', 'GCP_Usage']].mean().reset_index()

                    fig = px.bar(monthly_agg, x='Month', y=['AWS_Usage', 'Azure_Usage', 'GCP_Usage'],
                                 title='Average Monthly Usage by Provider',
                                 barmode='group')
                    st.plotly_chart(fig, use_container_width=True)

                with viz_tabs[2]:
                    fig = px.area(data, x='Date', y='Cost_USD',
                                  title='Cumulative Cloud Costs',
                                  labels={'Cost_USD': 'Cost (USD)'})
                    st.plotly_chart(fig, use_container_width=True)

                with viz_tabs[3]:
                    fig = px.scatter(data, x='AWS_Usage', y='Cost_USD',
                                     title='AWS Usage vs Cost Correlation',
                                     trendline='ols')
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")

    with diagram_tabs[3]:
        st.markdown("### Custom Diagram Builder")

        col1, col2 = st.columns([2, 1])

        with col1:
            diagram_title = st.text_input("Diagram Title", "My Cloud Architecture")

            # Node management
            st.subheader("➕ Add Nodes")
            node_name = st.text_input("Node Name", "Load Balancer")
            node_type = st.selectbox("Node Type", ["Database", "Server", "Service", "Storage", "Network"])

            if st.button("Add Node", key="add_node"):
                if node_name:
                    st.session_state.custom_nodes.append({
                        "name": node_name,
                        "type": node_type
                    })
                    st.success(f"Added node: {node_name}")

        with col2:
            st.subheader("Current Nodes")
            if st.session_state.custom_nodes:
                for i, node in enumerate(st.session_state.custom_nodes):
                    st.write(f"{i + 1}. {node['name']} ({node['type']})")
            else:
                st.info("No nodes added yet")

        # Edge management
        st.markdown("---")
        st.subheader("🔗 Connect Nodes")

        if len(st.session_state.custom_nodes) >= 2:
            col1, col2, col3 = st.columns(3)

            with col1:
                from_node = st.selectbox("From Node",
                                         [n["name"] for n in st.session_state.custom_nodes],
                                         key="from_node")

            with col2:
                to_node = st.selectbox("To Node",
                                       [n["name"] for n in st.session_state.custom_nodes],
                                       key="to_node")

            with col3:
                connection_type = st.selectbox("Connection Type",
                                               ["HTTP", "Database", "Message", "Sync", "Async"],
                                               key="conn_type")

            if st.button("Add Connection", key="add_connection"):
                st.session_state.custom_edges.append({
                    "from": from_node,
                    "to": to_node,
                    "type": connection_type
                })
                st.success(f"Connected {from_node} → {to_node}")

        # Generate custom diagram
        st.markdown("---")
        if st.button("🎨 Generate Custom Diagram", type="primary"):
            if st.session_state.custom_nodes:
                try:
                    custom_diagram = diagram_gen.generate_custom_diagram(
                        nodes=st.session_state.custom_nodes,
                        edges=st.session_state.custom_edges,
                        title=diagram_title
                    )

                    st.success("✅ Custom diagram generated!")
                    st.code(custom_diagram, language="mermaid")

                except Exception as e:
                    st.error(f"Error generating custom diagram: {str(e)}")
            else:
                st.warning("Please add at least one node to generate a diagram.")

elif page == "📝 Assessments & Quizzes":
    st.markdown('<div class="main-header">📝 Assessments & Quizzes</div>', unsafe_allow_html=True)

    assessment_tabs = st.tabs(["🎯 Generate Quiz", "📊 Take Assessment", "📝 Practice Questions", "🔍 Question Bank"])

    with assessment_tabs[0]:
        st.markdown('<div class="section-header">Generate AI-Powered Quiz</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            quiz_topic = st.text_input(
                "Quiz Topic",
                placeholder="e.g., AWS S3, Cloud Security, Docker Basics",
                key="quiz_topic"
            )

            num_questions = st.slider(
                "Number of Questions",
                min_value=5,
                max_value=20,
                value=10,
                key="num_questions"
            )

        with col2:
            question_types = st.multiselect(
                "Question Types",
                ["Multiple Choice", "True/False", "Fill in Blanks", "Matching", "Scenario-based"],
                default=["Multiple Choice", "True/False"],
                key="q_types"
            )

            quiz_difficulty = st.select_slider(
                "Difficulty",
                options=["Easy", "Medium", "Hard", "Expert"],
                value="Medium",
                key="quiz_diff"
            )

        if st.button("🤖 Generate Quiz", type="primary", use_container_width=True):
            if quiz_topic:
                with st.spinner("Generating quiz questions..."):
                    try:
                        # Generate quiz using QuizGenerator
                        quiz = st.session_state.quiz_generator.generate_quiz(
                            topic=quiz_topic,
                            num_questions=num_questions,
                            difficulty=quiz_difficulty,
                            question_types=question_types,
                            textbook_processor=st.session_state.textbook_processor if st.session_state.textbook_loaded else None
                        )

                        st.session_state.current_quiz = quiz
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False

                        st.success(f"✅ Generated {len(quiz.get('questions', []))} questions!")

                        # Auto-navigate to Take Assessment tab
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error generating quiz: {str(e)}")
            else:
                st.warning("Please enter a quiz topic.")

    with assessment_tabs[1]:
        st.markdown('<div class="section-header">Take Assessment</div>', unsafe_allow_html=True)

        if not st.session_state.current_quiz:
            st.info("🎯 No active quiz. Generate one from the 'Generate Quiz' tab first.")
        else:
            quiz = st.session_state.current_quiz

            st.markdown(f"### 📝 {quiz.get('title', 'Cloud Computing Quiz')}")
            st.markdown(f"**Topic:** {quiz.get('topic', 'General Cloud Computing')}")
            st.markdown(f"**Difficulty:** {quiz.get('difficulty', 'Medium')}")
            st.markdown(f"**Time Estimate:** {quiz.get('estimated_time', '15')} minutes")

            st.markdown("---")

            # Display questions
            for i, question in enumerate(quiz.get('questions', [])):
                with st.container():
                    st.markdown(f"#### Question {i + 1}: {question.get('question_text', '')}")

                    question_type = question.get('type', 'multiple_choice')

                    if question_type == 'multiple_choice':
                        options = question.get('options', [])
                        for j, option in enumerate(options):
                            key = f"q{i}_option{j}"
                            if key not in st.session_state.quiz_answers:
                                st.session_state.quiz_answers[key] = False

                            st.session_state.quiz_answers[key] = st.radio(
                                f"Select answer for Q{i + 1}:",
                                options,
                                key=f"q{i}_radio",
                                index=None if key not in st.session_state.quiz_answers else options.index(
                                    st.session_state.quiz_answers[key]) if st.session_state.quiz_answers[
                                                                               key] in options else None
                            )

                    elif question_type == 'true_false':
                        answer = st.radio(
                            f"True or False for Q{i + 1}:",
                            ["True", "False"],
                            key=f"q{i}_tf",
                            index=None
                        )
                        st.session_state.quiz_answers[f"q{i}_tf"] = answer

                    elif question_type == 'fill_blank':
                        answer = st.text_input(
                            f"Fill in the blank for Q{i + 1}:",
                            key=f"q{i}_fill"
                        )
                        st.session_state.quiz_answers[f"q{i}_fill"] = answer

                    st.markdown("---")

            # Submit button
            if not st.session_state.quiz_submitted:
                if st.button("📤 Submit Quiz", type="primary", use_container_width=True):
                    st.session_state.quiz_submitted = True

                    # Evaluate quiz
                    evaluator = AssessmentEvaluator(config)
                    result = evaluator.evaluate_quiz(
                        quiz=quiz,
                        answers=st.session_state.quiz_answers
                    )

                    st.session_state.quiz_result = result
                    st.success("✅ Quiz submitted! Check your results below.")
                    st.rerun()

            # Show results if submitted
            if st.session_state.quiz_submitted and st.session_state.quiz_result:
                result = st.session_state.quiz_result

                st.markdown('<div class="section-header">📊 Quiz Results</div>', unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Score", f"{result.get('score', 0)}%")

                with col2:
                    correct = result.get('correct_answers', 0)
                    total = result.get('total_questions', 1)
                    st.metric("Correct Answers", f"{correct}/{total}")

                with col3:
                    st.metric("Time Spent", result.get('time_spent', 'N/A'))

                # Detailed feedback
                st.markdown("### 📝 Detailed Feedback")

                if 'feedback' in result:
                    for i, fb in enumerate(result['feedback']):
                        with st.expander(f"Question {i + 1} Feedback"):
                            if fb.get('is_correct', False):
                                st.success("✅ Correct!")
                            else:
                                st.error("❌ Incorrect")

                            st.markdown(f"**Your answer:** {fb.get('user_answer', 'N/A')}")
                            st.markdown(f"**Correct answer:** {fb.get('correct_answer', 'N/A')}")

                            if 'explanation' in fb:
                                st.markdown(f"**Explanation:** {fb['explanation']}")

                # Recommendations
                if 'recommendations' in result:
                    st.markdown("### 🎯 Recommendations")
                    for rec in result['recommendations']:
                        st.markdown(f"- {rec}")

                # Store in progress tracker
                if st.session_state.progress_tracker:
                    st.session_state.progress_tracker.add_quiz_result(
                        quiz_topic=quiz.get('topic', 'Unknown'),
                        score=result.get('score', 0),
                        details=result
                    )

    with assessment_tabs[2]:
        st.markdown('<div class="section-header">Practice Questions</div>', unsafe_allow_html=True)

        # Quick practice by topic
        practice_topics = [
            "Cloud Fundamentals",
            "AWS Services",
            "Azure Services",
            "GCP Services",
            "Cloud Security",
            "Containers & Kubernetes",
            "Serverless Computing",
            "Cloud Networking"
        ]

        selected_topic = st.selectbox("Select Practice Topic", practice_topics, key="practice_topic")
        num_practice = st.slider("Number of Questions", 1, 10, 5, key="num_practice")

        if st.button("🎯 Start Practice Session", type="primary"):
            with st.spinner(f"Generating {num_practice} {selected_topic} questions..."):
                try:
                    practice_quiz = st.session_state.quiz_generator.generate_quiz(
                        topic=selected_topic,
                        num_questions=num_practice,
                        difficulty="Medium",
                        question_types=["Multiple Choice", "True/False"]
                    )

                    # Display practice questions
                    for i, question in enumerate(practice_quiz.get('questions', [])):
                        st.markdown(f"**Q{i + 1}:** {question.get('question_text', '')}")

                        if question.get('type') == 'multiple_choice':
                            options = question.get('options', [])
                            selected = st.radio(
                                f"Select answer for Q{i + 1}:",
                                options,
                                key=f"practice_q{i}",
                                index=None,
                                label_visibility="collapsed"
                            )

                        # Show answer button
                        if st.button(f"Show Answer {i + 1}", key=f"show_ans{i}"):
                            st.info(f"**Answer:** {question.get('correct_answer', 'Not available')}")
                            if 'explanation' in question:
                                st.write(f"**Explanation:** {question['explanation']}")

                        st.markdown("---")

                except Exception as e:
                    st.error(f"Error generating practice questions: {str(e)}")

    with assessment_tabs[3]:
        st.markdown('<div class="section-header">Question Bank</div>', unsafe_allow_html=True)

        # Search questions
        search_query = st.text_input("🔍 Search Questions", placeholder="Search by keyword...")

        # Filter by type and difficulty
        col1, col2 = st.columns(2)
        with col1:
            filter_type = st.multiselect(
                "Question Type",
                ["Multiple Choice", "True/False", "Fill in Blanks", "Scenario"],
                default=["Multiple Choice", "True/False"]
            )

        with col2:
            filter_difficulty = st.multiselect(
                "Difficulty",
                ["Easy", "Medium", "Hard"],
                default=["Easy", "Medium", "Hard"]
            )

        # Sample question bank (in practice, this would come from database)
        sample_questions = [
            {
                "id": 1,
                "question": "What does IaaS stand for in cloud computing?",
                "type": "Multiple Choice",
                "difficulty": "Easy",
                "topic": "Cloud Fundamentals"
            },
            {
                "id": 2,
                "question": "Which AWS service provides object storage?",
                "type": "Multiple Choice",
                "difficulty": "Easy",
                "topic": "AWS Services"
            },
            {
                "id": 3,
                "question": "True or False: Azure Functions is a serverless compute service.",
                "type": "True/False",
                "difficulty": "Easy",
                "topic": "Azure Services"
            },
            {
                "id": 4,
                "question": "What is the primary benefit of containerization?",
                "type": "Multiple Choice",
                "difficulty": "Medium",
                "topic": "Containers"
            }
        ]

        # Filter questions
        filtered_questions = sample_questions

        if search_query:
            filtered_questions = [q for q in filtered_questions if search_query.lower() in q['question'].lower()]

        if filter_type:
            filtered_questions = [q for q in filtered_questions if q['type'] in filter_type]

        if filter_difficulty:
            filtered_questions = [q for q in filtered_questions if q['difficulty'] in filter_difficulty]

        # Display filtered questions
        st.markdown(f"**Found {len(filtered_questions)} questions**")

        for q in filtered_questions:
            with st.expander(f"{q['topic']} - {q['type']} - {q['difficulty']}"):
                st.write(q['question'])

                # Option to add to custom quiz
                if st.button(f"Add to Custom Quiz", key=f"add_q{q['id']}"):
                    st.success(f"Added question {q['id']} to custom quiz")

elif page == "📊 Learning Analytics":
    st.markdown('<div class="main-header">📊 Learning Analytics</div>', unsafe_allow_html=True)

    analytics_tabs = st.tabs(
        ["📈 Progress Dashboard", "🎯 Recommendations", "🔍 Detailed Insights", "📊 Performance Metrics"])

    with analytics_tabs[0]:
        st.markdown('<div class="section-header">Learning Progress Dashboard</div>', unsafe_allow_html=True)

        if not st.session_state.progress_tracker:
            st.info("No progress data available yet. Complete some lessons or quizzes to see analytics.")
        else:
            progress_data = st.session_state.progress_tracker.get_progress_data()

            # Overall Progress
            col1, col2, col3 = st.columns(3)

            with col1:
                overall_progress = progress_data.get('overall_progress', 0)
                st.metric("Overall Progress", f"{overall_progress:.1f}%")
                st.progress(overall_progress / 100)

            with col2:
                lessons_completed = progress_data.get('lessons_completed', 0)
                st.metric("Lessons Completed", lessons_completed)

            with col3:
                avg_quiz_score = progress_data.get('average_quiz_score', 0)
                st.metric("Average Quiz Score", f"{avg_quiz_score:.1f}%")

            # Progress by Topic
            st.markdown("### 📊 Progress by Topic")

            if 'topic_progress' in progress_data:
                topic_data = progress_data['topic_progress']

                # Convert to DataFrame for visualization
                import pandas as pd

                topics = list(topic_data.keys())
                progress_values = [topic_data[topic].get('progress', 0) for topic in topics]

                # Create a bar chart
                import plotly.graph_objects as go

                fig = go.Figure(data=[
                    go.Bar(
                        x=topics,
                        y=progress_values,
                        text=[f"{v:.1f}%" for v in progress_values],
                        textposition='auto',
                        marker_color='#3B82F6'
                    )
                ])

                fig.update_layout(
                    title="Learning Progress by Topic",
                    xaxis_title="Topic",
                    yaxis_title="Progress (%)",
                    yaxis_range=[0, 100],
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            # Recent Activity
            st.markdown("### 📝 Recent Activity")

            if 'recent_activities' in progress_data and progress_data['recent_activities']:
                for activity in progress_data['recent_activities'][:5]:  # Show last 5
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        st.write(f"**{activity.get('type', 'Activity')}:** {activity.get('description', '')}")

                    with col2:
                        st.write(f"Score: {activity.get('score', 'N/A')}")

                    with col3:
                        st.write(f"Date: {activity.get('date', '')}")

                    st.markdown("---")
            else:
                st.info("No recent activities recorded.")

    with analytics_tabs[1]:
        st.markdown('<div class="section-header">🎯 Personalized Recommendations</div>', unsafe_allow_html=True)

        if not st.session_state.recommender:
            st.session_state.recommender = RecommenderSystem(config)

        # Get recommendations
        if st.session_state.progress_tracker:
            progress_data = st.session_state.progress_tracker.get_progress_data()
            recommendations = st.session_state.recommender.get_recommendations(
                user_id="current_user",
                progress_data=progress_data,
                textbook_processor=st.session_state.textbook_processor if st.session_state.textbook_loaded else None
            )

            st.session_state.recommendations = recommendations

            if recommendations:
                st.success(f"Found {len(recommendations)} recommendations for you!")

                # Display recommendations
                for i, rec in enumerate(recommendations):
                    with st.container():
                        st.markdown(f"### 🎯 Recommendation {i + 1}")

                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"**{rec.get('title', 'Study Recommendation')}**")
                            st.write(rec.get('description', ''))

                        with col2:
                            st.metric("Priority", rec.get('priority', 'Medium'))

                        # Action buttons
                        rec_type = rec.get('type', 'lesson')

                        if rec_type == 'lesson':
                            if st.button(f"📖 Study This Topic", key=f"rec_lesson_{i}"):
                                # Set topic for lesson generator
                                st.session_state.lesson_topic = rec.get('topic', 'Cloud Computing')
                                st.session_state.navigation = "🎓 AI Lesson Generator"
                                st.rerun()

                        elif rec_type == 'quiz':
                            if st.button(f"📝 Take Quiz", key=f"rec_quiz_{i}"):
                                # Generate quiz for this topic
                                st.session_state.quiz_topic = rec.get('topic', 'Cloud Computing')
                                st.rerun()

                        elif rec_type == 'practice':
                            if st.button(f"💻 Practice Exercise", key=f"rec_practice_{i}"):
                                st.info("Practice exercise will be generated for this topic.")

                        st.markdown("---")
            else:
                st.info("Complete some activities to get personalized recommendations.")

        # Manual recommendation request
        st.markdown("### 🔍 Request Specific Recommendations")

        request_topic = st.text_input("What area do you want to improve in?",
                                      placeholder="e.g., cloud security, AWS services, networking")

        if st.button("🤖 Get Custom Recommendations", key="custom_recs"):
            if request_topic:
                with st.spinner("Analyzing your request..."):
                    try:
                        custom_recs = st.session_state.recommender.get_topic_recommendations(
                            topic=request_topic,
                            textbook_processor=st.session_state.textbook_processor if st.session_state.textbook_loaded else None
                        )

                        if custom_recs:
                            st.success(f"Found {len(custom_recs)} resources for '{request_topic}'")

                            for rec in custom_recs:
                                with st.expander(rec.get('title', 'Resource')):
                                    st.write(rec.get('description', ''))
                                    st.markdown(f"**Type:** {rec.get('type', 'resource')}")
                                    st.markdown(f"**Estimated Time:** {rec.get('estimated_time', '30')} minutes")
                        else:
                            st.info(f"No specific resources found for '{request_topic}'. Try a broader topic.")

                    except Exception as e:
                        st.error(f"Error getting recommendations: {str(e)}")
            else:
                st.warning("Please enter a topic.")

    with analytics_tabs[2]:
        st.markdown('<div class="section-header">🔍 Detailed Learning Insights</div>', unsafe_allow_html=True)

        if not st.session_state.insights_generator:
            st.session_state.insights_generator = InsightsGenerator(config)

        if st.session_state.progress_tracker:
            progress_data = st.session_state.progress_tracker.get_progress_data()

            # Generate insights
            insights = st.session_state.insights_generator.generate_insights(progress_data)
            st.session_state.learning_insights = insights

            # Display insights
            if insights:
                # Strengths
                if 'strengths' in insights and insights['strengths']:
                    st.markdown("### 💪 Your Strengths")
                    for strength in insights['strengths']:
                        st.success(f"**{strength.get('area', 'Area')}:** {strength.get('description', '')}")
                        if 'evidence' in strength:
                            st.caption(f"Evidence: {strength['evidence']}")

                # Areas for improvement
                if 'improvement_areas' in insights and insights['improvement_areas']:
                    st.markdown("### 🎯 Areas for Improvement")
                    for area in insights['improvement_areas']:
                        st.warning(f"**{area.get('area', 'Area')}:** {area.get('description', '')}")
                        if 'suggestions' in area:
                            for suggestion in area['suggestions']:
                                st.markdown(f"- {suggestion}")

                # Learning patterns
                if 'learning_patterns' in insights and insights['learning_patterns']:
                    st.markdown("### 📈 Learning Patterns")

                    for pattern in insights['learning_patterns']:
                        with st.expander(pattern.get('title', 'Pattern')):
                            st.write(pattern.get('description', ''))

                            if 'data' in pattern:
                                # Create visualization for pattern data
                                try:
                                    import plotly.express as px
                                    import pandas as pd

                                    pattern_data = pattern['data']
                                    if isinstance(pattern_data, dict) and len(pattern_data) > 0:
                                        df = pd.DataFrame(list(pattern_data.items()), columns=['Metric', 'Value'])
                                        fig = px.bar(df, x='Metric', y='Value', title=pattern['title'])
                                        st.plotly_chart(fig, use_container_width=True)
                                except:
                                    pass

                # Generate report
                if st.button("📥 Generate Insights Report", key="insights_report"):
                    report = st.session_state.insights_generator.generate_report(insights)

                    st.download_button(
                        label="📥 Download Insights Report",
                        data=report,
                        file_name=f"learning_insights_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
            else:
                st.info("Complete more activities to generate detailed insights.")
        else:
            st.info("No progress data available yet.")

    with analytics_tabs[3]:
        st.markdown('<div class="section-header">📊 Performance Metrics</div>', unsafe_allow_html=True)

        if st.session_state.progress_tracker:
            progress_data = st.session_state.progress_tracker.get_progress_data()

            # Time-based metrics
            st.markdown("### ⏱️ Time-Based Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                total_time = progress_data.get('total_study_time', 0)
                st.metric("Total Study Time", f"{total_time} min")

            with col2:
                avg_session = progress_data.get('average_session_time', 0)
                st.metric("Avg Session Time", f"{avg_session:.1f} min")

            with col3:
                sessions = progress_data.get('study_sessions', 0)
                st.metric("Study Sessions", sessions)

            # Quiz performance over time
            st.markdown("### 📝 Quiz Performance Trend")

            if 'quiz_history' in progress_data and progress_data['quiz_history']:
                quiz_data = progress_data['quiz_history']

                # Create performance chart
                import plotly.graph_objects as go

                dates = [q.get('date', '') for q in quiz_data]
                scores = [q.get('score', 0) for q in quiz_data]
                topics = [q.get('topic', 'Quiz') for q in quiz_data]

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=dates,
                    y=scores,
                    mode='lines+markers',
                    name='Quiz Scores',
                    line=dict(color='#3B82F6', width=3),
                    marker=dict(size=8),
                    text=topics,
                    hoverinfo='text+y'
                ))

                # Add average line
                avg_score = sum(scores) / len(scores) if scores else 0
                fig.add_hline(
                    y=avg_score,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Average: {avg_score:.1f}%",
                    annotation_position="bottom right"
                )

                fig.update_layout(
                    title="Quiz Performance Over Time",
                    xaxis_title="Date",
                    yaxis_title="Score (%)",
                    yaxis_range=[0, 100],
                    height=400,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Take some quizzes to see performance trends.")

            # Topic mastery
            st.markdown("### 🎓 Topic Mastery Levels")

            if 'topic_mastery' in progress_data:
                mastery_data = progress_data['topic_mastery']

                # Create gauge charts for each topic
                for topic, mastery in list(mastery_data.items())[:6]:  # Show first 6 topics
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.write(f"**{topic}**")

                    with col2:
                        st.progress(mastery / 100)
                        st.caption(f"{mastery:.1f}% mastery")

                    st.markdown("---")

elif page == "⚙️ System Settings":
    st.markdown('<div class="main-header">⚙️ System Settings</div>', unsafe_allow_html=True)

    settings_tabs = st.tabs(["🔧 Application Settings", "🤖 AI Configuration", "📚 Content Settings", "🔒 User Management"])

    with settings_tabs[0]:
        st.markdown('<div class="section-header">Application Configuration</div>', unsafe_allow_html=True)

        # General settings
        col1, col2 = st.columns(2)

        with col1:
            theme = st.selectbox(
                "Theme",
                ["Light", "Dark", "System Default"],
                key="app_theme"
            )

            language = st.selectbox(
                "Language",
                ["English", "Spanish", "French", "German", "Chinese"],
                key="app_language"
            )

        with col2:
            auto_save = st.checkbox("Auto-save Progress", value=True, key="auto_save")
            notifications = st.checkbox("Enable Notifications", value=True, key="notifications")

        # Storage settings
        st.markdown("### 💾 Storage Settings")

        storage_col1, storage_col2 = st.columns(2)

        with storage_col1:
            cache_size = st.slider(
                "Cache Size (MB)",
                min_value=100,
                max_value=1000,
                value=500,
                key="cache_size"
            )

            clear_cache = st.button("🗑️ Clear Cache", key="clear_cache")
            if clear_cache:
                st.success("Cache cleared successfully!")

        with storage_col2:
            data_retention = st.selectbox(
                "Data Retention Period",
                ["30 days", "90 days", "1 year", "Forever"],
                key="data_retention"
            )

            export_data = st.button("📥 Export All Data", key="export_all")
            if export_data:
                st.info("Export functionality coming soon!")

        # Save settings
        st.markdown("---")
        if st.button("💾 Save Settings", type="primary", key="save_settings"):
            # Here you would save settings to config file
            st.success("Settings saved successfully!")

    with settings_tabs[1]:
        st.markdown('<div class="section-header">AI Model Configuration</div>', unsafe_allow_html=True)

        # AI Model Selection
        st.markdown("### 🤖 AI Model Settings")

        model_col1, model_col2 = st.columns(2)

        with model_col1:
            nlp_model = st.selectbox(
                "NLP Model",
                ["spaCy (en_core_web_md)", "BERT Base", "DistilBERT", "Custom Model"],
                key="nlp_model"
            )

            llm_model = st.selectbox(
                "Text Generation Model",
                ["GPT-2", "T5 Base", "LLaMA 7B", "Custom Fine-tuned"],
                key="llm_model"
            )

        with model_col2:
            embedding_model = st.selectbox(
                "Embedding Model",
                ["Sentence-BERT", "OpenAI Embeddings", "Custom Embeddings"],
                key="embedding_model"
            )

            model_temperature = st.slider(
                "Creativity (Temperature)",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="model_temp"
            )

        # Performance settings
        st.markdown("### ⚡ Performance Settings")

        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            batch_size = st.select_slider(
                "Batch Size",
                options=[8, 16, 32, 64, 128],
                value=32,
                key="batch_size"
            )

            use_gpu = st.checkbox("Use GPU Acceleration", value=torch.cuda.is_available(), key="use_gpu")

        with perf_col2:
            max_tokens = st.number_input(
                "Max Tokens (Response Length)",
                min_value=50,
                max_value=2000,
                value=500,
                key="max_tokens"
            )

        # Model testing
        st.markdown("### 🧪 Test AI Models")

        test_input = st.text_area(
            "Test input for AI models:",
            "Explain cloud computing in simple terms.",
            height=100,
            key="test_input"
        )

        if st.button("🧪 Test AI Response", key="test_ai"):
            with st.spinner("Testing AI models..."):
                try:
                    if st.session_state.ai_engine:
                        response = st.session_state.ai_engine.generate_text(test_input)
                        st.success("AI Response Generated!")

                        with st.expander("View Response", expanded=True):
                            st.write(response)
                    else:
                        st.warning("AI Engine not initialized. Please load a textbook first.")
                except Exception as e:
                    st.error(f"Error testing AI: {str(e)}")

    with settings_tabs[2]:
        st.markdown('<div class="section-header">Content & Learning Settings</div>', unsafe_allow_html=True)

        # Content settings
        st.markdown("### 📚 Content Preferences")

        content_col1, content_col2 = st.columns(2)

        with content_col1:
            default_difficulty = st.selectbox(
                "Default Difficulty",
                ["Beginner", "Intermediate", "Advanced"],
                key="default_difficulty"
            )

            include_examples = st.checkbox("Always Include Code Examples", value=True, key="include_examples")

        with content_col2:
            show_diagrams = st.checkbox("Always Show Diagrams", value=True, key="show_diagrams")

            auto_generate_quiz = st.checkbox("Auto-generate Quiz After Lessons", value=True, key="auto_quiz")

        # Learning path settings
        st.markdown("### 🎓 Learning Path Settings")

        learning_col1, learning_col2 = st.columns(2)

        with learning_col1:
            adaptive_learning = st.checkbox("Enable Adaptive Learning", value=True, key="adaptive")

            daily_goal = st.number_input(
                "Daily Learning Goal (minutes)",
                min_value=15,
                max_value=240,
                value=60,
                step=15,
                key="daily_goal"
            )

        with learning_col2:
            enable_spaced_repetition = st.checkbox("Enable Spaced Repetition", value=True, key="spaced_rep")

            review_frequency = st.selectbox(
                "Review Frequency",
                ["Weekly", "Bi-weekly", "Monthly"],
                key="review_freq"
            )

        # Content management
        st.markdown("### 🗂️ Content Management")

        if st.button("🔄 Refresh Content Index", key="refresh_content"):
            with st.spinner("Refreshing content index..."):
                if st.session_state.textbook_processor:
                    st.session_state.textbook_processor.refresh_index()
                    st.success("Content index refreshed!")
                else:
                    st.warning("No textbook loaded.")

        if st.button("🧹 Clear Generated Content", key="clear_content"):
            st.session_state.lessons_generated = []
            st.session_state.current_lesson = None
            st.session_state.current_quiz = None
            st.success("Generated content cleared!")

    with settings_tabs[3]:
        st.markdown('<div class="section-header">User & Security Settings</div>', unsafe_allow_html=True)

        # User profile
        st.markdown("### 👤 User Profile")

        profile_col1, profile_col2 = st.columns(2)

        with profile_col1:
            username = st.text_input("Username", value=st.session_state.current_user or "Student", key="username")
            email = st.text_input("Email", value="student@example.com", key="user_email")

        with profile_col2:
            role = st.selectbox(
                "Role",
                ["Student", "Instructor", "Administrator", "Researcher"],
                key="user_role_setting"
            )

            experience_level = st.select_slider(
                "Experience Level",
                options=["Beginner", "Intermediate", "Advanced", "Expert"],
                value="Intermediate",
                key="exp_level"
            )

        # Security settings
        st.markdown("### 🔒 Security & Privacy")

        security_col1, security_col2 = st.columns(2)

        with security_col1:
            enable_2fa = st.checkbox("Enable Two-Factor Authentication", value=False, key="enable_2fa")

            data_encryption = st.checkbox("Encrypt Local Data", value=True, key="data_encrypt")

        with security_col2:
            auto_logout = st.selectbox(
                "Auto-logout After",
                ["15 minutes", "30 minutes", "1 hour", "4 hours", "Never"],
                key="auto_logout"
            )

        # Data privacy
        st.markdown("### 📊 Data & Privacy")

        privacy_col1, privacy_col2 = st.columns(2)

        with privacy_col1:
            share_analytics = st.checkbox(
                "Share Anonymous Analytics",
                value=True,
                help="Help improve the platform by sharing anonymous usage data",
                key="share_analytics"
            )

        with privacy_col2:
            delete_account = st.button("🗑️ Delete Account Data", key="delete_account")
            if delete_account:
                st.warning("This will delete all your data. Are you sure?")
                confirm = st.checkbox("I understand this action cannot be undone")
                if confirm and st.button("⚠️ Permanently Delete", type="primary"):
                    # Here you would delete user data
                    st.success("Account data deleted successfully!")
                    st.session_state.initialized = False
                    st.rerun()

        # Export/Import
        st.markdown("### 📥 Export/Import Data")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "CSV", "PDF Report"],
                key="export_format"
            )

        with export_col2:
            if st.button("📤 Export Learning Data", key="export_learning"):
                # Generate export data
                export_data = {
                    "user": username,
                    "progress": st.session_state.progress_tracker.get_progress_data() if st.session_state.progress_tracker else {},
                    "lessons": st.session_state.lessons_generated,
                    "timestamp": datetime.now().isoformat()
                }

                import json

                st.download_button(
                    label="📥 Download Data",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"learning_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**AI Lesson Studio v2.0**")
    st.caption("Complete Cloud Computing Education Platform")

with footer_col2:
    st.markdown("**Built with**")
    st.caption("Streamlit • HuggingFace • spaCy • Transformers")

with footer_col3:
    st.markdown("**License**")
    st.caption("MIT Open Source • 100% Free")

st.markdown(
    "<div style='text-align: center; color: #666; margin-top: 2rem;'>"
    "☁️ Transform Cloud Education with AI • 🚀 No Cloud Costs • 🔒 Privacy First"
    "</div>",
    unsafe_allow_html=True
)