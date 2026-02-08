# AI Lesson Studio - API Documentation

## Overview
AI Lesson Studio provides a comprehensive RESTful API for interacting with the cloud computing learning platform. All endpoints return JSON responses.

**Base URL:** `http://localhost:8501/api/v1/` (Streamlit app)
**WebSocket URL:** `ws://localhost:8501/ws/` (Real-time updates)

## Authentication
Most endpoints don't require authentication for basic functionality. Advanced features use session-based authentication.

```python
# Python Example
import requests

BASE_URL = "http://localhost:8501/api/v1/"

# Create a session
session = requests.Session()
session.headers.update({
    'Content-Type': 'application/json',
    'User-Id': 'student_123'  # Optional user identifier
})