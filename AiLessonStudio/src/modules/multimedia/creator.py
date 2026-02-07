import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import json


class MultimediaCreator:
    """Create interactive multimedia content for lessons"""

    def __init__(self, config):
        self.config = config

    def create_interactive_chart(self, data: Dict[str, Any], chart_type: str = "bar") -> str:
        """Create interactive Plotly chart"""
        if chart_type == "bar":
            fig = px.bar(
                x=list(data.keys()),
                y=list(data.values()),
                title="Cloud Computing Metrics",
                labels={'x': 'Metric', 'y': 'Value'}
            )
        elif chart_type == "pie":
            fig = px.pie(
                values=list(data.values()),
                names=list(data.keys()),
                title="Distribution"
            )
        else:
            fig = px.line(
                x=list(range(len(data))),
                y=list(data.values()),
                title="Trend Analysis"
            )

        return fig.to_html(full_html=False)

    def create_timeline(self, events: List[Dict[str, Any]]) -> str:
        """Create interactive timeline"""
        fig = px.timeline(
            events,
            x_start="start",
            x_end="end",
            y="event",
            title="Cloud Computing Timeline"
        )
        return fig.to_html(full_html=False)