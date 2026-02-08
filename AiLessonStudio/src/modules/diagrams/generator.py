import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from io import BytesIO
import base64
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np


@dataclass
class DiagramDefinition:
    """Diagram definition and metadata"""
    diagram_id: str
    diagram_type: str
    title: str
    description: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    layout: str
    styling: Dict[str, Any]
    metadata: Dict[str, Any]


class DiagramGenerator:
    """Advanced diagram generator for cloud computing concepts"""

    # Cloud architecture templates
    CLOUD_TEMPLATES = {
        'three_tier': {
            'name': 'Three-Tier Web Application',
            'description': 'Traditional web application with presentation, application, and database tiers',
            'nodes': [
                {'id': 'client', 'type': 'client', 'label': 'User Browser'},
                {'id': 'elb', 'type': 'load_balancer', 'label': 'Load Balancer'},
                {'id': 'web1', 'type': 'server', 'label': 'Web Server 1'},
                {'id': 'web2', 'type': 'server', 'label': 'Web Server 2'},
                {'id': 'app1', 'type': 'server', 'label': 'App Server 1'},
                {'id': 'app2', 'type': 'server', 'label': 'App Server 2'},
                {'id': 'db_master', 'type': 'database', 'label': 'Master DB'},
                {'id': 'db_replica', 'type': 'database', 'label': 'Read Replica'},
                {'id': 'cache', 'type': 'cache', 'label': 'Redis Cache'},
                {'id': 'cdn', 'type': 'cdn', 'label': 'CDN'},
                {'id': 'storage', 'type': 'storage', 'label': 'Object Storage'}
            ],
            'edges': [
                {'from': 'client', 'to': 'cdn', 'label': 'Static Assets'},
                {'from': 'client', 'to': 'elb', 'label': 'HTTP/HTTPS'},
                {'from': 'elb', 'to': 'web1', 'label': 'Load Balanced'},
                {'from': 'elb', 'to': 'web2', 'label': 'Load Balanced'},
                {'from': 'web1', 'to': 'app1', 'label': 'API Calls'},
                {'from': 'web2', 'to': 'app2', 'label': 'API Calls'},
                {'from': 'app1', 'to': 'db_master', 'label': 'Write'},
                {'from': 'app2', 'to': 'db_master', 'label': 'Write'},
                {'from': 'app1', 'to': 'db_replica', 'label': 'Read'},
                {'from': 'app2', 'to': 'db_replica', 'label': 'Read'},
                {'from': 'app1', 'to': 'cache', 'label': 'Cache'},
                {'from': 'app2', 'to': 'cache', 'label': 'Cache'},
                {'from': 'web1', 'to': 'storage', 'label': 'Uploads'},
                {'from': 'web2', 'to': 'storage', 'label': 'Uploads'}
            ]
        },
        'microservices': {
            'name': 'Microservices Architecture',
            'description': 'Distributed system with independently deployable services',
            'nodes': [
                {'id': 'api_gateway', 'type': 'gateway', 'label': 'API Gateway'},
                {'id': 'auth_service', 'type': 'service', 'label': 'Auth Service'},
                {'id': 'user_service', 'type': 'service', 'label': 'User Service'},
                {'id': 'order_service', 'type': 'service', 'label': 'Order Service'},
                {'id': 'product_service', 'type': 'service', 'label': 'Product Service'},
                {'id': 'payment_service', 'type': 'service', 'label': 'Payment Service'},
                {'id': 'notification_service', 'type': 'service', 'label': 'Notification Service'},
                {'id': 'message_queue', 'type': 'queue', 'label': 'Message Queue'},
                {'id': 'service_registry', 'type': 'registry', 'label': 'Service Registry'},
                {'id': 'config_server', 'type': 'config', 'label': 'Config Server'},
                {'id': 'monitoring', 'type': 'monitoring', 'label': 'Monitoring'},
                {'id': 'logging', 'type': 'logging', 'label': 'Centralized Logging'}
            ],
            'edges': [
                {'from': 'api_gateway', 'to': 'auth_service', 'label': 'Auth'},
                {'from': 'api_gateway', 'to': 'user_service', 'label': 'User Data'},
                {'from': 'api_gateway', 'to': 'order_service', 'label': 'Orders'},
                {'from': 'api_gateway', 'to': 'product_service', 'label': 'Products'},
                {'from': 'order_service', 'to': 'payment_service', 'label': 'Payment'},
                {'from': 'order_service', 'to': 'notification_service', 'label': 'Notifications'},
                {'from': 'auth_service', 'to': 'message_queue', 'label': 'Events'},
                {'from': 'user_service', 'to': 'message_queue', 'label': 'Events'},
                {'from': 'all_services', 'to': 'service_registry', 'label': 'Register'},
                {'from': 'all_services', 'to': 'config_server', 'label': 'Config'},
                {'from': 'all_services', 'to': 'monitoring', 'label': 'Metrics'},
                {'from': 'all_services', 'to': 'logging', 'label': 'Logs'}
            ]
        },
        'serverless': {
            'name': 'Serverless Application',
            'description': 'Event-driven architecture using serverless functions',
            'nodes': [
                {'id': 's3_trigger', 'type': 'storage', 'label': 'S3 Bucket'},
                {'id': 'api_gateway', 'type': 'gateway', 'label': 'API Gateway'},
                {'id': 'lambda1', 'type': 'function', 'label': 'Image Processing'},
                {'id': 'lambda2', 'type': 'function', 'label': 'Data Validation'},
                {'id': 'lambda3', 'type': 'function', 'label': 'Business Logic'},
                {'id': 'dynamodb', 'type': 'database', 'label': 'DynamoDB'},
                {'id': 'sns', 'type': 'notification', 'label': 'SNS Topic'},
                {'id': 'sqs', 'type': 'queue', 'label': 'SQS Queue'},
                {'id': 'cloudwatch', 'type': 'monitoring', 'label': 'CloudWatch'},
                {'id': 'cognito', 'type': 'auth', 'label': 'Cognito'},
                {'id': 'cloudfront', 'type': 'cdn', 'label': 'CloudFront'}
            ],
            'edges': [
                {'from': 's3_trigger', 'to': 'lambda1', 'label': 'Object Created'},
                {'from': 'api_gateway', 'to': 'lambda2', 'label': 'HTTP Request'},
                {'from': 'lambda1', 'to': 'dynamodb', 'label': 'Store Metadata'},
                {'from': 'lambda2', 'to': 'lambda3', 'label': 'Process Data'},
                {'from': 'lambda3', 'to': 'sns', 'label': 'Send Notification'},
                {'from': 'lambda3', 'to': 'sqs', 'label': 'Queue Message'},
                {'from': 'all_lambdas', 'to': 'cloudwatch', 'label': 'Logs/Metrics'},
                {'from': 'api_gateway', 'to': 'cognito', 'label': 'Authenticate'},
                {'from': 'cloudfront', 'to': 's3_trigger', 'label': 'Static Assets'}
            ]
        }
    }

    # Mermaid.js diagram templates
    MERMAID_TEMPLATES = {
        'flowchart': """graph TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Process]
    B -->|No| D[Alternative]
    C --> E[End]
    D --> E""",

        'sequence': """sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant Database

    User->>Frontend: Request Data
    Frontend->>Backend: API Call
    Backend->>Database: Query
    Database-->>Backend: Results
    Backend-->>Frontend: Response
    Frontend-->>User: Display""",

        'class': """classDiagram
    class CloudService {
        +String name
        +String provider
        +deploy()
        +scale()
    }

    class EC2 {
        +String instanceType
        +String ami
        +start()
        +stop()
    }

    class S3 {
        +String bucketName
        +upload()
        +download()
    }

    CloudService <|-- EC2
    CloudService <|-- S3""",

        'state': """stateDiagram-v2
    [*] --> Stopped
    Stopped --> Starting: start()
    Starting --> Running: initialized
    Running --> Stopping: stop()
    Stopping --> Stopped: terminated
    Running --> Error: failure
    Error --> Running: recover()
    Error --> Stopped: forceStop()"""
    }

    def __init__(self, config):
        self.config = config

        # Node styling configuration
        self.node_styles = {
            'client': {'color': '#4CAF50', 'shape': 'circle'},
            'server': {'color': '#2196F3', 'shape': 'square'},
            'database': {'color': '#FF9800', 'shape': 'cylinder'},
            'load_balancer': {'color': '#9C27B0', 'shape': 'diamond'},
            'cache': {'color': '#F44336', 'shape': 'hexagon'},
            'storage': {'color': '#009688', 'shape': 'folder'},
            'cdn': {'color': '#673AB7', 'shape': 'cloud'},
            'gateway': {'color': '#3F51B5', 'shape': 'pentagon'},
            'service': {'color': '#00BCD4', 'shape': 'ellipse'},
            'queue': {'color': '#FF5722', 'shape': 'queue'},
            'function': {'color': '#E91E63', 'shape': 'triangle'},
            'monitoring': {'color': '#795548', 'shape': 'star'}
        }

        print("✅ Diagram Generator initialized")

    def get_cloud_diagram_for_topic(self, topic: str,
                                    diagram_type: str = 'architecture') -> Dict[str, Any]:
        """Get appropriate diagram for cloud computing topic"""
        topic_lower = topic.lower()

        # Map topics to diagram types
        topic_mappings = {
            'three tier': 'three_tier',
            'microservice': 'microservices',
            'serverless': 'serverless',
            'aws': 'three_tier',
            'azure': 'microservices',
            'google cloud': 'serverless',
            'ec2': 'three_tier',
            'lambda': 'serverless',
            's3': 'serverless',
            'kubernetes': 'microservices',
            'docker': 'microservices'
        }

        # Find matching template
        selected_template = 'three_tier'  # default

        for keyword, template in topic_mappings.items():
            if keyword in topic_lower:
                selected_template = template
                break

        # Get template
        template = self.CLOUD_TEMPLATES.get(selected_template, self.CLOUD_TEMPLATES['three_tier'])

        # Create diagram definition
        diagram_id = f"diagram_{topic.replace(' ', '_')}_{datetime.now().timestamp()}"

        diagram = DiagramDefinition(
            diagram_id=diagram_id,
            diagram_type=diagram_type,
            title=template['name'],
            description=f"{template['description']} - Generated for: {topic}",
            nodes=template['nodes'],
            edges=template['edges'],
            layout='hierarchical',
            styling={'theme': 'cloud', 'animate': True},
            metadata={'topic': topic, 'template': selected_template}
        )

        return self._generate_diagram_output(diagram)

    def generate_custom_diagram(self, nodes: List[Dict[str, Any]],
                                edges: List[Dict[str, Any]],
                                title: str = "Custom Architecture",
                                diagram_type: str = "architecture") -> Dict[str, Any]:
        """Generate custom diagram from nodes and edges"""
        diagram_id = f"custom_{datetime.now().timestamp()}"

        diagram = DiagramDefinition(
            diagram_id=diagram_id,
            diagram_type=diagram_type,
            title=title,
            description="Custom cloud architecture diagram",
            nodes=nodes,
            edges=edges,
            layout='force_directed',
            styling={'theme': 'custom', 'animate': False},
            metadata={'custom': True}
        )

        return self._generate_diagram_output(diagram)

    def _generate_diagram_output(self, diagram: DiagramDefinition) -> Dict[str, Any]:
        """Generate multiple output formats for diagram"""
        outputs = {
            'mermaid': self._generate_mermaid(diagram),
            'plotly': self._generate_plotly(diagram),
            'matplotlib': self._generate_matplotlib(diagram),
            'networkx': self._generate_networkx(diagram)
        }

        return {
            'diagram': diagram.__dict__,
            'outputs': outputs
        }

    def _generate_mermaid(self, diagram: DiagramDefinition) -> Dict[str, Any]:
        """Generate Mermaid.js diagram code"""
        diagram_type = diagram.diagram_type

        if diagram_type == 'flowchart':
            return self._generate_mermaid_flowchart(diagram)
        elif diagram_type == 'sequence':
            return self._generate_mermaid_sequence(diagram)
        elif diagram_type == 'architecture':
            return self._generate_mermaid_architecture(diagram)
        else:
            # Default to flowchart
            return self._generate_mermaid_flowchart(diagram)

    def _generate_mermaid_flowchart(self, diagram: DiagramDefinition) -> Dict[str, Any]:
        """Generate Mermaid flowchart"""
        lines = ["graph TD"]

        # Add nodes
        for node in diagram.nodes:
            node_id = node['id']
            node_label = node.get('label', node_id)
            node_type = node.get('type', 'default')

            # Map node types to Mermaid shapes
            shape_map = {
                'client': '(({label}))',
                'server': '[{label}]',
                'database': '[({label})]',
                'load_balancer': '{{label}}',
                'cache': '[/{label}/]',
                'storage': '[[{label}]]',
                'default': '({label})'
            }

            shape = shape_map.get(node_type, shape_map['default'])
            line = f"    {node_id}{shape.format(label=node_label)}"
            lines.append(line)

        # Add edges
        for edge in diagram.edges:
            from_node = edge['from']
            to_node = edge['to']
            label = edge.get('label', '')

            if label:
                line = f"    {from_node} -->|{label}| {to_node}"
            else:
                line = f"    {from_node} --> {to_node}"
            lines.append(line)

        mermaid_code = "\n".join(lines)

        return {
            'type': 'mermaid',
            'format': 'flowchart',
            'code': mermaid_code,
            'html': f'<div class="mermaid">\n{mermaid_code}\n</div>'
        }

    def _generate_mermaid_sequence(self, diagram: DiagramDefinition) -> Dict[str, Any]:
        """Generate Mermaid sequence diagram"""
        lines = ["sequenceDiagram"]

        # Add participants
        participants = set()
        for node in diagram.nodes:
            node_id = node['id']
            node_label = node.get('label', node_id)
            participants.add((node_id, node_label))

        for participant_id, participant_label in participants:
            lines.append(f"    participant {participant_id} as {participant_label}")

        lines.append("")  # Empty line

        # Add interactions
        for edge in diagram.edges:
            from_node = edge['from']
            to_node = edge['to']
            label = edge.get('label', 'interact')

            line = f"    {from_node}->>{to_node}: {label}"
            lines.append(line)

        mermaid_code = "\n".join(lines)

        return {
            'type': 'mermaid',
            'format': 'sequence',
            'code': mermaid_code,
            'html': f'<div class="mermaid">\n{mermaid_code}\n</div>'
        }

    def _generate_mermaid_architecture(self, diagram: DiagramDefinition) -> Dict[str, Any]:
        """Generate Mermaid architecture diagram"""
        lines = ["graph TB"]

        # Group nodes by type for better visualization
        node_groups = {}
        for node in diagram.nodes:
            node_type = node.get('type', 'default')
            if node_type not in node_groups:
                node_groups[node_type] = []
            node_groups[node_type].append(node)

        # Add subgraphs for each node type
        for group_name, group_nodes in node_groups.items():
            group_label = group_name.replace('_', ' ').title()
            lines.append(f"    subgraph {group_label}")

            for node in group_nodes:
                node_id = node['id']
                node_label = node.get('label', node_id)
                lines.append(f"        {node_id}[{node_label}]")

            lines.append("    end")

        lines.append("")  # Empty line

        # Add edges
        for edge in diagram.edges:
            from_node = edge['from']
            to_node = edge['to']
            label = edge.get('label', '')

            if label:
                line = f"    {from_node} -->|{label}| {to_node}"
            else:
                line = f"    {from_node} --> {to_node}"
            lines.append(line)

        mermaid_code = "\n".join(lines)

        return {
            'type': 'mermaid',
            'format': 'architecture',
            'code': mermaid_code,
            'html': f'<div class="mermaid">\n{mermaid_code}\n</div>'
        }

    def _generate_plotly(self, diagram: DiagramDefinition) -> Dict[str, Any]:
        """Generate Plotly interactive diagram"""
        # Create node positions
        nodes = diagram.nodes
        edges = diagram.edges

        # Generate positions (simple grid layout)
        positions = {}
        num_nodes = len(nodes)
        cols = int(np.ceil(np.sqrt(num_nodes)))

        for i, node in enumerate(nodes):
            row = i // cols
            col = i % cols
            positions[node['id']] = (col * 2, -row * 2)

        # Create edge traces
        edge_x = []
        edge_y = []
        edge_text = []

        for edge in edges:
            from_node = edge['from']
            to_node = edge['to']

            if from_node in positions and to_node in positions:
                x0, y0 = positions[from_node]
                x1, y1 = positions[to_node]

                # Add line with arrow
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

                # Midpoint for label
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                edge_text.append((mid_x, mid_y, edge.get('label', '')))

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []

        for node in nodes:
            node_id = node['id']
            if node_id in positions:
                x, y = positions[node_id]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node.get('label', node_id))

                # Get color based on node type
                node_type = node.get('type', 'default')
                color_map = {
                    'client': '#4CAF50',
                    'server': '#2196F3',
                    'database': '#FF9800',
                    'load_balancer': '#9C27B0',
                    'default': '#607D8B'
                }
                node_colors.append(color_map.get(node_type, '#607D8B'))
                node_sizes.append(20)

        # Create figure
        fig = go.Figure()

        # Add edges
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                showlegend=False
            ))

        # Add edge labels
        for x, y, text in edge_text:
            if text:
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=text,
                    showarrow=False,
                    font=dict(size=10, color="#555"),
                    bgcolor="white",
                    bordercolor="#ddd",
                    borderwidth=1,
                    borderpad=2,
                    opacity=0.8
                )

        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            showlegend=False
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=diagram.title,
                font=dict(size=20)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=500
        )

        # Convert to HTML
        plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        return {
            'type': 'plotly',
            'figure': fig.to_dict(),
            'html': plotly_html
        }

    def _generate_matplotlib(self, diagram: DiagramDefinition) -> Dict[str, Any]:
        """Generate Matplotlib static diagram"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from io import BytesIO
        import base64

        fig, ax = plt.subplots(figsize=(12, 8))

        # Simple box diagram
        nodes = diagram.nodes
        edges = diagram.edges

        # Create positions
        positions = {}
        num_nodes = len(nodes)
        cols = min(5, num_nodes)
        rows = (num_nodes + cols - 1) // cols

        node_width = 1.5
        node_height = 0.8
        spacing = 0.5

        for i, node in enumerate(nodes):
            row = i // cols
            col = i % cols

            x = col * (node_width + spacing)
            y = -row * (node_height + spacing)

            positions[node['id']] = (x, y, node_width, node_height)

            # Draw node
            node_type = node.get('type', 'default')
            color_map = {
                'client': '#4CAF50',
                'server': '#2196F3',
                'database': '#FF9800',
                'load_balancer': '#9C27B0',
                'cache': '#F44336',
                'storage': '#009688',
                'default': '#607D8B'
            }

            color = color_map.get(node_type, '#607D8B')

            rect = patches.Rectangle(
                (x - node_width / 2, y - node_height / 2),
                node_width,
                node_height,
                linewidth=2,
                edgecolor='black',
                facecolor=color,
                alpha=0.7
            )
            ax.add_patch(rect)

            # Add label
            label = node.get('label', node['id'])
            ax.text(x, y, label, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')

        # Draw edges
        for edge in edges:
            from_node = edge['from']
            to_node = edge['to']

            if from_node in positions and to_node in positions:
                x1, y1, w1, h1 = positions[from_node]
                x2, y2, w2, h2 = positions[to_node]

                # Draw arrow
                ax.annotate('',
                            xy=(x2, y2),
                            xytext=(x1, y1),
                            arrowprops=dict(
                                arrowstyle='->',
                                color='gray',
                                lw=1.5,
                                alpha=0.6
                            )
                            )

                # Add edge label if exists
                label = edge.get('label', '')
                if label:
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    ax.text(mid_x, mid_y, label,
                            fontsize=8, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white',
                                      edgecolor='gray',
                                      alpha=0.8))

        # Set limits and remove axes
        ax.set_xlim(-1, cols * (node_width + spacing))
        ax.set_ylim(-rows * (node_height + spacing), 1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Add title
        plt.title(diagram.title, fontsize=16, fontweight='bold', pad=20)

        # Save to base64
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()

        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            'type': 'matplotlib',
            'format': 'png',
            'base64': image_base64,
            'html': f'<img src="data:image/png;base64,{image_base64}" alt="{diagram.title}">'
        }

    def _generate_networkx(self, diagram: DiagramDefinition) -> Dict[str, Any]:
        """Generate NetworkX graph representation"""
        import networkx as nx
        import json

        G = nx.DiGraph()

        # Add nodes
        for node in diagram.nodes:
            G.add_node(
                node['id'],
                label=node.get('label', node['id']),
                type=node.get('type', 'default')
            )

        # Add edges
        for edge in diagram.edges:
            G.add_edge(
                edge['from'],
                edge['to'],
                label=edge.get('label', ''),
                weight=1.0
            )

        # Convert to dictionary
        graph_dict = {
            'nodes': list(G.nodes(data=True)),
            'edges': list(G.edges(data=True)),
            'graph_info': {
                'number_of_nodes': G.number_of_nodes(),
                'number_of_edges': G.number_of_edges(),
                'is_directed': G.is_directed()
            }
        }

        return {
            'type': 'networkx',
            'graph': graph_dict,
            'json': json.dumps(graph_dict, default=str)
        }

    def generate_workflow_diagram(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate workflow diagram from steps"""
        mermaid_code = ["graph TD"]

        for i, step in enumerate(steps):
            step_id = f"step_{i}"
            step_label = step.get('action', f'Step {i + 1}')
            step_type = step.get('type', 'process')

            # Map step types to shapes
            shape_map = {
                'start': '(({label}))',
                'end': '(({label}))',
                'process': '[{label}]',
                'decision': '{{label}}',
                'input': '[/{label}/]',
                'output': '[\\{label}\\]',
                'database': '[({label})]'
            }

            shape = shape_map.get(step_type, shape_map['process'])
            line = f"    {step_id}{shape.format(label=step_label)}"
            mermaid_code.append(line)

        # Add connections
        for i in range(len(steps) - 1):
            current_id = f"step_{i}"
            next_id = f"step_{i + 1}"

            # Check for conditions
            condition = steps[i].get('condition')
            if condition:
                true_id = steps[i].get('true_step', next_id)
                false_id = steps[i].get('false_step', f"step_{i + 2}" if i + 2 < len(steps) else next_id)

                mermaid_code.append(f"    {current_id} -->|Yes| {true_id}")
                mermaid_code.append(f"    {current_id} -->|No| {false_id}")
            else:
                mermaid_code.append(f"    {current_id} --> {next_id}")

        mermaid_str = "\n".join(mermaid_code)

        return {
            'type': 'workflow',
            'mermaid': mermaid_str,
            'html': f'<div class="mermaid">\n{mermaid_str}\n</div>'
        }

    def get_diagram_template(self, template_name: str) -> Dict[str, Any]:
        """Get predefined diagram template"""
        if template_name in self.CLOUD_TEMPLATES:
            return self.CLOUD_TEMPLATES[template_name]
        elif template_name in self.MERMAID_TEMPLATES:
            return {'mermaid': self.MERMAID_TEMPLATES[template_name]}
        else:
            return self.CLOUD_TEMPLATES['three_tier']

    def export_diagram(self, diagram_output: Dict[str, Any],
                       format: str = 'html') -> str:
        """Export diagram in specified format"""
        if format == 'html':
            return diagram_output.get('outputs', {}).get('mermaid', {}).get('html', '')

        elif format == 'json':
            import json
            return json.dumps(diagram_output, indent=2, default=str)

        elif format == 'mermaid':
            return diagram_output.get('outputs', {}).get('mermaid', {}).get('code', '')

        elif format == 'png':
            # Return base64 encoded image
            return diagram_output.get('outputs', {}).get('matplotlib', {}).get('base64', '')

        else:
            return f"Format {format} not supported"