import re
import json
import yaml
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow
import matplotlib.patches as mpatches
import numpy as np


@dataclass
class Component:
    """Represents a cloud architecture component"""
    name: str
    type: str  # compute, storage, network, database, security, etc.
    service: Optional[str] = None  # EC2, S3, RDS, etc.
    position: Optional[Tuple[int, int]] = None
    connections: List[str] = None
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.connections is None:
            self.connections = []
        if self.properties is None:
            self.properties = {}


class ArchitectureBuilder:
    """Builds and visualizes cloud architectures programmatically"""

    # Cloud service icons mapping
    SERVICE_ICONS = {
        'EC2': '🖥️',
        'S3': '📦',
        'RDS': '🗄️',
        'Lambda': 'λ',
        'VPC': '🌐',
        'IAM': '🔒',
        'CloudFront': '⚡',
        'Route53': '📍',
        'ELB': '⚖️',
        'AutoScaling': '📈',
        'DynamoDB': '⚡🗄️',
        'SQS': '📨',
        'SNS': '📢',
        'API Gateway': '🚪',
        'CloudWatch': '👁️',
        'EKS': '☸️',
        'ECS': '📦🔄',
        'EBS': '💾',
        'EFS': '📁'
    }

    # Color schemes for different component types
    COLOR_SCHEMES = {
        'compute': {'bg': '#FF6B6B', 'text': '#FFFFFF'},
        'storage': {'bg': '#4ECDC4', 'text': '#FFFFFF'},
        'database': {'bg': '#45B7D1', 'text': '#FFFFFF'},
        'network': {'bg': '#96CEB4', 'text': '#000000'},
        'security': {'bg': '#FFEAA7', 'text': '#000000'},
        'monitoring': {'bg': '#DDA0DD', 'text': '#FFFFFF'},
        'container': {'bg': '#98D8C8', 'text': '#000000'}
    }

    def __init__(self):
        self.components = {}
        self.architecture_graph = nx.DiGraph()

    def add_component(self, component: Component) -> str:
        """Add a component to the architecture"""
        component_id = f"{component.type}_{len(self.components)}"
        self.components[component_id] = component
        self.architecture_graph.add_node(
            component_id,
            name=component.name,
            type=component.type,
            service=component.service
        )
        return component_id

    def connect_components(self, source_id: str, target_id: str,
                           connection_type: str = "network",
                           label: Optional[str] = None) -> None:
        """Connect two components"""
        if source_id in self.components and target_id in self.components:
            self.components[source_id].connections.append(target_id)
            self.architecture_graph.add_edge(
                source_id,
                target_id,
                type=connection_type,
                label=label or f"{connection_type}_connection"
            )

    def generate_mermaid_diagram(self, title: str = "Cloud Architecture") -> str:
        """Generate Mermaid.js diagram code"""
        mermaid_code = f"graph TD\n    subgraph {title}\n"

        # Add nodes
        for comp_id, component in self.components.items():
            icon = self.SERVICE_ICONS.get(component.service, '📦')
            mermaid_code += f"        {comp_id}[{icon} {component.name}]"

            # Add styling based on component type
            if component.type in self.COLOR_SCHEMES:
                color = self.COLOR_SCHEMES[component.type]['bg']
                mermaid_code += f":::type_{component.type}\n"
            else:
                mermaid_code += "\n"

        # Add connections
        for comp_id, component in self.components.items():
            for target_id in component.connections:
                edge_data = self.architecture_graph.get_edge_data(comp_id, target_id)
                connection_type = edge_data.get('type', 'network') if edge_data else 'network'
                label = edge_data.get('label', '') if edge_data else ''

                line_style = {
                    'network': '-->',
                    'data': '-.->',
                    'security': '-.->|Security|',
                    'monitoring': '==>|Monitoring|',
                    'async': '-. async .->'
                }.get(connection_type, '-->')

                if label:
                    mermaid_code += f"        {comp_id} {line_style} {target_id}\n"
                else:
                    mermaid_code += f"        {comp_id} {line_style} {target_id}\n"

        mermaid_code += "    end\n"

        # Add CSS styling
        mermaid_code += "\n    classDef type_compute fill:#FF6B6B,color:#FFFFFF,stroke:#333\n"
        mermaid_code += "    classDef type_storage fill:#4ECDC4,color:#FFFFFF,stroke:#333\n"
        mermaid_code += "    classDef type_database fill:#45B7D1,color:#FFFFFF,stroke:#333\n"
        mermaid_code += "    classDef type_network fill:#96CEB4,color:#000000,stroke:#333\n"
        mermaid_code += "    classDef type_security fill:#FFEAA7,color:#000000,stroke:#333\n"
        mermaid_code += "    classDef type_monitoring fill:#DDA0DD,color:#FFFFFF,stroke:#333\n"

        return mermaid_code

    def generate_matplotlib_diagram(self, figsize: Tuple[int, int] = (12, 8)):
        """Generate matplotlib visualization of the architecture"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')

        # Calculate positions if not specified
        positions = {}
        n_components = len(self.components)

        if n_components > 0:
            # Create a grid layout
            rows = int(np.ceil(np.sqrt(n_components)))
            cols = int(np.ceil(n_components / rows))

            idx = 0
            for comp_id in self.components:
                if self.components[comp_id].position:
                    positions[comp_id] = self.components[comp_id].position
                else:
                    row = idx // cols
                    col = idx % cols
                    x = 20 + col * 60
                    y = 80 - row * 40
                    positions[comp_id] = (x, y)
                    idx += 1

        # Draw components
        for comp_id, component in self.components.items():
            x, y = positions[comp_id]

            # Get color based on component type
            colors = self.COLOR_SCHEMES.get(component.type, {'bg': '#CCCCCC', 'text': '#000000'})

            # Draw rectangle for component
            rect = Rectangle((x - 15, y - 10), 30, 20,
                             facecolor=colors['bg'],
                             edgecolor='black',
                             linewidth=2,
                             alpha=0.9)
            ax.add_patch(rect)

            # Add text
            icon = self.SERVICE_ICONS.get(component.service, '📦')
            ax.text(x, y, f"{icon}\n{component.name}",
                    ha='center', va='center',
                    fontsize=8, color=colors['text'],
                    fontweight='bold')

            # Add component type label
            ax.text(x, y - 15, component.type,
                    ha='center', va='top',
                    fontsize=6, color='gray',
                    style='italic')

        # Draw connections
        for comp_id, component in self.components.items():
            if comp_id in positions:
                source_x, source_y = positions[comp_id]

                for target_id in component.connections:
                    if target_id in positions:
                        target_x, target_y = positions[target_id]

                        # Draw arrow
                        ax.annotate('',
                                    xy=(target_x, target_y + 10),
                                    xytext=(source_x, source_y - 10),
                                    arrowprops=dict(arrowstyle='->',
                                                    color='gray',
                                                    linewidth=1,
                                                    alpha=0.7))

        plt.title('Cloud Architecture Diagram', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig

    def create_standard_architecture(self, arch_type: str) -> Dict:
        """Create standard cloud architectures"""
        architectures = {
            'three_tier': self._create_three_tier_architecture,
            'serverless': self._create_serverless_architecture,
            'microservices': self._create_microservices_architecture,
            'data_lake': self._create_data_lake_architecture,
            'high_availability': self._create_ha_architecture
        }

        if arch_type in architectures:
            return architectures[arch_type]()
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")

    def _create_three_tier_architecture(self) -> Dict:
        """Create a standard three-tier web application architecture"""
        # Reset components
        self.components = {}
        self.architecture_graph.clear()

        # Web tier components
        web_tier = self.add_component(Component(
            name="Web Servers",
            type="compute",
            service="EC2",
            properties={"tier": "web", "auto_scaling": True}
        ))

        alb = self.add_component(Component(
            name="Application Load Balancer",
            type="network",
            service="ELB",
            properties={"type": "application", "https": True}
        ))

        # Application tier components
        app_tier = self.add_component(Component(
            name="Application Servers",
            type="compute",
            service="EC2",
            properties={"tier": "application", "auto_scaling": True}
        ))

        # Database tier components
        database = self.add_component(Component(
            name="Database",
            type="database",
            service="RDS",
            properties={"engine": "MySQL", "multi_az": True}
        ))

        # Security components
        security_group = self.add_component(Component(
            name="Security Groups",
            type="security",
            service="IAM",
            properties={"rules": ["HTTP", "HTTPS", "SSH"]}
        ))

        # Connect components
        self.connect_components(alb, web_tier, "network", "HTTP/HTTPS")
        self.connect_components(web_tier, app_tier, "network", "API Calls")
        self.connect_components(app_tier, database, "data", "SQL Queries")
        self.connect_components(security_group, web_tier, "security", "Firewall Rules")
        self.connect_components(security_group, app_tier, "security", "Firewall Rules")

        return {
            "name": "Three-Tier Web Application",
            "description": "Standard web application with web, application, and database tiers",
            "components": list(self.components.keys())
        }

    def _create_serverless_architecture(self) -> Dict:
        """Create a serverless architecture"""
        self.components = {}
        self.architecture_graph.clear()

        # Serverless components
        api_gateway = self.add_component(Component(
            name="API Gateway",
            type="network",
            service="API Gateway"
        ))

        lambda_func = self.add_component(Component(
            name="Lambda Function",
            type="compute",
            service="Lambda"
        ))

        dynamodb = self.add_component(Component(
            name="DynamoDB",
            type="database",
            service="DynamoDB"
        ))

        s3_bucket = self.add_component(Component(
            name="Static Assets",
            type="storage",
            service="S3"
        ))

        cloudfront = self.add_component(Component(
            name="CDN",
            type="network",
            service="CloudFront"
        ))

        # Connect components
        self.connect_components(api_gateway, lambda_func, "network", "Triggers")
        self.connect_components(lambda_func, dynamodb, "data", "Reads/Writes")
        self.connect_components(cloudfront, s3_bucket, "network", "Serves Content")
        self.connect_components(s3_bucket, lambda_func, "async", "S3 Events")

        return {
            "name": "Serverless Application",
            "description": "Event-driven serverless architecture",
            "components": list(self.components.keys())
        }

    def export_architecture(self, format: str = "json") -> str:
        """Export architecture to various formats"""
        export_data = {
            "metadata": {
                "total_components": len(self.components),
                "total_connections": self.architecture_graph.number_of_edges(),
                "component_types": list(set(c.type for c in self.components.values()))
            },
            "components": {
                comp_id: {
                    "name": comp.name,
                    "type": comp.type,
                    "service": comp.service,
                    "connections": comp.connections,
                    "properties": comp.properties
                }
                for comp_id, comp in self.components.items()
            },
            "connections": [
                {
                    "source": edge[0],
                    "target": edge[1],
                    "type": self.architecture_graph.edges[edge].get('type', 'network')
                }
                for edge in self.architecture_graph.edges()
            ]
        }

        if format == "json":
            return json.dumps(export_data, indent=2)
        elif format == "yaml":
            return yaml.dump(export_data, default_flow_style=False)
        elif format == "plantuml":
            return self._generate_plantuml(export_data)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _generate_plantuml(self, export_data: Dict) -> str:
        """Generate PlantUML diagram code"""
        plantuml = "@startuml\n"
        plantuml += "title Cloud Architecture\n\n"

        # Define component types
        type_colors = {
            'compute': 'LightBlue',
            'storage': 'LightGreen',
            'database': 'LightCoral',
            'network': 'LightYellow',
            'security': 'LightGray'
        }

        for comp_id, comp_data in export_data["components"].items():
            comp_type = comp_data["type"]
            color = type_colors.get(comp_type, 'White')

            plantuml += f"rectangle \"{comp_data['name']}\" as {comp_id} #{color}\n"

        # Add connections
        for conn in export_data["connections"]:
            plantuml += f"{conn['source']} --> {conn['target']} : {conn['type']}\n"

        plantuml += "@enduml"
        return plantuml

    def validate_architecture(self) -> List[str]:
        """Validate the architecture for common issues"""
        issues = []

        # Check for disconnected components
        for comp_id, component in self.components.items():
            if not component.connections and len(self.components) > 1:
                issues.append(f"Component '{component.name}' has no connections")

        # Check for cycles
        try:
            cycles = list(nx.find_cycle(self.architecture_graph))
            if cycles:
                issues.append(f"Architecture contains cycles: {cycles}")
        except nx.NetworkXNoCycle:
            pass

        # Check for security groups if using AWS-like components
        aws_components = [c for c in self.components.values() if c.service in ['EC2', 'RDS', 'Lambda']]
        security_components = [c for c in self.components.values() if c.type == 'security']

        if aws_components and not security_components:
            issues.append("No security components defined for AWS services")

        return issues

    def calculate_cost_estimate(self, region: str = "us-east-1") -> Dict:
        """Generate cost estimate for the architecture"""
        # Mock cost data (in reality, this would use AWS Pricing API)
        service_costs = {
            'EC2': {'small': 0.023, 'medium': 0.046, 'large': 0.092},
            'S3': {'standard': 0.023, 'infrequent': 0.0125, 'glacier': 0.004},
            'RDS': {'small': 0.017, 'medium': 0.034, 'large': 0.068},
            'Lambda': {'requests': 0.0000002, 'compute': 0.00001667},
            'ELB': {'application': 0.0225, 'network': 0.0225},
            'CloudFront': {'requests': 0.085, 'data_out': 0.09}
        }

        total_monthly = 0
        cost_breakdown = {}

        for comp_id, component in self.components.items():
            if component.service in service_costs:
                # Simple estimation based on service type
                service_cost = service_costs[component.service]

                if isinstance(service_cost, dict):
                    # Take average cost for estimation
                    avg_cost = sum(service_cost.values()) / len(service_cost)
                    estimated_cost = avg_cost * 730  # Monthly hours (24*30.5)
                else:
                    estimated_cost = service_cost * 730

                cost_breakdown[component.name] = {
                    'service': component.service,
                    'estimated_monthly': round(estimated_cost, 2),
                    'unit': 'USD'
                }
                total_monthly += estimated_cost

        return {
            'total_monthly': round(total_monthly, 2),
            'currency': 'USD',
            'region': region,
            'breakdown': cost_breakdown,
            'assumptions': [
                '24/7 operation',
                'Average usage patterns',
                'No reserved instances',
                'No data transfer costs included'
            ]
        }