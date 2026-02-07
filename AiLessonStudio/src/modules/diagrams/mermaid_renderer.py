import re
from typing import Dict, List, Any, Optional
import json


class MermaidRenderer:
    """Render and manipulate Mermaid.js diagrams"""

    def __init__(self):
        self.themes = {
            'default': {
                'theme': 'default',
                'themeVariables': {
                    'primaryColor': '#BB2528',
                    'primaryTextColor': '#fff',
                    'primaryBorderColor': '#7C0000',
                    'lineColor': '#F8B229',
                    'secondaryColor': '#006100',
                    'tertiaryColor': '#fff'
                }
            },
            'neutral': {
                'theme': 'neutral',
                'themeVariables': {
                    'primaryColor': '#64748b',
                    'primaryTextColor': '#fff',
                    'primaryBorderColor': '#475569',
                    'lineColor': '#94a3b8',
                    'secondaryColor': '#334155',
                    'tertiaryColor': '#f1f5f9'
                }
            },
            'dark': {
                'theme': 'dark',
                'themeVariables': {
                    'primaryColor': '#1e293b',
                    'primaryTextColor': '#f8fafc',
                    'primaryBorderColor': '#0f172a',
                    'lineColor': '#475569',
                    'secondaryColor': '#334155',
                    'tertiaryColor': '#64748b'
                }
            },
            'forest': {
                'theme': 'forest',
                'themeVariables': {
                    'primaryColor': '#2d5a27',
                    'primaryTextColor': '#fff',
                    'primaryBorderColor': '#1a3c1c',
                    'lineColor': '#4a7c59',
                    'secondaryColor': '#3d8b3d',
                    'tertiaryColor': '#a8d5ba'
                }
            }
        }

    def render_html(self, mermaid_code: str,
                    theme: str = 'default',
                    height: int = 400,
                    width: str = '100%') -> str:
        """Generate HTML for Mermaid diagram"""
        theme_config = self.themes.get(theme, self.themes['default'])

        html = f'''
        <div class="mermaid-container" style="width: {width}; height: {height}px; overflow: auto;">
            <div class="mermaid" data-theme='{json.dumps(theme_config)}'>
                {mermaid_code}
            </div>
        </div>

        <script>
            // Mermaid initialization
            mermaid.initialize({{
                startOnLoad: true,
                theme: '{theme}',
                flowchart: {{
                    useMaxWidth: true,
                    htmlLabels: true,
                    curve: 'basis'
                }},
                sequenceDiagram: {{
                    useMaxWidth: true,
                    diagramMarginX: 50,
                    diagramMarginY: 10,
                    actorMargin: 50
                }},
                themeVariables: {json.dumps(theme_config['themeVariables'])}
            }});

            // Re-render on window resize
            window.addEventListener('resize', function() {{
                mermaid.init();
            }});
        </script>
        '''

        return html

    def validate_mermaid(self, mermaid_code: str) -> Dict[str, Any]:
        """Validate Mermaid syntax"""
        errors = []
        warnings = []

        # Basic validation rules
        lines = mermaid_code.strip().split('\n')

        if not lines:
            errors.append("Empty Mermaid code")
            return {'valid': False, 'errors': errors, 'warnings': warnings}

        # Check for valid diagram type
        first_line = lines[0].strip().lower()
        valid_types = ['graph', 'flowchart', 'sequenceDiagram',
                       'classDiagram', 'stateDiagram', 'erDiagram',
                       'gantt', 'pie', 'gitGraph', 'journey']

        diagram_type = None
        for vtype in valid_types:
            if first_line.startswith(vtype):
                diagram_type = vtype
                break

        if not diagram_type:
            errors.append(f"Invalid diagram type. Must start with one of: {', '.join(valid_types)}")

        # Check for syntax errors
        if diagram_type in ['graph', 'flowchart']:
            self._validate_graph_syntax(lines, errors, warnings)
        elif diagram_type == 'sequenceDiagram':
            self._validate_sequence_syntax(lines, errors, warnings)

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'diagram_type': diagram_type
        }

    def _validate_graph_syntax(self, lines: List[str], errors: List[str], warnings: List[str]):
        """Validate graph/flowchart syntax"""
        node_ids = set()
        edges = []

        for i, line in enumerate(lines[1:], start=2):  # Skip first line
            line = line.strip()
            if not line or line.startswith('%%'):  # Comment
                continue

            # Check for node definition
            node_match = re.match(r'(\w+)\[?(.+?)?\]?', line)
            if node_match and '-->' not in line and '---' not in line:
                node_id = node_match.group(1)
                node_ids.add(node_id)

            # Check for edge definition
            if '-->' in line or '---' in line:
                edge_parts = re.split(r'(?:-->|--)', line)
                if len(edge_parts) >= 2:
                    from_node = edge_parts[0].strip()
                    to_node_part = edge_parts[1].strip()

                    # Extract to_node (before | if present)
                    to_node = to_node_part.split('|')[0].strip() if '|' in to_node_part else to_node_part

                    edges.append((from_node, to_node))

                    # Check if nodes exist
                    if from_node not in node_ids:
                        warnings.append(f"Line {i}: Node '{from_node}' used before definition")
                    if to_node not in node_ids:
                        warnings.append(f"Line {i}: Node '{to_node}' used before definition")

        # Check for unreachable nodes
        reachable = set()
        for from_node, to_node in edges:
            reachable.add(from_node)
            reachable.add(to_node)

        unreachable = node_ids - reachable
        if unreachable:
            warnings.append(f"Isolated nodes detected: {', '.join(unreachable)}")

    def _validate_sequence_syntax(self, lines: List[str], errors: List[str], warnings: List[str]):
        """Validate sequence diagram syntax"""
        participants = set()

        for i, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line or line.startswith('%%'):
                continue

            # Check for participant definition
            if line.startswith('participant'):
                parts = line.split()
                if len(parts) >= 2:
                    participant = parts[1]
                    participants.add(participant)

            # Check for interactions
            elif '->>' in line or '-->>' in line:
                parts = re.split(r'(?:->>|-->>)', line)
                if len(parts) >= 2:
                    from_part = parts[0].strip()
                    to_part = parts[1].split(':')[0].strip() if ':' in parts[1] else parts[1].strip()

                    # Check if participants are defined
                    if from_part not in participants and not from_part.startswith('participant '):
                        warnings.append(f"Line {i}: Participant '{from_part}' used before definition")
                    if to_part not in participants and not to_part.startswith('participant '):
                        warnings.append(f"Line {i}: Participant '{to_part}' used before definition")

    def extract_nodes_and_edges(self, mermaid_code: str) -> Dict[str, Any]:
        """Extract nodes and edges from Mermaid code"""
        lines = mermaid_code.strip().split('\n')

        if not lines:
            return {'nodes': [], 'edges': []}

        nodes = []
        edges = []
        node_styles = {}

        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('%%'):
                continue

            # Extract node with style
            node_match = re.match(r'(\w+)(\[(.+?)\])?(\((.+?)\))?(\{(.+?)\})?', line)
            if node_match and '-->' not in line and '---' not in line:
                node_id = node_match.group(1)
                node_label = node_match.group(3) or node_match.group(5) or node_id
                node_style = node_match.group(7)

                node = {
                    'id': node_id,
                    'label': node_label,
                    'type': self._detect_node_type(node_label)
                }

                if node_style:
                    node_styles[node_id] = node_style
                    node['style'] = node_style

                nodes.append(node)

            # Extract edge
            if '-->' in line or '---' in line:
                edge_match = re.match(r'(\w+)\s*(?:-->|--)\s*(\w+)(?:\|(.+?)\|)?', line)
                if edge_match:
                    from_node = edge_match.group(1)
                    to_node = edge_match.group(2)
                    edge_label = edge_match.group(3)

                    edge = {
                        'from': from_node,
                        'to': to_node,
                        'type': 'arrow' if '-->' in line else 'line'
                    }

                    if edge_label:
                        edge['label'] = edge_label

                    edges.append(edge)

        return {
            'nodes': nodes,
            'edges': edges,
            'node_styles': node_styles
        }

    def _detect_node_type(self, label: str) -> str:
        """Detect node type from label"""
        label_lower = label.lower()

        type_mapping = {
            'start': ['start', 'begin', 'init'],
            'end': ['end', 'stop', 'finish'],
            'decision': ['decision', 'check', 'if', 'condition'],
            'process': ['process', 'step', 'action', 'task'],
            'input': ['input', 'read', 'receive'],
            'output': ['output', 'write', 'send', 'display'],
            'database': ['database', 'db', 'store', 'storage'],
            'document': ['document', 'file', 'report'],
            'predefined': ['predefined', 'subroutine', 'function']
        }

        for node_type, keywords in type_mapping.items():
            if any(keyword in label_lower for keyword in keywords):
                return node_type

        return 'default'

    def generate_interactive_script(self, mermaid_code: str,
                                    container_id: str = 'mermaid-diagram') -> str:
        """Generate interactive JavaScript for diagram"""
        script = f'''
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const diagram = document.querySelector('#{container_id}');
            if (!diagram) return;

            // Add click handlers to nodes
            diagram.addEventListener('click', function(e) {{
                const node = e.target.closest('.node');
                if (node) {{
                    const nodeId = node.getAttribute('data-node-id');
                    showNodeInfo(nodeId);
                }}
            }});

            // Add hover effects
            diagram.addEventListener('mouseover', function(e) {{
                const node = e.target.closest('.node');
                if (node) {{
                    node.style.cursor = 'pointer';
                    node.style.filter = 'brightness(1.1)';
                }}
            }});

            diagram.addEventListener('mouseout', function(e) {{
                const node = e.target.closest('.node');
                if (node) {{
                    node.style.filter = 'brightness(1)';
                }}
            }});

            // Zoom functionality
            let scale = 1;
            diagram.addEventListener('wheel', function(e) {{
                e.preventDefault();
                scale += e.deltaY * -0.01;
                scale = Math.min(Math.max(0.1, scale), 4);
                diagram.style.transform = `scale(${{scale}})`;
            }}, {{ passive: false }});

            // Pan functionality
            let isDragging = false;
            let startX, startY;
            let translateX = 0, translateY = 0;

            diagram.addEventListener('mousedown', function(e) {{
                isDragging = true;
                startX = e.clientX - translateX;
                startY = e.clientY - translateY;
                diagram.style.cursor = 'grabbing';
            }});

            document.addEventListener('mousemove', function(e) {{
                if (!isDragging) return;
                e.preventDefault();
                translateX = e.clientX - startX;
                translateY = e.clientY - startY;
                diagram.style.transform = `translate(${{translateX}}px, ${{translateY}}px) scale(${{scale}})`;
            }});

            document.addEventListener('mouseup', function() {{
                isDragging = false;
                diagram.style.cursor = 'default';
            }});
        }});

        function showNodeInfo(nodeId) {{
            // This would be implemented based on your application
            console.log('Clicked node:', nodeId);
            // Example: show modal with node information
            // document.getElementById('node-info').innerText = 'Node: ' + nodeId;
            // document.getElementById('node-modal').style.display = 'block';
        }}

        function resetDiagram() {{
            const diagram = document.querySelector('#{container_id}');
            diagram.style.transform = 'scale(1) translate(0, 0)';
            scale = 1;
            translateX = 0;
            translateY = 0;
        }}
        </script>
        '''

        return script