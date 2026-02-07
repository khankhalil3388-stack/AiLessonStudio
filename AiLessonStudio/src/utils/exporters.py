import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import markdown
from weasyprint import HTML
import io


class Exporters:
    """Data export utilities"""

    @staticmethod
    def export_to_json(data: Any, file_path: Path, indent: int = 2) -> bool:
        """Export data to JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False

    @staticmethod
    def export_to_csv(data: List[Dict[str, Any]], file_path: Path) -> bool:
        """Export data to CSV file"""
        try:
            if not data:
                return False

            # Extract headers
            headers = list(data[0].keys())

            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(data)

            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False

    @staticmethod
    def export_to_excel(data: List[Dict[str, Any]], file_path: Path,
                        sheet_name: str = "Data") -> bool:
        """Export data to Excel file"""
        try:
            df = pd.DataFrame(data)
            df.to_excel(file_path, index=False, sheet_name=sheet_name)
            return True
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            return False

    @staticmethod
    def export_to_pdf(html_content: str, file_path: Path) -> bool:
        """Export HTML content to PDF"""
        try:
            HTML(string=html_content).write_pdf(str(file_path))
            return True
        except Exception as e:
            print(f"Error exporting to PDF: {e}")
            return False

    @staticmethod
    def export_lesson_to_pdf(lesson_data: Dict[str, Any], file_path: Path) -> bool:
        """Export lesson to PDF"""
        try:
            # Convert lesson to HTML
            html = Exporters._lesson_to_html(lesson_data)

            # Export to PDF
            return Exporters.export_to_pdf(html, file_path)
        except Exception as e:
            print(f"Error exporting lesson to PDF: {e}")
            return False

    @staticmethod
    def _lesson_to_html(lesson_data: Dict[str, Any]) -> str:
        """Convert lesson data to HTML"""
        lesson = lesson_data.get('lesson', {})

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{lesson.get('title', 'Lesson')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
                h2 {{ color: #34495e; }}
                h3 {{ color: #7f8c8d; }}
                .metadata {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .objective {{ background: #e8f4f8; padding: 10px; border-left: 4px solid #3498db; margin: 5px 0; }}
                .example {{ background: #f9f9f9; padding: 15px; border: 1px solid #ddd; margin: 10px 0; }}
                .code {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; font-family: monospace; }}
                .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <h1>{lesson.get('title', 'Lesson Title')}</h1>

            <div class="metadata">
                <p><strong>Difficulty:</strong> {lesson.get('difficulty', 'Intermediate').title()}</p>
                <p><strong>Estimated Time:</strong> {lesson.get('estimated_time', 30)} minutes</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>

            <div class="section">
                <h2>Introduction</h2>
                <p>{lesson.get('introduction', 'Introduction content')}</p>
            </div>

            <div class="section">
                <h2>Learning Objectives</h2>
                {''.join(f'<div class="objective">{obj}</div>' for obj in lesson.get('learning_objectives', []))}
            </div>

            <div class="section">
                <h2>Key Concepts</h2>
                <ul>
                    {''.join(f'<li>{concept}</li>' for concept in lesson.get('key_concepts', []))}
                </ul>
            </div>
        """

        # Add content sections
        for section in lesson.get('content_sections', []):
            html += f"""
            <div class="section">
                <h2>{section.get('title', 'Section')}</h2>
                <p>{section.get('content', 'Content')}</p>
            </div>
            """

        # Add examples
        if lesson.get('examples'):
            html += """
            <div class="section">
                <h2>Examples</h2>
            """

            for example in lesson.get('examples', []):
                html += f"""
                <div class="example">
                    <h3>{example.get('title', 'Example')}</h3>
                    <p>{example.get('description', 'Description')}</p>
                    {f'<div class="code">{example.get("code", "")}</div>' if example.get('code') else ''}
                    <p><em>{example.get('explanation', 'Explanation')}</em></p>
                </div>
                """

            html += "</div>"

        # Add summary
        html += f"""
            <div class="section">
                <h2>Summary</h2>
                <p>{lesson.get('summary', 'Summary content')}</p>
            </div>

            <div class="footer">
                <p>Generated by AI Lesson Studio - Cloud Computing Education Platform</p>
                <p>© {datetime.now().year} - All rights reserved</p>
            </div>
        </body>
        </html>
        """

        return html

    @staticmethod
    def export_progress_report(progress_data: Dict[str, Any],
                               format: str = 'json') -> Optional[str]:
        """Export progress report in specified format"""
        try:
            if format.lower() == 'json':
                return json.dumps(progress_data, indent=2, default=str)

            elif format.lower() == 'csv':
                # Convert progress data to CSV format
                output = io.StringIO()
                writer = csv.writer(output)

                # Write header
                writer.writerow(['Metric', 'Value'])

                # Write data
                for key, value in progress_data.items():
                    if isinstance(value, (str, int, float, bool)):
                        writer.writerow([key, value])

                return output.getvalue()

            elif format.lower() == 'html':
                return Exporters._progress_to_html(progress_data)

            else:
                return None

        except Exception as e:
            print(f"Error exporting progress report: {e}")
            return None

    @staticmethod
    def _progress_to_html(progress_data: Dict[str, Any]) -> str:
        """Convert progress data to HTML"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Learning Progress Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .metric {{ background: white; padding: 20px; border-radius: 8px; 
                         box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
                .activity {{ border-left: 4px solid #764ba2; padding-left: 15px; margin: 10px 0; }}
                .footer {{ margin-top: 40px; text-align: center; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Learning Progress Report</h1>
                <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
            </div>

            <div class="metric">
                <h2>Overall Progress</h2>
                <div class="metric-value">
                    {progress_data.get('mastery_percentage', 0):.1f}% Mastery
                </div>
                <p>{progress_data.get('mastered_concepts', 0)} out of {progress_data.get('total_concepts', 0)} concepts mastered</p>
            </div>
        """

        # Add recent activity if available
        if 'recent_activity' in progress_data and progress_data['recent_activity']:
            html += """
            <div class="metric">
                <h2>Recent Learning Activity</h2>
            """

            for activity in progress_data['recent_activity'][:5]:
                html += f"""
                <div class="activity">
                    <strong>{activity.get('concept', 'Concept')}</strong><br>
                    {activity.get('type', 'Activity').title()} | 
                    Success: {activity.get('success', 0):.0%} | 
                    Duration: {activity.get('duration', 0):.0f}s
                </div>
                """

            html += "</div>"

        html += """
            <div class="footer">
                <p>AI Lesson Studio - Transform Your Cloud Computing Education</p>
                <p>Generated automatically based on your learning activities</p>
            </div>
        </body>
        </html>
        """

        return html