import re
import pandas as pd
from typing import List, Dict, Any
import pdfplumber


class TableExtractor:
    """Extract and process tables from textbooks"""

    def __init__(self):
        self.table_patterns = [
            r'Table \d+[:.]',
            r'TABLE \d+',
            r'\|\s*.+\s*\|',  # Markdown tables
            r'\+[-]+\+'  # ASCII tables
        ]

    def extract_tables_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables from plain text"""
        tables = []

        # Find table sections
        table_sections = self._find_table_sections(text)

        for section in table_sections:
            table = self._parse_table_section(section)
            if table:
                tables.append(table)

        return tables

    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF using pdfplumber"""
        tables = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()

                    for table_num, table_data in enumerate(page_tables):
                        if table_data:
                            table = {
                                'page': page_num + 1,
                                'table_num': table_num,
                                'data': table_data,
                                'rows': len(table_data),
                                'columns': len(table_data[0]) if table_data[0] else 0,
                                'source': 'pdfplumber'
                            }
                            tables.append(table)
        except Exception as e:
            print(f"PDF table extraction error: {e}")

        return tables

    def _find_table_sections(self, text: str) -> List[str]:
        """Find potential table sections in text"""
        sections = []
        lines = text.split('\n')

        current_section = []
        in_table = False

        for line in lines:
            line_stripped = line.strip()

            # Check if line looks like table start
            if any(re.match(pattern, line_stripped, re.IGNORECASE)
                   for pattern in self.table_patterns):
                if current_section and in_table:
                    sections.append('\n'.join(current_section))
                    current_section = []
                in_table = True

            if in_table:
                current_section.append(line)

                # Check for table end (blank line or non-table content)
                if not line_stripped or len(line_stripped.split()) > 10:
                    if len(current_section) > 3:  # Minimum table size
                        sections.append('\n'.join(current_section))
                    current_section = []
                    in_table = False

        return sections

    def _parse_table_section(self, section: str) -> Dict[str, Any]:
        """Parse a table section into structured data"""
        lines = [line.strip() for line in section.split('\n') if line.strip()]

        if len(lines) < 2:
            return None

        # Try to detect table format
        if any('|' in line for line in lines[:3]):
            return self._parse_markdown_table(lines)
        elif any('+' in line for line in lines[:3]):
            return self._parse_ascii_table(lines)
        else:
            return self._parse_simple_table(lines)

    def _parse_markdown_table(self, lines: List[str]) -> Dict[str, Any]:
        """Parse markdown-style table"""
        data = []

        for line in lines:
            if '|' in line:
                # Split by pipe, remove empty cells
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:
                    data.append(cells)

        if len(data) >= 2:
            return {
                'format': 'markdown',
                'data': data,
                'headers': data[0] if data else [],
                'rows': data[1:] if len(data) > 1 else []
            }

        return None

    def _parse_ascii_table(self, lines: List[str]) -> Dict[str, Any]:
        """Parse ASCII-art table"""
        # Remove border lines
        content_lines = [line for line in lines if not re.match(r'^[+\-]+$', line)]

        data = []
        for line in content_lines:
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|')]
                data.append(cells)

        if data:
            return {
                'format': 'ascii',
                'data': data,
                'headers': data[0] if data else [],
                'rows': data[1:] if len(data) > 1 else []
            }

        return None

    def _parse_simple_table(self, lines: List[str]) -> Dict[str, Any]:
        """Parse simple aligned text table"""
        data = []

        for line in lines:
            # Split by multiple spaces
            cells = re.split(r'\s{2,}', line)
            cells = [cell.strip() for cell in cells if cell.strip()]

            if cells:
                data.append(cells)

        if data:
            return {
                'format': 'simple',
                'data': data,
                'headers': data[0] if data else [],
                'rows': data[1:] if len(data) > 1 else []
            }

        return None

    def convert_to_dataframe(self, table_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert extracted table to pandas DataFrame"""
        if not table_data or 'data' not in table_data:
            return pd.DataFrame()

        data = table_data['data']

        if len(data) >= 2:
            # Use first row as headers
            headers = data[0]
            rows = data[1:]

            # Ensure all rows have same number of columns
            max_cols = len(headers)
            normalized_rows = []

            for row in rows:
                if len(row) < max_cols:
                    row = row + [''] * (max_cols - len(row))
                elif len(row) > max_cols:
                    row = row[:max_cols]
                normalized_rows.append(row)

            return pd.DataFrame(normalized_rows, columns=headers)

        return pd.DataFrame()