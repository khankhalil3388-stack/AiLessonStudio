import os
import json
import pickle
import yaml
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime


class FileHandler:
    """Advanced file handling utilities"""

    @staticmethod
    def read_json(file_path: Path) -> Dict[str, Any]:
        """Read JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON {file_path}: {e}")
            return {}

    @staticmethod
    def write_json(data: Dict[str, Any], file_path: Path, indent: int = 2):
        """Write data to JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            print(f"Error writing JSON {file_path}: {e}")
            return False

    @staticmethod
    def read_csv(file_path: Path) -> List[Dict[str, Any]]:
        """Read CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        except Exception as e:
            print(f"Error reading CSV {file_path}: {e}")
            return []

    @staticmethod
    def save_dataframe(df: pd.DataFrame, file_path: Path,
                       format: str = 'csv'):
        """Save DataFrame in various formats"""
        try:
            if format == 'csv':
                df.to_csv(file_path, index=False)
            elif format == 'excel':
                df.to_excel(file_path, index=False)
            elif format == 'json':
                df.to_json(file_path, orient='records', indent=2)
            elif format == 'parquet':
                df.to_parquet(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            return True
        except Exception as e:
            print(f"Error saving {format} {file_path}: {e}")
            return False