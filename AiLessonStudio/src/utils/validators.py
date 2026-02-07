import re
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import mimetypes


class Validators:
    """Input validation utilities"""

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename"""
        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in filename for char in invalid_chars):
            return False

        # Check length
        if len(filename) > 255:
            return False

        return True

    @staticmethod
    def validate_file_extension(file_path: Path, allowed_extensions: List[str]) -> bool:
        """Validate file extension"""
        return file_path.suffix.lower() in [ext.lower() for ext in allowed_extensions]

    @staticmethod
    def validate_file_size(file_path: Path, max_size_mb: int) -> Tuple[bool, str]:
        """Validate file size"""
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        if size_mb > max_size_mb:
            return False, f"File size ({size_mb:.1f} MB) exceeds maximum ({max_size_mb} MB)"

        return True, f"File size: {size_mb:.1f} MB"

    @staticmethod
    def validate_json(content: str) -> Tuple[bool, Optional[Dict]]:
        """Validate JSON content"""
        try:
            data = json.loads(content)
            return True, data
        except json.JSONDecodeError as e:
            return False, None

    @staticmethod
    def validate_text_length(text: str, min_length: int = 1,
                             max_length: int = 10000) -> Tuple[bool, str]:
        """Validate text length"""
        length = len(text)

        if length < min_length:
            return False, f"Text too short (minimum {min_length} characters)"

        if length > max_length:
            return False, f"Text too long (maximum {max_length} characters)"

        return True, f"Text length: {length} characters"

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL"""
        pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))

    @staticmethod
    def validate_username(username: str) -> Tuple[bool, str]:
        """Validate username"""
        # Check length
        if len(username) < 3:
            return False, "Username must be at least 3 characters"

        if len(username) > 30:
            return False, "Username must be at most 30 characters"

        # Check characters (alphanumeric and underscores)
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "Username can only contain letters, numbers, and underscores"

        return True, "Username is valid"

    @staticmethod
    def validate_password(password: str) -> Tuple[bool, List[str]]:
        """Validate password strength"""
        errors = []

        if len(password) < 8:
            errors.append("Password must be at least 8 characters")

        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")

        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")

        if not re.search(r'[0-9]', password):
            errors.append("Password must contain at least one number")

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")

        return len(errors) == 0, errors

    @staticmethod
    def detect_file_type(file_path: Path) -> str:
        """Detect file type from content"""
        # Try mimetype first
        mime_type, _ = mimetypes.guess_type(str(file_path))

        if mime_type:
            return mime_type

        # Fallback to extension
        extension_map = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.json': 'application/json',
            '.py': 'text/x-python',
            '.jpg': 'image/jpeg',
            '.png': 'image/png'
        }

        return extension_map.get(file_path.suffix.lower(), 'application/octet-stream')