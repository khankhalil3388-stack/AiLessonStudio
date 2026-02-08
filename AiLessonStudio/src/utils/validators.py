import re
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import mimetypes


class Validator:
    """Data validator for various types of input"""

    def __init__(self, config):
        self.config = config

    def validate(self, data, data_type='generic', **kwargs):
        """Validate data based on type"""
        if data_type == 'email':
            return self.validate_email(data)
        elif data_type == 'filename':
            return self.validate_filename(data)
        elif data_type == 'file':
            return self.validate_file(data, **kwargs)
        elif data_type == 'json':
            return self.validate_json(data)
        elif data_type == 'text':
            return self.validate_text(data, **kwargs)
        elif data_type == 'url':
            return self.validate_url(data)
        elif data_type == 'username':
            return self.validate_username(data)
        elif data_type == 'password':
            return self.validate_password(data)
        else:
            # Generic validation - just check if data exists
            return data is not None and data != ''

    def validate_email(self, email: str) -> Tuple[bool, str]:
        """Validate email address"""
        if not email:
            return False, "Email cannot be empty"

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if bool(re.match(pattern, email)):
            return True, "Email is valid"
        else:
            return False, "Invalid email format"

    def validate_filename(self, filename: str) -> Tuple[bool, str]:
        """Validate filename"""
        if not filename:
            return False, "Filename cannot be empty"

        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in filename for char in invalid_chars):
            return False, f"Filename contains invalid characters: {invalid_chars}"

        # Check for reserved names
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3',
                          'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                          'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6',
                          'LPT7', 'LPT8', 'LPT9']
        if filename.upper() in reserved_names:
            return False, "Filename is a reserved system name"

        # Check length
        if len(filename) > 255:
            return False, "Filename too long (max 255 characters)"

        # Check for empty filename
        if filename.strip() == '':
            return False, "Filename cannot be empty"

        return True, "Filename is valid"

    def validate_file(self, file_path: Union[str, Path],
                      allowed_extensions: Optional[List[str]] = None,
                      max_size_mb: Optional[int] = None) -> Tuple[bool, str]:
        """Validate file"""
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"

        # Check if it's a file (not directory)
        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"

        # Check file extension
        if allowed_extensions:
            is_valid_ext = file_path.suffix.lower() in [ext.lower() for ext in allowed_extensions]
            if not is_valid_ext:
                allowed_str = ', '.join(allowed_extensions)
                return False, f"Invalid file extension. Allowed: {allowed_str}"

        # Check file size
        if max_size_mb is not None:
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)

            if size_mb > max_size_mb:
                return False, f"File size ({size_mb:.1f} MB) exceeds maximum ({max_size_mb} MB)"

        return True, "File is valid"

    def validate_json(self, content: str) -> Tuple[bool, Optional[Dict]]:
        """Validate JSON content"""
        if not content:
            return False, None, "JSON content is empty"

        try:
            data = json.loads(content)
            return True, data, "JSON is valid"
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {str(e)}"

    def validate_text(self, text: str,
                      min_length: int = 1,
                      max_length: int = 10000,
                      allow_empty: bool = False) -> Tuple[bool, str]:
        """Validate text"""
        if not text and not allow_empty:
            return False, "Text cannot be empty"

        if text is None:
            return False, "Text cannot be None"

        length = len(text)

        if min_length and length < min_length:
            return False, f"Text too short (minimum {min_length} characters)"

        if max_length and length > max_length:
            return False, f"Text too long (maximum {max_length} characters)"

        return True, "Text is valid"

    def validate_url(self, url: str) -> Tuple[bool, str]:
        """Validate URL"""
        if not url:
            return False, "URL cannot be empty"

        pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
        if bool(re.match(pattern, url)):
            return True, "URL is valid"
        else:
            return False, "Invalid URL format"

    def validate_username(self, username: str) -> Tuple[bool, str]:
        """Validate username"""
        if not username:
            return False, "Username cannot be empty"

        # Check length
        if len(username) < 3:
            return False, "Username must be at least 3 characters"

        if len(username) > 30:
            return False, "Username must be at most 30 characters"

        # Check characters (alphanumeric and underscores)
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "Username can only contain letters, numbers, and underscores"

        # Check if starts with letter
        if not username[0].isalpha():
            return False, "Username must start with a letter"

        return True, "Username is valid"

    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength"""
        if not password:
            return False, ["Password cannot be empty"]

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

        # Check for common passwords
        common_passwords = ['password', '123456', 'qwerty', 'admin', 'letmein']
        if password.lower() in common_passwords:
            errors.append("Password is too common")

        return len(errors) == 0, errors

    def validate_number(self, value: Any,
                        min_value: Optional[float] = None,
                        max_value: Optional[float] = None,
                        is_integer: bool = False) -> Tuple[bool, str]:
        """Validate number"""
        if value is None:
            return False, "Number cannot be None"

        # Try to convert to number
        try:
            if is_integer:
                num = int(value)
            else:
                num = float(value)
        except (ValueError, TypeError):
            return False, "Invalid number format"

        # Check min value
        if min_value is not None and num < min_value:
            return False, f"Number must be at least {min_value}"

        # Check max value
        if max_value is not None and num > max_value:
            return False, f"Number must be at most {max_value}"

        return True, "Number is valid"

    def validate_list(self, items: List,
                      min_items: Optional[int] = None,
                      max_items: Optional[int] = None,
                      item_validator=None) -> Tuple[bool, str]:
        """Validate list"""
        if items is None:
            return False, "List cannot be None"

        if not isinstance(items, list):
            return False, "Value must be a list"

        # Check min items
        if min_items is not None and len(items) < min_items:
            return False, f"List must contain at least {min_items} items"

        # Check max items
        if max_items is not None and len(items) > max_items:
            return False, f"List must contain at most {max_items} items"

        # Validate individual items if validator provided
        if item_validator:
            for i, item in enumerate(items):
                if not item_validator(item):
                    return False, f"Item at position {i} is invalid"

        return True, "List is valid"

    def validate_dict(self, data: Dict,
                      required_keys: Optional[List[str]] = None,
                      allowed_keys: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Validate dictionary"""
        if data is None:
            return False, "Dictionary cannot be None"

        if not isinstance(data, dict):
            return False, "Value must be a dictionary"

        # Check required keys
        if required_keys:
            for key in required_keys:
                if key not in data:
                    return False, f"Missing required key: {key}"

        # Check allowed keys
        if allowed_keys:
            for key in data.keys():
                if key not in allowed_keys:
                    return False, f"Disallowed key: {key}"

        return True, "Dictionary is valid"

    def detect_file_type(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """Detect file type from content"""
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not file_path.exists():
            return False, "File does not exist"

        # Try mimetype first
        mime_type, _ = mimetypes.guess_type(str(file_path))

        if mime_type:
            return True, mime_type

        # Fallback to extension
        extension_map = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.json': 'application/json',
            '.py': 'text/x-python',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.csv': 'text/csv',
            '.html': 'text/html',
            '.md': 'text/markdown'
        }

        file_type = extension_map.get(file_path.suffix.lower(), 'application/octet-stream')
        return True, file_type


class Validators:
    """Input validation utilities - static methods for backward compatibility"""

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address"""
        if not email:
            return False

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename"""
        if not filename:
            return False

        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in filename for char in invalid_chars):
            return False

        # Check for reserved names
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3',
                          'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                          'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6',
                          'LPT7', 'LPT8', 'LPT9']
        if filename.upper() in reserved_names:
            return False

        # Check length
        if len(filename) > 255:
            return False

        # Check for empty filename
        if filename.strip() == '':
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


# Add alias for backward compatibility
DataValidator = Validator