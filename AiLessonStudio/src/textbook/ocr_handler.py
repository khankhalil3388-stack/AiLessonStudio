import pytesseract
from PIL import Image
import io
import base64
from typing import Optional, List, Dict, Any
import fitz  # PyMuPDF


class OCRHandler:
    """Handle OCR for scanned textbook pages"""

    def __init__(self, config):
        self.config = config
        self.language = config.processing.OCR_LANGUAGE

        # Configure Tesseract path if needed
        try:
            pytesseract.get_tesseract_version()
        except:
            # Try to set path for common locations
            import platform
            system = platform.system()

            if system == "Windows":
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            elif system == "Linux":
                pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
            elif system == "Darwin":  # macOS
                pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

    def extract_text_from_image(self, image_bytes: bytes,
                                language: str = None) -> Dict[str, Any]:
        """Extract text from image using OCR"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Preprocess image for better OCR
            processed_image = self._preprocess_image(image)

            # Perform OCR
            lang = language or self.language
            text = pytesseract.image_to_string(
                processed_image,
                lang=lang,
                config='--psm 3 --oem 3'  # Automatic page segmentation, LSTM OCR engine
            )

            # Get additional data
            data = pytesseract.image_to_data(
                processed_image,
                lang=lang,
                output_type=pytesseract.Output.DICT
            )

            # Calculate confidence
            confidences = [conf for conf in data['conf'] if conf > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return {
                'text': text.strip(),
                'confidence': avg_confidence,
                'word_count': len(text.split()),
                'language': lang,
                'success': True
            }

        except Exception as e:
            return {
                'text': '',
                'confidence': 0,
                'error': str(e),
                'success': False
            }

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')

        # Increase contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)

        # Resize if too small
        if image.width < 300 or image.height < 300:
            image = image.resize(
                (max(300, image.width * 2), max(300, image.height * 2)),
                Image.Resampling.LANCZOS
            )

        return image

    def extract_text_from_pdf_page(self, pdf_path: str, page_num: int,
                                   language: str = None) -> Dict[str, Any]:
        """Extract text from specific PDF page using OCR"""
        try:
            doc = fitz.open(pdf_path)

            if page_num < 0 or page_num >= len(doc):
                return {'text': '', 'error': 'Invalid page number', 'success': False}

            page = doc[page_num]

            # Get page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
            img_data = pix.tobytes("png")

            # Perform OCR
            result = self.extract_text_from_image(img_data, language)

            doc.close()

            return result

        except Exception as e:
            return {
                'text': '',
                'confidence': 0,
                'error': str(e),
                'success': False
            }

    def batch_process_pdf(self, pdf_path: str,
                          start_page: int = 0,
                          end_page: Optional[int] = None,
                          language: str = None) -> List[Dict[str, Any]]:
        """Batch process multiple pages of a PDF"""
        results = []

        try:
            doc = fitz.open(pdf_path)

            if end_page is None:
                end_page = len(doc)

            for page_num in range(start_page, min(end_page, len(doc))):
                print(f"Processing page {page_num + 1}/{len(doc)}...")

                result = self.extract_text_from_pdf_page(pdf_path, page_num, language)
                result['page'] = page_num + 1

                results.append(result)

            doc.close()

        except Exception as e:
            print(f"Batch processing error: {e}")

        return results

    def detect_language(self, image_bytes: bytes) -> str:
        """Detect language in image"""
        try:
            # Simple language detection using Tesseract
            image = Image.open(io.BytesIO(image_bytes))

            # Try common languages
            languages = ['eng', 'spa', 'fra', 'deu', 'ita']

            best_lang = 'eng'
            best_score = 0

            for lang in languages:
                try:
                    text = pytesseract.image_to_string(
                        image,
                        lang=lang,
                        config='--psm 3 --oem 3'
                    )

                    # Simple scoring: count valid words
                    words = text.split()
                    valid_words = [w for w in words if len(w) > 2]
                    score = len(valid_words)

                    if score > best_score:
                        best_score = score
                        best_lang = lang

                except:
                    continue

            return best_lang

        except:
            return 'eng'  # Default to English

    def extract_with_layout(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract text with layout preservation"""
        try:
            image = Image.open(io.BytesIO(image_bytes))

            # Get OCR data with bounding boxes
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                output_type=pytesseract.Output.DICT,
                config='--psm 3 --oem 3'
            )

            # Group by lines
            lines = {}
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    line_num = data['line_num'][i]
                    if line_num not in lines:
                        lines[line_num] = []

                    lines[line_num].append({
                        'text': data['text'][i],
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': data['conf'][i]
                    })

            # Sort lines by vertical position
            sorted_lines = sorted(lines.items(), key=lambda x: x[1][0]['top'])

            # Reconstruct text with layout
            reconstructed = []
            for line_num, words in sorted_lines:
                # Sort words horizontally
                words.sort(key=lambda w: w['left'])

                line_text = ' '.join(w['text'] for w in words)
                reconstructed.append(line_text)

            full_text = '\n'.join(reconstructed)

            return {
                'text': full_text,
                'layout_data': lines,
                'success': True
            }

        except Exception as e:
            return {
                'text': '',
                'error': str(e),
                'success': False
            }