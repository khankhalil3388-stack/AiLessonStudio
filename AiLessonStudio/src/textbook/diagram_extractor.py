import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import os
from typing import List, Dict, Any
import cv2
import numpy as np


class DiagramExtractor:
    """Extract and process diagrams from textbooks"""

    def __init__(self, config):
        self.config = config

    def extract(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract diagrams from PDF"""
        diagrams = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Get image list
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    # Extract image
                    xref = img[0]
                    base_image = doc.extract_image(xref)

                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Check if image is large enough to be a diagram
                        if len(image_bytes) > self.config.processing.DIAGRAM_MIN_SIZE:
                            # Convert to base64 for storage
                            image_b64 = base64.b64encode(image_bytes).decode()

                            # Create diagram entry
                            diagram = {
                                'page': page_num + 1,
                                'index': img_index,
                                'format': image_ext,
                                'size': len(image_bytes),
                                'data': image_b64,
                                'type': self._classify_diagram(image_bytes),
                                'description': self._generate_description(image_bytes)
                            }

                            diagrams.append(diagram)

            doc.close()

        except Exception as e:
            print(f"⚠️ Diagram extraction error: {e}")

        return diagrams

    def _classify_diagram(self, image_bytes: bytes) -> str:
        """Classify diagram type"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Basic classification based on features
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect edges
            edges = cv2.Canny(gray, 50, 150)

            # Count contours (shapes)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 10:
                return "architecture_diagram"
            elif len(contours) > 5:
                return "flowchart"
            else:
                return "illustration"

        except:
            return "unknown"

    def _generate_description(self, image_bytes: bytes) -> str:
        """Generate description for diagram"""
        # In a real implementation, this would use OCR or image captioning
        diagram_type = self._classify_diagram(image_bytes)

        descriptions = {
            "architecture_diagram": "System architecture showing components and connections",
            "flowchart": "Process flowchart with decision points",
            "illustration": "Conceptual illustration or diagram",
            "unknown": "Image from textbook"
        }

        return descriptions.get(diagram_type, "Textbook diagram")