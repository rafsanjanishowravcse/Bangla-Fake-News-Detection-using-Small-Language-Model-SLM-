import os
import json
import uuid
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from PIL import Image, ExifTags
import cv2
import numpy as np

# OCR imports with fallback handling
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not available. Will use Tesseract fallback.")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. OCR functionality will be limited.")

from fact_check_llm import verify_news

class ImageFactChecker:
    """Handles image-based fact checking with OCR and verification"""
    
    def __init__(self, upload_dir: str = "uploaded_images", metadata_dir: str = "image_metadata"):
        self.upload_dir = upload_dir
        self.metadata_dir = metadata_dir
        self.ocr_reader = None
        
        # Ensure directories exist
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Initialize OCR reader if available
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['bn', 'en'], gpu=False)
                print("EasyOCR initialized successfully")
            except Exception as e:
                print(f"Failed to initialize EasyOCR: {e}")
                self.ocr_reader = None
    
    def process_image_upload(self, image_path_or_url: str) -> Dict:
        """
        Process image upload from file path or URL
        Returns metadata dict with file_path, timestamp, dimensions
        """
        try:
            # Generate unique image ID
            image_id = str(uuid.uuid4())
            
            # Handle URL vs file path
            if image_path_or_url.startswith(('http://', 'https://')):
                # Download from URL
                response = requests.get(image_path_or_url, timeout=30)
                response.raise_for_status()
                
                # Determine file extension from URL or content type
                content_type = response.headers.get('content-type', '')
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = '.jpg'
                elif 'png' in content_type:
                    ext = '.png'
                elif 'webp' in content_type:
                    ext = '.webp'
                else:
                    ext = '.jpg'  # Default fallback
                
                filename = f"{image_id}{ext}"
                file_path = os.path.join(self.upload_dir, filename)
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                # Local file path
                if not os.path.exists(image_path_or_url):
                    raise FileNotFoundError(f"Image file not found: {image_path_or_url}")
                
                # Get file extension
                _, ext = os.path.splitext(image_path_or_url)
                if not ext:
                    ext = '.jpg'  # Default fallback
                
                filename = f"{image_id}{ext}"
                file_path = os.path.join(self.upload_dir, filename)
                
                # Copy file to upload directory
                with open(image_path_or_url, 'rb') as src, open(file_path, 'wb') as dst:
                    dst.write(src.read())
            
            # Validate and normalize image
            normalized_path = self._normalize_image_orientation(file_path)
            
            # Get image metadata
            with Image.open(normalized_path) as img:
                width, height = img.size
                format_name = img.format
            
            metadata = {
                'image_id': image_id,
                'original_path': image_path_or_url,
                'stored_path': normalized_path,
                'filename': filename,
                'upload_timestamp': datetime.now().isoformat(),
                'dimensions': {'width': width, 'height': height},
                'format': format_name,
                'file_size': os.path.getsize(normalized_path)
            }
            
            # Save metadata
            self._save_image_metadata(metadata, image_id)
            
            return metadata
            
        except Exception as e:
            print(f"Error processing image upload: {e}")
            raise
    
    def _normalize_image_orientation(self, image_path: str) -> str:
        """
        Normalize image orientation based on EXIF data
        Returns path to normalized image
        """
        try:
            with Image.open(image_path) as img:
                # Check for EXIF orientation
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    exif = dict(img._getexif().items())
                    orientation = exif.get(ExifTags.Orientation, 1)
                    
                    # Rotate image based on orientation
                    if orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)
                
                # Save normalized image
                normalized_path = image_path.replace('.', '_normalized.')
                img.save(normalized_path, quality=95, optimize=True)
                
                # Replace original with normalized version
                os.replace(normalized_path, image_path)
                
                return image_path
                
        except Exception as e:
            print(f"Warning: Could not normalize image orientation: {e}")
            return image_path
    
    def _save_image_metadata(self, metadata: Dict, image_id: str):
        """Save image metadata to JSON file"""
        metadata_path = os.path.join(self.metadata_dir, f"{image_id}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """
        Extract text from image using EasyOCR (primary) or Tesseract (fallback)
        Returns dict with ocr_text, language, confidence, segments
        """
        try:
            # Try EasyOCR first
            if self.ocr_reader:
                return self._extract_with_easyocr(image_path)
            
            # Fallback to Tesseract
            if TESSERACT_AVAILABLE:
                return self._extract_with_tesseract(image_path)
            
            raise Exception("No OCR engine available")
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return {
                'ocr_text': '',
                'language': 'unknown',
                'confidence': 0.0,
                'segments': [],
                'error': str(e)
            }
    
    def _extract_with_easyocr(self, image_path: str) -> Dict:
        """Extract text using EasyOCR"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not read image")
            
            # Run OCR
            results = self.ocr_reader.readtext(image)
            
            # Process results
            segments = []
            full_text = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low confidence results
                    segments.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    full_text.append(text)
            
            ocr_text = ' '.join(full_text)
            
            # Detect language
            language = self._detect_language(ocr_text)
            
            return {
                'ocr_text': ocr_text,
                'language': language,
                'confidence': np.mean([s['confidence'] for s in segments]) if segments else 0.0,
                'segments': segments,
                'method': 'easyocr'
            }
            
        except Exception as e:
            print(f"EasyOCR extraction failed: {e}")
            raise
    
    def _extract_with_tesseract(self, image_path: str) -> Dict:
        """Extract text using Tesseract as fallback"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not read image")
            
            # Convert to RGB for Tesseract
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run OCR with Bengali and English
            custom_config = r'--oem 3 --psm 6 -l ben+eng'
            text = pytesseract.image_to_string(image_rgb, config=custom_config)
            
            # Get detailed data for confidence
            data = pytesseract.image_to_data(image_rgb, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            
            # Detect language
            language = self._detect_language(text)
            
            return {
                'ocr_text': text.strip(),
                'language': language,
                'confidence': avg_confidence,
                'segments': [],  # Tesseract doesn't provide detailed segments easily
                'method': 'tesseract'
            }
            
        except Exception as e:
            print(f"Tesseract extraction failed: {e}")
            raise
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text (simple heuristic)"""
        if not text.strip():
            return 'unknown'
        
        # Check for Bengali characters
        bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars > 0 and bengali_chars / total_chars > 0.3:
            return 'bengali'
        else:
            return 'english'
    
    def clean_ocr_text(self, raw_text: str) -> str:
        """Clean and normalize OCR text"""
        if not raw_text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(raw_text.split())
        
        # Remove common OCR artifacts
        text = text.replace('|', 'I')  # Common OCR mistake
        text = text.replace('0', 'O')  # In some contexts
        
        # Normalize unicode
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        
        return text.strip()
    
    def verify_image_news(self, image_input) -> Tuple[str, str, str, str]:
        """
        Main function to verify news from image
        Returns: (claim, verdict_english, verdict_original, ocr_text)
        """
        try:
            # Process image upload
            metadata = self.process_image_upload(image_input)
            
            # Extract OCR text
            ocr_result = self.extract_text_from_image(metadata['stored_path'])
            ocr_text = self.clean_ocr_text(ocr_result['ocr_text'])
            
            if not ocr_text.strip():
                error_msg = "No text could be extracted from the image. Please try with a clearer image."
                return error_msg, error_msg, error_msg, "No text detected"
            
            # Update metadata with OCR results
            metadata['ocr_result'] = ocr_result
            self._save_image_metadata(metadata, metadata['image_id'])
            
            # Verify the extracted text using existing pipeline
            claim, verdict_english, verdict_original = verify_news(ocr_text)
            
            return claim, verdict_english, verdict_original, ocr_text
            
        except Exception as e:
            print(f"Error in image verification: {e}")
            error_msg = f"Error processing image: {str(e)}"
            return error_msg, error_msg, error_msg, "Error occurred"

# Global instance
image_fact_checker = ImageFactChecker()

def verify_image_news(image_input):
    """Wrapper function for Gradio integration"""
    return image_fact_checker.verify_image_news(image_input)
