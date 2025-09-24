from typing import List, Dict
import os
import logging
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import cv2
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()

client = OpenAI(api_key=os.getenv('openai_api'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TesseractRAGPipeline:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.embeddings = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Configure Tesseract (you might need to change this path based on your installation)
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
    def preprocess_image(self, image):
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Apply dilation to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        gray = cv2.dilate(gray, kernel, iterations=1)
        
        return gray

    def detect_tables(self, image):
        """Detect table regions in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine horizontal and vertical lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area
        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 1000:  # Minimum area threshold
                table_regions.append((x, y, w, h))
        
        return table_regions

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text and tables from PDF using Tesseract OCR."""
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            full_text = []
            
            for i, image in enumerate(images):
                # Convert PIL image to OpenCV format
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Detect tables first
                table_regions = self.detect_tables(opencv_image)
                
                # Process tables
                table_texts = []
                for (x, y, w, h) in table_regions:
                    table_roi = opencv_image[y:y+h, x:x+w]
                    # Preprocess table region
                    processed_roi = self.preprocess_image(table_roi)
                    # Extract text from table with specific configuration
                    table_text = pytesseract.image_to_string(
                        processed_roi,
                        config='--oem 3 --psm 6'
                    )
                    if table_text.strip():
                        table_texts.append(f"[TABLE]\n{table_text}\n[/TABLE]")
                        # Mask out the table region
                        opencv_image[y:y+h, x:x+w] = 255
                
                # Process remaining text
                processed_image = self.preprocess_image(opencv_image)
                text = pytesseract.image_to_string(processed_image)
                
                # Combine page text with table text
                page_text = f"\n=== Page {i+1} ===\n"
                if text.strip():
                    page_text += text.strip() + "\n"
                if table_texts:
                    page_text += "\nDetected Tables:\n" + "\n".join(table_texts)
                
                full_text.append(page_text)
            
            return "\n".join(full_text)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using OpenAI API."""
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return []

    def build_retrieval_index(self, chunks: List[str]):
        """Build embeddings index."""
        self.chunks = chunks
        self.embeddings = []
        for chunk in chunks:
            embedding = self.get_embedding(chunk)
            if embedding:
                self.embeddings.append(embedding)

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve most relevant chunks using OpenAI embeddings."""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        similarities = []
        for i, emb in enumerate(self.embeddings):
            score = self.cosine_similarity(query_embedding, emb)
            similarities.append((score, self.chunks[i]))
        
        similarities.sort(reverse=True)
        return [{"score": score, "text": text} for score, text in similarities[:k]]

def main():
    pdf_path = "Test.pdf"
    pipeline = TesseractRAGPipeline()
    
    print("Extracting text from PDF...")
    full_text = pipeline.extract_text_from_pdf(pdf_path)
    if not full_text:
        logger.error("No text extracted from PDF.")
        return
    
    print("Creating chunks...")
    chunks = pipeline.chunk_text(full_text)
    logger.info(f"Created {len(chunks)} chunks.")
    
    print("Building retrieval index...")
    pipeline.build_retrieval_index(chunks)
    
    # Test queries
    test_queries = [
        "What tables are present in the document?",
        "What is the content of the first table?",
        "Is there any text around the tables?"
    ]
    
    print("\nTesting retrieval:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = pipeline.retrieve(query)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.4f}):")
            print(result['text'])

if __name__ == "__main__":
    main()