from typing import List, Dict
import os
import logging
import numpy as np
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Text, Image

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('openai_api'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DoclingRAGPipeline:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.embeddings = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.document = None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text and structure from PDF using unstructured."""
        try:
            # Extract elements from PDF
            elements = partition_pdf(pdf_path)
            
            # Get the full text with structure preserved
            full_text = []
            
            # Process each element
            for element in elements:
                if isinstance(element, Text):
                    full_text.append(element.text)
                elif isinstance(element, Table):
                    full_text.append("\nTable Content:")
                    full_text.append(str(element))
                elif isinstance(element, Image):
                    full_text.append(f"\n[Image: {element.text if element.text else 'No caption'}]")
                
                # Add separator between different sections
                full_text.append("-" * 50)
            
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

        # Calculate similarities
        similarities = []
        for i, emb in enumerate(self.embeddings):
            score = self.cosine_similarity(query_embedding, emb)
            similarities.append((score, self.chunks[i]))

        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Return top k results
        return [{"score": score, "text": text} for score, text in similarities[:k]]

def main():
    pdf_path = "Test.pdf"
    pipeline = DoclingRAGPipeline()

    print("Extracting text and structure from PDF...")
    full_text = pipeline.extract_text_from_pdf(pdf_path)
    if not full_text:
        logger.error("No text extracted from PDF.")
        return

    print("\nChunking text...")
    chunks = pipeline.chunk_text(full_text)
    logger.info(f"Created {len(chunks)} chunks.")
    
    print("\nBuilding retrieval index...")
    pipeline.build_retrieval_index(chunks)

    # Test queries
    test_queries = [
        "What tables are present in the document?",
        "What images are in the document?",
        "Is there any code or formulas in the document?"
    ]

    print("\nTesting retrieval:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = pipeline.retrieve(query)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.4f}):")
            print("-" * 40)
            print(result['text'][:300] + "..." if len(result['text']) > 300 else result['text'])

if __name__ == "__main__":
    main()