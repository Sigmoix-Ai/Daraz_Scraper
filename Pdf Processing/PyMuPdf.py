import os
import logging
from typing import List, Dict
import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import numpy as np
import json

load_dotenv()


client = OpenAI(api_key=os.getenv('openai_api'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MuPdfRAGPipeline:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.embeddings = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.api_tracker = None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text and metadata from a PDF file using PyMuPDF with enhanced processing."""
        try:
           
            doc = fitz.open(pdf_path)
            full_text = []

            
            metadata = doc.metadata
            if metadata:
                full_text.append("Document Metadata:")
                for key, value in metadata.items():
                    if value and str(value).strip():
                        full_text.append(f"{key}: {value}")
                full_text.append("-" * 50)

          
            for page_num, page in enumerate(doc, 1):
                full_text.append(f"\nPage {page_num}:")
                
                
                blocks = page.get_text("blocks")
                
               
                tables = page.find_tables()
                if tables and tables.tables:
                    full_text.append("\nTables found on this page:")
                    for table in tables:
                        cells = table.extract()
                        table_text = []
                        for row in cells:
                            # Format each row with pipe separators
                            formatted_row = " | ".join(str(cell) for cell in row if str(cell).strip())
                            if formatted_row:
                                table_text.append(formatted_row)
                        if table_text:
                            full_text.append("\n".join(table_text))
                            full_text.append("-" * 50)

                # Extract links
                links = page.get_links()
                if links:
                    full_text.append("\nLinks found on this page:")
                    for link in links:
                        if "uri" in link:
                            full_text.append(f"Link: {link['uri']}")
                    full_text.append("-" * 50)

                # Extract text with proper formatting
                text_with_format = []
                for block in blocks:
                    if block[6] == 0:  # Regular text block
                        text_with_format.append(block[4])
                    elif block[6] == 1:  # Image block
                        text_with_format.append("[Image]")
                
                full_text.append("\n".join(text_with_format))
                full_text.append("-" * 50)  # Page separator

                # Extract images (optional, commented out by default)
                # images = page.get_images()
                # if images:
                #     full_text.append(f"\nFound {len(images)} images on page {page_num}")

            doc.close()
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
            # Use tracked client if available
            api_client = self.api_tracker.client if self.api_tracker and self.api_tracker.client else client
            
            response = api_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            # Track the API call if we have a tracker
            if self.api_tracker:
                self.api_tracker.track_embedding(text)
                
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

def evaluate_answer(generated_answer: str, ground_truth: str) -> dict:
    """Evaluate the generated answer against ground truth using LLM."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Compare the generated answer with the ground truth and provide a detailed evaluation."},
                {"role": "user", "content": f"""
                Compare the following answers and provide evaluation metrics:
                
                Generated Answer:
                {generated_answer}
                
                Ground Truth:
                {ground_truth}
                
                Provide evaluation in the following format:
                1. Accuracy (0-100): How accurate is the generated answer compared to ground truth?
                2. Completeness (0-100): Does it cover all points from ground truth?
                3. Relevance (0-100): How relevant is the answer to the expected content?
                4. Explanation: Brief explanation of the scores
                """}
            ]
        )
        
        evaluation = response.choices[0].message.content
        scores = {
            "accuracy": 0,
            "completeness": 0,
            "relevance": 0,
            "explanation": ""
        }
        
        lines = evaluation.split('\n')
        for line in lines:
            if "Accuracy" in line and ":" in line:
                scores["accuracy"] = int(line.split(":")[1].split()[0])
            elif "Completeness" in line and ":" in line:
                scores["completeness"] = int(line.split(":")[1].split()[0])
            elif "Relevance" in line and ":" in line:
                scores["relevance"] = int(line.split(":")[1].split()[0])
            elif "Explanation" in line and ":" in line:
                scores["explanation"] = line.split(":")[1].strip()
        
        return scores
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        return {
            "accuracy": 0,
            "completeness": 0,
            "relevance": 0,
            "explanation": f"Error in evaluation: {str(e)}"
        }

def main():
    pdf_path = "Test.pdf"
    pipeline = MuPdfRAGPipeline()
    
    full_text = pipeline.extract_text_from_pdf(pdf_path)
    if not full_text:
        logger.error("No text extracted from PDF.")
        return
    
    chunks = pipeline.chunk_text(full_text)
    logger.info(f"Created {len(chunks)} chunks.")
    pipeline.build_retrieval_index(chunks)
    
    # Test cases with ground truth
    test_cases = [
        {
            "question": "What are the tables in the document?",
            "ground_truth": """The document contains several tables showing product information and pricing details. These tables are structured with clear rows and columns, containing numerical data and text entries."""
        },
        {
            "question": "Extract the links from the pdf?",
            "ground_truth": """The document contains several hyperlinks, including external website references and internal document links. Each link is properly formatted and active."""
        }
    ]
    
    output_dir = "evaluation_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    evaluation_results = []
    
    for test_case in test_cases:
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]
        
        results = pipeline.retrieve(question)
        output_file = os.path.join(output_dir, f"pymupdf_results_{len(evaluation_results)}.txt")
        
        context = "\n".join([r['text'] for r in results])
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the question strictly based on the provided context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ]
            )
            generated_answer = response.choices[0].message.content
            
            # Evaluate the answer
            evaluation = evaluate_answer(generated_answer, ground_truth)
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Question:\n{question}\n\n")
                f.write("Retrieved Chunks:\n")
                f.write("="* 50 + "\n\n")
                for i, result in enumerate(results, 1):
                    f.write(f"Chunk {i} (Score: {result['score']:.3f}):\n")
                    f.write(result['text'])
                    f.write("\n" + "-"* 50 + "\n\n")
                
                f.write("\nGenerated Answer:\n")
                f.write("="* 50 + "\n")
                f.write(generated_answer)
                
                f.write("\n\nGround Truth:\n")
                f.write("="* 50 + "\n")
                f.write(ground_truth)
                
                f.write("\n\nEvaluation Results:\n")
                f.write("="* 50 + "\n")
                f.write(f"Accuracy: {evaluation['accuracy']}%\n")
                f.write(f"Completeness: {evaluation['completeness']}%\n")
                f.write(f"Relevance: {evaluation['relevance']}%\n")
                f.write(f"Explanation: {evaluation['explanation']}\n")
            
            evaluation_results.append({
                "question": question,
                "evaluation": evaluation,
                "generated_answer": generated_answer
            })
            
            print(f"Results have been written to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            
    # Save overall evaluation summary
    summary_file = os.path.join(output_dir, "pymupdf_evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2)

if __name__ == "__main__":
    main()
