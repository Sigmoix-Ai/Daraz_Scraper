import os
import logging
from typing import List, Dict, Tuple
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import numpy as np


load_dotenv()


client = OpenAI(api_key=os.getenv('openai_api'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAGPipeline:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.embeddings = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.api_tracker = None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text and metadata from a PDF file with enhanced processing."""
        try:
            reader = PdfReader(pdf_path)
            full_text = []

           
            if reader.metadata:
                metadata = []
                for key, value in reader.metadata.items():
                    if value and str(value).strip():
                        metadata.append(f"{key}: {value}")
                if metadata:
                    full_text.append("Document Metadata:")
                    full_text.extend(metadata)
                    full_text.append("-" * 50)

            # Process each page
            for page_num, page in enumerate(reader.pages, 1):
                # Extract main text with page numbers
                text = page.extract_text()
                if text:
                    full_text.append(f"\nPage {page_num}:")
                    full_text.append(text)

                # Try to extract links if available
                try:
                    if '/Annots' in page and isinstance(page['/Annots'], list):
                        links = []
                        for annot in page['/Annots']:
                            if annot.get_object().get('/Subtype') == '/Link':
                                if '/A' in annot.get_object():
                                    action = annot.get_object()['/A']
                                    if '/URI' in action:
                                        links.append(f"Link: {action['/URI']}")
                        if links:
                            full_text.append("\nLinks found on this page:")
                            full_text.extend(links)
                except Exception as e:
                    logger.debug(f"Could not extract links from page {page_num}: {str(e)}")

                # Try to detect and format tabular content
                try:
                    lines = text.split('\n')
                    for i, line in enumerate(lines):
                        if '  ' in line:  # Look for multiple spaces indicating possible table
                            # Add table formatting if it looks like tabular data
                            cells = [cell.strip() for cell in line.split('  ') if cell.strip()]
                            if len(cells) > 1:
                                lines[i] = ' | '.join(cells)
                    text = '\n'.join(lines)
                    full_text.append(text)
                except Exception as e:
                    logger.debug(f"Could not process tables on page {page_num}: {str(e)}")

                full_text.append("-" * 50)  # Page separator

            return '\n'.join(full_text)

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def _extract_table_structure(self, page) -> str:
        """Helper method to extract table-like structures from a page"""
        try:
            # Get raw text
            text = page.extract_text()
            
            # Look for common table indicators
            lines = text.split('\n')
            table_lines = []
            in_table = False
            
            for line in lines:
                # Check for table-like patterns (multiple spaces or tabs between words)
                if '    ' in line or '\t' in line:
                    if not in_table:
                        in_table = True
                        table_lines.append('-' * 50)  # Table separator
                    table_lines.append(line)
                elif in_table:
                    in_table = False
                    table_lines.append('-' * 50)  # Table separator
            
            return '\n'.join(table_lines) if table_lines else ""
            
        except Exception as e:
            logger.error(f"Error extracting table structure: {str(e)}")
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
            
            # Track the embedding call if we have an API tracker
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
    pipeline = SimpleRAGPipeline()

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
        output_file = os.path.join(output_dir, f"pypdf2_results_{len(evaluation_results)}.txt")
        
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
    summary_file = os.path.join(output_dir, "pypdf2_evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(evaluation_results, f, indent=2)

if __name__ == "__main__":
    main()
