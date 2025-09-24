from typing import List, Dict, Optional
import os
import logging
import json
import numpy as np
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()

client = OpenAI(api_key=os.getenv('openai_api'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PageNode:
    title: str
    node_id: str
    start_index: int
    end_index: int
    summary: str
    nodes: List['PageNode']
    content: Optional[str] = None

class PageIndexRAGPipeline:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.embeddings = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.page_tree = None
        self.full_text = None  # Store the full text for compatibility
        
    def _generate_section_title(self, content: str) -> str:
        """Generate a title for a section using GPT."""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a document structure expert. Generate a concise title for this section."},
                    {"role": "user", "content": f"Generate a short, descriptive title (3-7 words) for this content:\n\n{content[:500]}..."}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating section title: {str(e)}")
            return "Untitled Section"

    def _generate_section_summary(self, content: str) -> str:
        """Generate a summary for a section using GPT."""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a document summarization expert. Create a brief, informative summary."},
                    {"role": "user", "content": f"Summarize this content in 2-3 sentences:\n\n{content}"}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating section summary: {str(e)}")
            return ""

    def _build_tree_structure(self, content: str) -> PageNode:
        """Build a hierarchical tree structure from the document content."""
        try:
            if not content.strip():
                # Return an empty root node if content is empty
                return PageNode(
                    title="Empty Document",
                    node_id="0000",
                    start_index=0,
                    end_index=0,
                    summary="",
                    nodes=[],
                    content=""
                )
                
            # First, ask GPT to identify major sections
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a document structure expert. Identify major sections in this document."},
                    {"role": "user", "content": f"Identify the main sections in this document and their approximate locations:\n\n{content}"}
                ]
            )
            
            # Parse the sections and create nodes
            sections = []
            current_pos = 0
            
            # Create root node
            root = PageNode(
                title="Document Root",
                node_id="0000",
                start_index=0,
                end_index=len(content),
                summary=self._generate_section_summary(content[:1000]),
                nodes=[],
                content=content
            )
            
            # Process each major section
            section_texts = content.split('\n\n')  # Simple section splitting
            for i, section_text in enumerate(section_texts):
                if len(section_text.strip()) < 50:  # Skip very small sections
                    continue
                    
                node = PageNode(
                    title=self._generate_section_title(section_text),
                    node_id=f"{i+1:04d}",
                    start_index=current_pos,
                    end_index=current_pos + len(section_text),
                    summary=self._generate_section_summary(section_text),
                    nodes=[],
                    content=section_text
                )
                root.nodes.append(node)
                current_pos += len(section_text) + 2  # +2 for '\n\n'
                
            return root
            
        except Exception as e:
            logger.error(f"Error building tree structure: {str(e)}")
            return None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text and build PageIndex tree structure from PDF."""
        try:
            # Use PyMuPDF for initial text extraction
            import fitz
            doc = fitz.open(pdf_path)
            full_text = []
            
            # Extract text from each page
            for page in doc:
                text = page.get_text()
                if text.strip():
                    full_text.append(text)
                    
            content = "\n\n".join(full_text)
            
            # Store the full text for compatibility with benchmark
            self.full_text = content
            
            # Build the PageIndex tree structure
            self.page_tree = self._build_tree_structure(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def _search_tree(self, query: str, node: PageNode, results: List[Dict], depth: int = 0):
        """Recursively search the tree structure for relevant content."""
        try:
            # Check relevance of current node
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a document search expert. Score the relevance of this section to the query."},
                    {"role": "user", "content": f"Query: {query}\n\nSection Title: {node.title}\nSection Summary: {node.summary}\n\nScore the relevance from 0-100:"}
                ]
            )
            
            relevance_score = float(response.choices[0].message.content.strip().split()[0])
            
            if relevance_score > 50:  # If somewhat relevant
                results.append({
                    "score": relevance_score / 100,
                    "text": node.content if node.content else node.summary,
                    "title": node.title,
                    "node_id": node.node_id
                })
            
            # Recursively search child nodes
            if depth < 3:  # Limit depth to prevent too much recursion
                for child in node.nodes:
                    self._search_tree(query, child, results, depth + 1)
                    
        except Exception as e:
            logger.error(f"Error searching tree: {str(e)}")

    def chunk_text(self, text: str) -> List[str]:
        """Compatibility method for benchmark system.
        Instead of traditional chunking, we use our tree structure nodes as chunks."""
        if not self.page_tree:
            return []
            
        # Collect all node contents as chunks
        chunks = []
        def collect_chunks(node: PageNode):
            if node.content:
                chunks.append(node.content)
            for child in node.nodes:
                collect_chunks(child)
                
        collect_chunks(self.page_tree)
        self.chunks = chunks
        return chunks

    def build_retrieval_index(self, chunks: List[str]):
        """Compatibility method for benchmark system.
        Instead of building embeddings, we'll just store the chunks for later use."""
        self.chunks = chunks
        # If we don't have a tree yet (which is possible in the benchmark flow),
        # build one from the chunks
        if not self.page_tree:
            combined_text = "\n\n".join(chunks)
            self.page_tree = self._build_tree_structure(combined_text)

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant sections using tree-based search."""
        if not self.page_tree:
            return []
            
        results = []
        self._search_tree(query, self.page_tree, results)
        
        # Sort by relevance score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top k results in the format expected by the benchmark
        return [{"score": r["score"], "text": r["text"]} for r in results[:k]]

def main():
    pdf_path = "Test.pdf"
    pipeline = PageIndexRAGPipeline()
    
    print("Extracting text and building PageIndex tree...")
    full_text = pipeline.extract_text_from_pdf(pdf_path)
    if not full_text:
        logger.error("No text extracted from PDF.")
        return
    
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
            print(f"Section: {result['title']}")
            print("-" * 40)
            print(result['text'][:300] + "..." if len(result['text']) > 300 else result['text'])

if __name__ == "__main__":
    main()