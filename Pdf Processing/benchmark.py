import time
import psutil
import os
import tracemalloc
import time
import psutil
import os
import tracemalloc
from PyMuPdf import MuPdfRAGPipeline
from PdfPlumber import PDFProcessor
from PyPdf import SimpleRAGPipeline
from Tesseract import TesseractRAGPipeline
from pageindex import PageIndexRAGPipeline
from docling import DoclingRAGPipeline

def evaluate_answer(generated_answer: str, ground_truth: str, client) -> dict:
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
        print(f"Error in evaluation: {str(e)}")
        return {
            "accuracy": 0,
            "completeness": 0,
            "relevance": 0,
            "explanation": f"Error in evaluation: {str(e)}"
        }

def measure_resources(func, api_tracker=None):
    # Start memory tracking
    tracemalloc.start()
    process = psutil.Process()
    
    # Record start metrics
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    start_cpu = process.cpu_percent()
    
    # Execute function
    result = func()
    
    # Record end metrics
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024
    end_cpu = process.cpu_percent()
    
    # Get memory peak
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    metrics = {
        "execution_time": end_time - start_time,
        "memory_used": end_memory - start_memory,
        "peak_memory": peak / 1024 / 1024,  # Convert to MB
        "cpu_usage": (start_cpu + end_cpu) / 2  # Average CPU usage
    }
    
    return result, metrics

def evaluate_pipeline_accuracy(pipeline, pdf_path):
    """Evaluate the accuracy of a RAG pipeline using predefined test cases."""
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    
    # Load OpenAI client
    load_dotenv()
    client = OpenAI(api_key=os.getenv('openai_api'))
    
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
    
    # First extract and process the text
    if hasattr(pipeline, 'process_pdf'):
        full_text = pipeline.process_pdf(pdf_path)
    else:
        full_text = pipeline.extract_text_from_pdf(pdf_path)
    
    if not full_text:
        return 0
    
    # Build the index
    chunks = pipeline.chunk_text(full_text)
    pipeline.build_retrieval_index(chunks)
    
    # Run test cases and calculate average accuracy
    total_accuracy = 0
    total_completeness = 0
    total_relevance = 0
    total_cases = len(test_cases)
    
    for test_case in test_cases:
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]
            
        # Get response from pipeline
        results = pipeline.retrieve(question)
        context = "\n".join([r['text'] for r in results])
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the question strictly based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            generated_answer = response.choices[0].message.content
            
            # Use the evaluate_answer function which is defined at module level
            evaluation = evaluate_answer(generated_answer, ground_truth, client)
                
            total_accuracy += evaluation['accuracy']
            total_completeness += evaluation['completeness']
            total_relevance += evaluation['relevance']
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            continue
    
    if total_cases > 0:
        # Return average of accuracy, completeness, and relevance
        return (total_accuracy + total_completeness + total_relevance) / (3 * total_cases)
    return 0

def benchmark_pdf_processing(pdf_path):
    results = {}
    
    # Test PyMuPDF
    print("\nTesting PyMuPDF...")
    pymupdf_pipeline = MuPdfRAGPipeline()
    _, metrics = measure_resources(lambda: pymupdf_pipeline.extract_text_from_pdf(pdf_path))
    accuracy = evaluate_pipeline_accuracy(pymupdf_pipeline, pdf_path)
    metrics['accuracy'] = accuracy
    results['PyMuPDF'] = metrics
    
    # Test pdfplumber
    print("\nTesting pdfplumber...")
    pdfplumber_pipeline = PDFProcessor()
    _, metrics = measure_resources(lambda: pdfplumber_pipeline.process_pdf(pdf_path))
    accuracy = evaluate_pipeline_accuracy(pdfplumber_pipeline, pdf_path)
    metrics['accuracy'] = accuracy
    results['pdfplumber'] = metrics
    
    # Test PyPDF2
    print("\nTesting PyPDF2...")
    pypdf_pipeline = SimpleRAGPipeline()
    _, metrics = measure_resources(lambda: pypdf_pipeline.extract_text_from_pdf(pdf_path))
    accuracy = evaluate_pipeline_accuracy(pypdf_pipeline, pdf_path)
    metrics['accuracy'] = accuracy
    results['PyPDF2'] = metrics
    
    # Test Tesseract
    print("\nTesting Tesseract...")
    tesseract_pipeline = TesseractRAGPipeline()
    _, metrics = measure_resources(lambda: tesseract_pipeline.extract_text_from_pdf(pdf_path))
    accuracy = evaluate_pipeline_accuracy(tesseract_pipeline, pdf_path)
    metrics['accuracy'] = accuracy
    results['Tesseract'] = metrics
    
    # Test PageIndex
    print("\nTesting PageIndex...")
    pageindex_pipeline = PageIndexRAGPipeline()
    _, metrics = measure_resources(lambda: pageindex_pipeline.extract_text_from_pdf(pdf_path))
    accuracy = evaluate_pipeline_accuracy(pageindex_pipeline, pdf_path)
    metrics['accuracy'] = accuracy
    results['PageIndex'] = metrics
    
    # Test Docling
    print("\nTesting Docling...")
    docling_pipeline = DoclingRAGPipeline()
    _, metrics = measure_resources(lambda: docling_pipeline.extract_text_from_pdf(pdf_path))
    accuracy = evaluate_pipeline_accuracy(docling_pipeline, pdf_path)
    metrics['accuracy'] = accuracy
    results['Docling'] = metrics
    
    return results

def print_results(results):
    print("\nPerformance Comparison:")
    print("-" * 80)
    headers = [
        ('Library', 15),
        ('Time (s)', 12),
        ('Memory (MB)', 15),
        ('Peak Mem (MB)', 15),
        ('CPU %', 10),
        ('Accuracy %', 10)
    ]
    
    # Print headers
    header_line = ""
    for header, width in headers:
        header_line += f"{header:<{width}}"
    print(header_line)
    print("-" * 80)
    
    # Print metrics
    for lib, metrics in results.items():
        line = (
            f"{lib:<15}"
            f"{metrics['execution_time']:<12.2f}"
            f"{metrics['memory_used']:<15.2f}"
            f"{metrics['peak_memory']:<15.2f}"
            f"{metrics['cpu_usage']:<10.2f}"
            f"{metrics['accuracy']:<10.2f}"
        )
        print(line)
    print("-" * 80)

if __name__ == "__main__":
    pdf_path = "Test.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found!")
        exit(1)
    
    print(f"Running benchmarks on {pdf_path}...")
    results = benchmark_pdf_processing(pdf_path)
    print_results(results)