import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
import torch
import gc
import pynvml

def check_gpu():
    """Check if GPU is available and return information about it."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return False, None, None
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        device_name = pynvml.nvmlDeviceGetName(handle)
        
        total_memory = info.total / 1024**3  # Convert to GB
        free_memory = info.free / 1024**3    # Convert to GB
        
        print(f"GPU: {device_name}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total GPU Memory: {total_memory:.2f} GB")
        print(f"Free GPU Memory: {free_memory:.2f} GB")
        
        return True, device_name, free_memory
    except Exception as e:
        print(f"Error getting GPU information: {str(e)}")
        print("Using CUDA without detailed GPU information.")
        return True, "Unknown", None
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

def process_pdf(pdf_path):
    """Process a single PDF document and return its text content."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in tqdm(pdf_reader.pages, desc=f"Processing {os.path.basename(pdf_path)}", leave=False):
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return ""

def batch_process_texts(texts, embeddings_model, batch_size=32, device="cpu"):
    """Process texts in batches to generate embeddings."""
    embeddings_list = []
    
    # Adjust batch size based on device
    if device == "cuda":
        # Use smaller batches for GPU to avoid OOM errors
        batch_size = min(batch_size, 16)
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", unit="batch"):
        batch = texts[i:i + batch_size]
        
        # Clear CUDA cache between batches if using GPU
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            
        batch_embeddings = embeddings_model.embed_documents(batch)
        embeddings_list.extend(batch_embeddings)
    
    return embeddings_list

def process_directory(input_dir, output_dir, num_processes=None, batch_size=None, force_cpu=False):
    """Process all PDFs in a directory and save their embeddings."""
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files in the input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process PDFs in parallel
    pdf_paths = [os.path.join(input_dir, pdf_file) for pdf_file in pdf_files]
    
    with Pool(processes=num_processes) as pool:
        texts = list(tqdm(
            pool.imap(process_pdf, pdf_paths),
            total=len(pdf_paths),
            desc="Processing PDFs",
            unit="file"
        ))
    
    # Combine all texts
    all_text = "\n\n".join(filter(None, texts))
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    print("Splitting text into chunks...")
    chunks = text_splitter.split_text(all_text)
    print(f"Created {len(chunks)} text chunks")
    
    # Check GPU availability and get information
    has_gpu, gpu_name, free_memory = check_gpu()
    
    # Set device based on availability and user preference
    device = "cpu" if force_cpu or not has_gpu else "cuda"
    torch_device = torch.device(device)
    
    # Determine optimal batch size if not specified
    if batch_size is None:
        if device == "cuda" and free_memory is not None:
            # Heuristic: 1GB of VRAM can handle ~64 embeddings at once for most models
            # Adjust based on your specific model and embedding size
            batch_size = max(4, min(64, int(free_memory * 64)))
        else:
            batch_size = 32
    
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    
    # Create Ollama embeddings instance
    embeddings = OllamaEmbeddings(
        model="llama3.1:8b"  # Ensure this model supports GPU
    )
    
    # Process embeddings in batches
    embeddings_list = batch_process_texts(chunks, embeddings, batch_size=batch_size, device=device)
    
    # Create FAISS index
    print("Creating FAISS index...")
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(chunks, embeddings_list)),
        embedding=embeddings
    )
    
    # Save the vector store
    vector_store.save_local(output_dir)
    print(f"Embeddings saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDF documents and create embeddings")
    parser.add_argument("--input_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", required=True, help="Directory to save embeddings")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes to use (default: CPU count - 1)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for embedding generation (default: auto)")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage even if GPU is available")
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir, args.processes, args.batch_size, args.force_cpu)