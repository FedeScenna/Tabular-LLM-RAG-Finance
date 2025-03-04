import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np

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

def batch_process_texts(texts, embeddings_model, batch_size=32):
    """Process texts in batches to generate embeddings."""
    embeddings_list = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", unit="batch"):
        batch = texts[i:i + batch_size]
        batch_embeddings = embeddings_model.embed_documents(batch)
        embeddings_list.extend(batch_embeddings)
    
    return embeddings_list

def process_directory(input_dir, output_dir, num_processes=None):
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
    
    # Create vector store using Ollama embeddings with batch processing
    embeddings = OllamaEmbeddings(
        model="llama3.2:1b",
        show_progress=True,
    )
    
    # Process embeddings in batches
    embeddings_list = batch_process_texts(chunks, embeddings)
    
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
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir, args.processes) 