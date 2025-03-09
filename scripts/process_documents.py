#!/usr/bin/env python
"""
Document Processing Script

This script processes PDF documents and creates embeddings for use with the financial assistant.
"""

import argparse
from financial_assistant.document_processing.processor import process_directory


def main():
    """Main function to run the document processor from the command line."""
    parser = argparse.ArgumentParser(description="Process PDF documents and create embeddings")
    parser.add_argument("--input_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", required=True, help="Directory to save embeddings")
    parser.add_argument("--model", default="llama3.1:8b", help="Model to use for embeddings")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes to use (default: CPU count - 1)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for embedding generation (default: auto)")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage even if GPU is available")
    args = parser.parse_args()
    
    process_directory(
        args.input_dir, 
        args.output_dir, 
        args.model,
        args.processes, 
        args.batch_size, 
        args.force_cpu
    )


if __name__ == "__main__":
    main() 