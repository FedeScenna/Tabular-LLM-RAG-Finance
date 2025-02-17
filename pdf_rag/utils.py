import gradio as gr
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import re



def process_pdfs_from_folder(folder_path):
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return None, None

    all_chunks = []
    
    # Iterate over all PDF files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)

            loader = PyMuPDFLoader(pdf_path)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=100
            )
            chunks = text_splitter.split_documents(data)
            all_chunks.extend(chunks)  # Store all chunks

    if not all_chunks:
        return None, None  # No valid PDFs found

    # Create vector database with all PDFs' text
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    vectorstore = Chroma.from_documents(
        documents=all_chunks, embedding=embeddings, persist_directory="./chroma_db"
    )
    retriever = vectorstore.as_retriever()

    return vectorstore, retriever


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"

    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    response_content = response["message"]["content"]

    # Remove content between <think> and </think> tags
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()

    return final_answer


def rag_chain(question, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_content = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_content)