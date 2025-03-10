import requests
import traceback
import torch
import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

def check_ollama_server():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            available_models = [model["name"] for model in response.json()["models"]]
            return True, available_models
        return False, []
    except requests.exceptions.ConnectionError:
        return False, []

def check_gpu():
    """Check if GPU is available and return information about it."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        return True, f"{device_name} (CUDA {cuda_version})"
    return False, "Not available"

def load_vector_store(embeddings_dir, model_name="llama3:8b"):
    """Load the vector store from disk."""
    try:
        # Important: We need to use the SAME embedding model that was used to create the embeddings
        # This is critical to avoid dimension mismatch errors
        
        # First try to load the index to determine what embedding model was used
        import pickle
        import os
        
        # Try to find the original model used for embeddings
        try:
            # Check if there's a metadata file that might contain model info
            metadata_path = os.path.join(embeddings_dir, "index_metadata.pickle")
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                    if "model" in metadata:
                        original_model = metadata["model"]
                        st.info(f"Using original embedding model: {original_model}")
                        embeddings = OllamaEmbeddings(model=original_model)
                    else:
                        # If no model info, use the model that created the embeddings
                        # This is "llama3.1:8b" based on user information
                        embeddings = OllamaEmbeddings(model="llama3.1:8b")
                        st.warning("Using default embedding model: llama3.1:8b")
            else:
                # If no metadata file, use the model that created the embeddings
                embeddings = OllamaEmbeddings(model="llama3.1:8b")
                st.warning("Using default embedding model: llama3.1:8b")
        except Exception as e:
            # If any error occurs during metadata loading, use the model that created the embeddings
            st.warning(f"Error loading embedding metadata: {str(e)}. Using default model: llama3.1:8b")
            embeddings = OllamaEmbeddings(model="llama3.1:8b")
        
        # Load the vector store with the determined embeddings
        vector_store = FAISS.load_local(embeddings_dir, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        st.session_state.debug_info = f"Error in load_vector_store: {str(e)}\n{traceback.format_exc()}"
        raise

def get_conversation_chain(vector_store, model_name="llama3:8b", temperature=0.7):
    """Create a conversation chain for RAG."""
    try:
        # Initialize Ollama with the selected model
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Define the custom prompt template for financial assistance
        financial_template = """
        You are an expert financial assistant specializing in providing concise and accurate answers about companies in the S&P 500 index. Use the provided context to generate responses. If the context does not contain relevant information, state that you do not have enough data rather than making assumptions. Ensure responses are factual, clear, and to the point.
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Question: {question}
        
        Answer:"""
        
        # Create the prompt template
        PROMPT = PromptTemplate(
            template=financial_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        llm = Ollama(
            model=model_name, 
            temperature=temperature,
            callback_manager=callback_manager
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            verbose=True  # Enable verbose mode for debugging
        )
        return conversation_chain
    except Exception as e:
        st.session_state.debug_info = f"Error in get_conversation_chain: {str(e)}\n{traceback.format_exc()}"
        raise
