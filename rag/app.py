import streamlit as st
import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
import torch

# Initialize session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def check_gpu():
    """Check if GPU is available and return information about it."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        return True, f"{device_name} (CUDA {cuda_version})"
    return False, "Not available"

def load_vector_store(embeddings_dir):
    """Load the vector store from disk."""
    # Use OllamaEmbeddings with the same model used for creating embeddings
    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    vector_store = FAISS.load_local(embeddings_dir, embeddings, allow_dangerous_deserialization=True)
    return vector_store

def get_conversation_chain(vector_store):
    """Create a conversation chain for RAG."""
    # Initialize Ollama with llama3.2 model
    llm = Ollama(model="llama3.1:8b", temperature=0.7)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain

# Set page configuration
st.set_page_config(page_title="Chat with your PDFs", layout="wide")
st.title("📚 Chat with your PDFs using Llama 3.2")

# Add a note about Ollama
has_gpu, gpu_info = check_gpu()
gpu_status = f"🔥 GPU Acceleration: {gpu_info}" if has_gpu else "⚠️ GPU not detected, using CPU"

st.info(f"This application uses Llama 3.2 (1B parameters) via Ollama for local inference. {gpu_status}")

# Sidebar for embeddings directory selection
with st.sidebar:
    st.subheader("Select Embeddings Directory")
    
    # Get only directories in the current path
    all_items = os.listdir('.')
    directories = [item for item in all_items if os.path.isdir(item)]
    
    # Add option to enter custom path
    directories.insert(0, "Custom path")
    
    selection = st.selectbox(
        "Select a folder containing embeddings:",
        directories,
        index=0,
        help="Choose a folder where your embeddings are stored"
    )
    
    # If custom path is selected, show text input
    if selection == "Custom path":
        embeddings_dir = st.text_input(
            "Enter the path to your embeddings directory:",
            value="embeddings",
            help="This should be the directory where you saved your embeddings using process_documents.py"
        )
    else:
        embeddings_dir = selection
        
    # Display the selected path
    st.caption(f"Selected path: {os.path.abspath(embeddings_dir)}")
    
    model_name = st.selectbox(
        "Select Ollama Model:",
        ["llama3.1:8b", "llama3.2:1b"],
        index=0,
        help="Select the model to use for answering questions. Make sure it's available in your Ollama installation."
    )
    
    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    
    if st.button("Load Embeddings"):
        if os.path.exists(embeddings_dir):
            with st.spinner("Loading embeddings..."):
                try:
                    vector_store = load_vector_store(embeddings_dir)
                    # Update the conversation with the selected model and temperature
                    llm = Ollama(model=model_name, temperature=temperature)
                    memory = ConversationBufferMemory(
                        memory_key='chat_history',
                        return_messages=True,
                        output_key='answer'
                    )
                    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                        memory=memory,
                        return_source_documents=True
                    )
                    st.success("Embeddings loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading embeddings: {str(e)}")
        else:
            st.error(f"Directory {embeddings_dir} does not exist. Please process your documents first using process_documents.py")

# Chat interface
if st.session_state.conversation is not None:
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        with st.spinner("Thinking... (This might take a moment as processing is done locally)"):
            try:
                response = st.session_state.conversation({
                    'question': user_question
                })
                st.session_state.chat_history.append(("user", user_question))
                st.session_state.chat_history.append(("assistant", response['answer']))
                
                # Display source documents if available
                if 'source_documents' in response and response['source_documents']:
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(response['source_documents']):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(f"```\n{doc.page_content}\n```")
                            st.markdown("---")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    # Display chat history
    for i, (role, message) in enumerate(reversed(st.session_state.chat_history)):
        if i >= 10:  # Only show the last 10 messages to avoid clutter
            break
        if role == "user":
            st.write(f"👤 **You:** {message}")
        else:
            st.write(f"🤖 **Assistant:** {message}")
else:
    st.info("Please load your embeddings to start chatting!") 