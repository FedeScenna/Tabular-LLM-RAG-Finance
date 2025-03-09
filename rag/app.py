import streamlit as st
import os
import sys
import traceback
from utils import check_ollama_server, check_gpu, load_vector_store, get_conversation_chain
# Initialize session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = ""

# Set page configuration
st.set_page_config(page_title="Chat with your PDFs", layout="wide")
st.title("📚 Chat with your PDFs using Llama")

# Check Ollama server
ollama_running, available_models = check_ollama_server()
if not ollama_running:
    st.error("⚠️ Ollama server is not running. Please start Ollama with 'ollama serve' in a terminal.")
    st.stop()

# Add a note about Ollama
has_gpu, gpu_info = check_gpu()
gpu_status = f"🔥 GPU Acceleration: {gpu_info}" if has_gpu else "⚠️ GPU not detected, using CPU"

st.info(f"This application uses Llama via Ollama for local inference. {gpu_status}")

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
    
    # Show available models from Ollama
    if available_models:
        model_options = available_models
    else:
        model_options = ["llama3:8b", "llama3:70b", "llama2", "mistral"]
    
    model_name = st.selectbox(
        "Select Ollama Model:",
        model_options,
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
                    vector_store = load_vector_store(embeddings_dir, model_name)
                    # Update the conversation with the selected model and temperature
                    st.session_state.conversation = get_conversation_chain(
                        vector_store, 
                        model_name=model_name, 
                        temperature=temperature
                    )
                    st.success("Embeddings loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading embeddings: {str(e)}")
                    with st.expander("Debug Information"):
                        st.code(st.session_state.debug_info)
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
            except AssertionError as e:
                # This is likely a dimension mismatch error
                error_msg = "Dimension mismatch error: The embedding model used for querying doesn't match the one used for creating the index."
                st.error(error_msg)
                st.warning("Try reloading with the model 'llama3.2:1b' which was likely used to create the embeddings.")
                
                # Capture and display the full traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error_details = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                with st.expander("Error Details"):
                    st.code(error_details)
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                # Capture and display the full traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error_details = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                with st.expander("Error Details"):
                    st.code(error_details)

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

# Add debug expander at the bottom
with st.expander("Advanced Settings & Debug"):
    st.write("If you're experiencing issues, check the following:")
    st.write("1. Make sure Ollama is running (`ollama serve` in terminal)")
    st.write("2. Verify the model you selected is available in Ollama (`ollama list`)")
    st.write("3. Check that the embeddings directory contains valid FAISS index files")
    
    if st.button("Check Ollama Status"):
        ollama_running, models = check_ollama_server()
        if ollama_running:
            st.success(f"✅ Ollama is running with {len(models)} models available")
            st.write("Available models:", ", ".join(models))
        else:
            st.error("❌ Ollama server is not running or not responding") 