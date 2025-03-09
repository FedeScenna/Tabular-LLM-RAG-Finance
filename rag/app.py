import streamlit as st
import os
import sys
import traceback
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import torch
import requests
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pandas as pd
from PIL import Image
import io

# Import stock data handler
from stock_data import StockDataHandler, process_stock_query

# Initialize session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = ""
if 'stock_handler' not in st.session_state:
    st.session_state.stock_handler = None
if 'query_mode' not in st.session_state:
    st.session_state.query_mode = "rag"  # Default to RAG mode

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

def load_stock_data(csv_path):
    """Load stock data from CSV file."""
    try:
        stock_handler = StockDataHandler(csv_path)
        return stock_handler
    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        return None

def is_stock_query(query):
    """Determine if a query is related to stock data."""
    query_lower = query.lower()
    stock_keywords = [
        "stock", "price", "ticker", "share", "market", "trading", "investor", 
        "dividend", "return", "performance", "technical", "indicator", "rsi", 
        "macd", "bollinger", "volatility", "volume", "compare"
    ]
    
    # Check for stock ticker patterns (all caps 1-5 letters)
    words = query.split()
    has_ticker = any(word.isupper() and word.isalpha() and 1 <= len(word) <= 5 for word in words)
    
    # Check for stock-related keywords
    has_keywords = any(keyword in query_lower for keyword in stock_keywords)
    
    return has_ticker or has_keywords

# Set page configuration
st.set_page_config(page_title="Financial Assistant", layout="wide")
st.title("📊 Financial Assistant")

# Check Ollama server
ollama_running, available_models = check_ollama_server()
if not ollama_running:
    st.error("⚠️ Ollama server is not running. Please start Ollama with 'ollama serve' in a terminal.")
    st.stop()

# Add a note about Ollama
has_gpu, gpu_info = check_gpu()
gpu_status = f"🔥 GPU Acceleration: {gpu_info}" if has_gpu else "⚠️ GPU not detected, using CPU"

st.info(f"This application uses Llama via Ollama for local inference. {gpu_status}")

# Sidebar for configuration
with st.sidebar:
    st.subheader("Configuration")
    
    # Tabs for different data sources
    data_source = st.radio(
        "Select Data Source",
        ["Document Embeddings", "Stock Price Data", "Both"],
        index=2,
        help="Choose which data source to use for answering questions"
    )
    
    # Document embeddings configuration
    if data_source in ["Document Embeddings", "Both"]:
        st.subheader("Document Embeddings")
        
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
    
    # Stock data configuration
    if data_source in ["Stock Price Data", "Both"]:
        st.subheader("Stock Price Data")
        
        stock_data_path = st.text_input(
            "Path to stock price CSV file:",
            value="../data/price_data.csv",
            help="Path to the CSV file containing stock price history"
        )
    
    # Model configuration
    st.subheader("Model Configuration")
    
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
    
    # Load data button
    if st.button("Load Data"):
        with st.spinner("Loading data..."):
            # Load document embeddings if selected
            if data_source in ["Document Embeddings", "Both"]:
                if os.path.exists(embeddings_dir):
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
            
            # Load stock data if selected
            if data_source in ["Stock Price Data", "Both"]:
                if os.path.exists(stock_data_path):
                    try:
                        st.session_state.stock_handler = load_stock_data(stock_data_path)
                        if st.session_state.stock_handler:
                            st.success(f"Stock data loaded successfully! Found {len(st.session_state.stock_handler.tickers)} tickers.")
                    except Exception as e:
                        st.error(f"Error loading stock data: {str(e)}")
                else:
                    st.error(f"File {stock_data_path} does not exist.")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Chat interface
    st.subheader("Chat with Financial Assistant")
    
    # Check if any data is loaded
    data_loaded = (st.session_state.conversation is not None or st.session_state.stock_handler is not None)
    
    if data_loaded:
        user_question = st.chat_input("Ask a question about finances, companies, or stock prices...")
        if user_question:
            # Determine if this is a stock-related query
            is_stock = is_stock_query(user_question)
            
            with st.spinner("Thinking... (This might take a moment as processing is done locally)"):
                try:
                    # Process stock query
                    if is_stock and st.session_state.stock_handler:
                        response_text = process_stock_query(user_question, st.session_state.stock_handler)
                        st.session_state.chat_history.append(("user", user_question))
                        st.session_state.chat_history.append(("assistant", response_text))
                        st.session_state.query_mode = "stock"
                    
                    # Process RAG query
                    elif st.session_state.conversation:
                        response = st.session_state.conversation({
                            'question': user_question
                        })
                        st.session_state.chat_history.append(("user", user_question))
                        st.session_state.chat_history.append(("assistant", response['answer']))
                        st.session_state.query_mode = "rag"
                        
                        # Store source documents for display
                        if 'source_documents' in response and response['source_documents']:
                            st.session_state.last_sources = response['source_documents']
                    
                    # No appropriate handler
                    else:
                        if is_stock:
                            error_msg = "Stock data is not loaded. Please load stock data to answer stock-related questions."
                        else:
                            error_msg = "Document embeddings are not loaded. Please load embeddings to answer document-related questions."
                        st.error(error_msg)
                
                except AssertionError as e:
                    # This is likely a dimension mismatch error
                    error_msg = "Dimension mismatch error: The embedding model used for querying doesn't match the one used for creating the index."
                    st.error(error_msg)
                    st.warning("Try reloading with the model 'llama3.1:8b' which was used to create the embeddings.")
                    
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
        st.info("Please load data using the sidebar to start chatting!")

with col2:
    # Information panel
    st.subheader("Additional Information")
    
    # Show different information based on the last query mode
    if st.session_state.query_mode == "rag" and hasattr(st.session_state, 'last_sources'):
        with st.expander("Source Documents", expanded=True):
            for i, doc in enumerate(st.session_state.last_sources):
                st.markdown(f"**Source {i+1}:**")
                st.markdown(f"```\n{doc.page_content}\n```")
                st.markdown("---")
    
    elif st.session_state.query_mode == "stock" and st.session_state.stock_handler:
        # Show stock visualization options
        st.subheader("Stock Visualization")
        
        # Get all tickers
        all_tickers = st.session_state.stock_handler.tickers
        
        # Extract ticker from last query if available
        last_query = st.session_state.chat_history[-2][1] if len(st.session_state.chat_history) >= 2 else ""
        from stock_data import extract_ticker_from_query
        default_ticker = extract_ticker_from_query(last_query) or (all_tickers[0] if all_tickers else "")
        
        # Ticker selection
        ticker = st.selectbox("Select Ticker", all_tickers, index=all_tickers.index(default_ticker) if default_ticker in all_tickers else 0)
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime("2022-12-31"))
        
        # Chart type
        chart_type = st.selectbox("Chart Type", ["Price History", "Technical Indicators"])
        
        # Generate and display chart
        if st.button("Generate Chart"):
            with st.spinner("Generating chart..."):
                if chart_type == "Price History":
                    chart_buf = st.session_state.stock_handler.plot_price_history(
                        ticker, 
                        start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d')
                    )
                    if chart_buf:
                        st.image(Image.open(chart_buf), caption=f"{ticker} Price History")
                    else:
                        st.warning(f"No price data available for {ticker} in the selected date range.")
                
                elif chart_type == "Technical Indicators":
                    # Get technical indicators
                    indicators = st.session_state.stock_handler.get_technical_indicators(
                        ticker, 
                        start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if indicators:
                        st.write(f"Technical indicators for {ticker} as of {indicators['date'].strftime('%Y-%m-%d')}:")
                        st.write(f"RSI: {indicators['rsi']:.2f}")
                        st.write(f"MACD: {indicators['macd']:.2f}")
                        st.write(f"Bollinger Bands: Low=${indicators['bb_low']:.2f}, Mid=${indicators['bb_mid']:.2f}, High=${indicators['bb_high']:.2f}")
                        st.write(f"Volatility: {indicators['volatility']:.4f}")
                        st.write(f"ATR: {indicators['atr']:.4f}")
                    else:
                        st.warning(f"No technical indicator data available for {ticker} in the selected date range.")

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