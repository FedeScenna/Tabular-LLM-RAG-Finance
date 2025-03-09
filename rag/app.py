import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain

# Initialize session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def load_vector_store(embeddings_dir):
    """Load the vector store from disk."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(embeddings_dir, embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    """Create a conversation chain for RAG."""
    # Initialize Ollama with llama3.2 model
    llm = Ollama(model="llama3.2:1b", temperature=0.7)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
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
st.info("This application uses Llama 3.2 (1B parameters) via Ollama for local inference.")

# Sidebar for embeddings directory selection
with st.sidebar:
    st.subheader("Select Embeddings Directory")
    embeddings_dir = st.text_input(
        "Enter the path to your embeddings directory:",
        value="embeddings",
        help="This should be the same directory where you saved your embeddings using process_documents.py"
    )
    
    if st.button("Load Embeddings"):
        if os.path.exists(embeddings_dir):
            with st.spinner("Loading embeddings..."):
                try:
                    vector_store = load_vector_store(embeddings_dir)
                    st.session_state.conversation = get_conversation_chain(vector_store)
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
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.write(f"👤 **You:** {message}")
        else:
            st.write(f"🤖 **Assistant:** {message}")
else:
    st.info("Please load your embeddings to start chatting!") 