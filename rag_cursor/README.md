# PDF RAG Chat Application

This application allows users to chat with their PDF documents using RAG (Retrieval Augmented Generation) technology.

## Features
- Upload and process PDF documents
- Chat interface to ask questions about your documents
- Efficient document retrieval using FAISS vector store
- Context-aware responses using LangChain

## Setup
1. Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage
1. Upload your PDF documents using the file uploader
2. Wait for the documents to be processed
3. Start chatting with your documents using the chat interface

## Note
Make sure you have a valid OpenAI API key to use this application. 