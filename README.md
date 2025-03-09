# Financial RAG Assistant

A powerful financial assistant that combines Retrieval Augmented Generation (RAG) with stock price analysis to provide comprehensive financial insights.

## Features

- **Document-based RAG**: Query financial documents using state-of-the-art language models
- **Stock Price Analysis**: Access historical stock data, technical indicators, and performance metrics
- **Interactive Visualizations**: Generate charts and visualizations for stock data
- **GPU Acceleration**: Utilize NVIDIA GPUs for faster processing
- **Local Inference**: Run everything locally using Ollama for privacy and control

## Project Structure

```
financial-rag-assistant/
├── data/                  # Data directory for stock prices and documents
├── embeddings/            # Generated embeddings from documents
├── financial_assistant/   # Main package
│   ├── __init__.py        # Package initialization
│   ├── app.py             # Streamlit application entry point
│   ├── document_processing/  # Document processing modules
│   │   ├── __init__.py
│   │   └── processor.py   # PDF processing functionality
│   ├── rag/               # RAG functionality
│   │   ├── __init__.py
│   │   └── engine.py      # RAG engine implementation
│   ├── stock_analysis/    # Stock analysis modules
│   │   ├── __init__.py
│   │   └── analyzer.py    # Stock data analysis functionality
│   └── utils/             # Utility functions
│       ├── __init__.py
│       └── helpers.py     # Helper functions
└── scripts/               # Utility scripts
    └── process_documents.py  # Script to process documents and create embeddings
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-rag-assistant.git
cd financial-rag-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama:
Follow the instructions at [Ollama's website](https://ollama.ai/download) to install Ollama for your platform.

4. Pull the required models:
```bash
ollama pull llama3.1:8b
```

## Usage

### Processing Documents

Process your financial documents to create embeddings:

```bash
python -m financial_assistant.scripts.process_documents --input_dir /path/to/pdf --output_dir embeddings
```

### Running the Application

Start the Streamlit application:

```bash
python -m financial_assistant.app
```

Or directly with Streamlit:

```bash
streamlit run financial_assistant/app.py
```

### Querying the Assistant

Once the application is running:

1. Select your data sources (documents, stock data, or both)
2. Load the data
3. Ask questions in natural language:
   - Document queries: "What was Apple's revenue growth in 2022?"
   - Stock queries: "What is the current price of AAPL?"
   - Technical analysis: "Show me the RSI for MSFT"
   - Comparisons: "Compare AMZN to NFLX performance"

## Requirements

- Python 3.9+
- CUDA-compatible GPU (optional, for acceleration)
- Ollama

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG implementation
- [Ollama](https://ollama.ai/) for local LLM inference
- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for vector storage and similarity search

### LLM for financial data

The aim of this project is to fine-tune an LLM with financial tabular and textual data.
Also, incorporate information generated from ML models.

The rationale is that there is an increasing interest in small LLMS for 
**Deliverable**: Python web app (streamlit) able to interact. 







### Recursos

[Github Papers Tabular LLM](https://github.com/SpursGoZmy/Awesome-Tabular-LLMs)

### Papers

* [FinGPT: Open-Source Financial Large Language Models](https://arxiv.org/pdf/2306.06031)
* [Attention is all you need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
* [TAT-LLM: A Specialized Language Model for Discrete Reasoning over Tabular and Textual Data](https://arxiv.org/pdf/2401.13223)
* [Financial Statement Analysis with Large Language Models](https://bfi.uchicago.edu/working-paper/financial-statement-analysis-with-large-language-models/)
* [In Defense of RAG in the Era of Long-Context Language Models](https://arxiv.org/pdf/2409.01666)
* [BloombergGPT](https://arxiv.org/abs/2303.17564)
* [Other Worlds: Using AI to Revisit Cybersyn and Rethink Economic Futures](https://arxiv.org/pdf/2411.05992)
* [Large Language Models in Finance: A Survey](https://dl.acm.org/doi/pdf/10.1145/3604237.3626869)
* [FinBERT: A Large Language Model for Extracting Information from Financial Text](https://onlinelibrary.wiley.com/doi/10.1111/1911-3846.12832)

### Videos

* [Table-GPT by Microsoft: Empower LLMs To Understand Tables ](https://www.youtube.com/watch?v=yGL0XZlGA0I)
* [Algorithmic Trading – Machine Learning & Quant Strategies Course with Python](https://www.youtube.com/watch?v=9Y3yaoi9rUQ&t=1094s)
* [Fine Tuning LLM Models – Generative AI Course](https://www.youtube.com/watch?v=iOdFUJiB0Zc)
* [Fine-tuning de grandes modelos de lenguaje con Manuel Romero | Hackathon Somos NLP 2023](https://www.youtube.com/watch?v=WYcJb8gYBZU&t=799s)
* [Agentic RAG for stock trading using Llama Index](https://www.youtube.com/watch?v=uOLhleiOM84)
* [Agentic RAG for stock trading using Llama Index - Notebook](https://github.com/adidror005/youtube-videos/blob/main/AI%20Trading%20Assistant%20Actual.ipynb)
* [ Building LLMs from the Ground Up: A 3-hour Coding Workshop ](https://www.youtube.com/watch?v=quh7z1q7-uc&t=242s)

### Libros

[Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/)

### Llama3

https://huggingface.co/meta-llama/Meta-Llama-3-8B
https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B

### Articulos

[Tabular Data, RAG, & LLMs: Improve Results Through Data Table Prompting](https://medium.com/intel-tech/tabular-data-rag-llms-improve-results-through-data-table-prompting-bcb42678914b)
[LLM Reading list](https://sebastianraschka.com/blog/2023/llm-reading-list.html)
[Small LLMS Hugging Face](https://huggingface.co/blog/smollm)
[Building a GraphRAG Agent With Neo4j and Milvus](https://neo4j.com/developer-blog/graphrag-agent-neo4j-milvus/)

### Conferences?

NLP conferences?
Pytorch conf?
Nerdearla
Neo4j o graph conferences?
