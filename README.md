# WikiLLM: Private Local RAG & Chatbot

WikiLLM is a private, local-first Large Language Model (LLM) project built using **LlamaIndex**. It allows you to index specific Wikipedia pages and interact with them via two distinct modes: a research notebook for setup and a terminal-based chatbot for local interaction.

## 🚀 Features
- **Privacy First**: No data leaves your machine; everything runs locally.
- **Local LLM**: Powered by `TinyLlama-1.1B-Chat` for high efficiency on standard CPUs.
- **Local Embeddings**: Uses `BAAI/bge-small-en-v1.5` for precise vector retrieval.
- **Optimized Retrieval**: Implements a `SentenceSplitter` with a chunk size of 512 and an overlap of 50 to ensure high-quality context for the LLM.

## 📊 Dataset & Data Sourcing

### What is the dataset?
The dataset is a custom, locally-built knowledge base focused on modern Artificial Intelligence. By default, the notebook is configured to fetch the following Wikipedia articles:
* **Generative Artificial Intelligence**
* **Large Language Model**
* **Transformer (machine learning)**
* **Diffusion Model**

### How is it downloaded?
The data is acquired dynamically via the **Wikipedia-API**:
1. **Automated Fetching**: The script loops through predefined topics and fetches the full text content from Wikipedia.
2. **Sanitization**: Page titles are automatically sanitized (replacing spaces with underscores) to serve as local filenames.
3. **Storage**: Content is saved as `.txt` files in a local `./data` directory. This folder is ignored by Git to keep the repository clean.

## 🛠️ Tech Stack
- **Framework**: LlamaIndex
- **LLM**: TinyLlama-1.1B-Chat-v1.0
- **Embeddings**: BAAI/bge-small-en-v1.5
- **Data Source**: Wikipedia-API

## 📋 Prerequisites
Install the required libraries:
```bash
pip install llama-index llama-index-embeddings-huggingface llama-index-llms-huggingface transformers accelerate wikipedia-api torch
