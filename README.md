# WikiLLM: Private Local RAG with TinyLlama

WikiLLM is a private, local-first project built using **LlamaIndex**. It indexes Wikipedia pages and interacts with them via a Retrieval-Augmented Generation (RAG) pipeline—all running locally on your hardware.

## 🚀 Features
- **Privacy First**: No data leaves your machine.
- **Local LLM**: Powered by `TinyLlama-1.1B-Chat` for CPU efficiency.
- **Local Embeddings**: Uses `BAAI/bge-small-en-v1.5` for vector search.
- **Automated Data**: Built-in Wikipedia API integration to fetch research topics.

## 🛠️ Tech Stack
- **Framework**: LlamaIndex
- **Models**: TinyLlama & BGE-Small
- **Data Source**: Wikipedia API

## 🔧 Setup & Usage
1. Install dependencies:
   `pip install llama-index llama-index-embeddings-huggingface llama-index-llms-huggingface transformers wikipedia-api torch`
2. Run `WikiLLM.ipynb` to download data and build your local index.