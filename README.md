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

## 🔧 Usage

### 1. Setup & Research (Jupyter)
Open `WikiLLM.ipynb` in VS Code to initialize the environment. Use this notebook to:
* **Test the LLM setup** and ensure your hardware is running correctly.
* **Acquire Data**: Run the Wikipedia cells to download your knowledge base into `./data`.
* **Validate RAG**: Check the response times and factual accuracy of the model.

### 2. Live Chat (Terminal)
Once your data is downloaded, you can skip the notebook and **run the chat locally** via the terminal:
```bash
python wikichat.py
