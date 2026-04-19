# WikiLLM: Private Local RAG & Chatbot

WikiLLM is a private, local-first Large Language Model (LLM) project built using **LlamaIndex**. It allows you to index specific Wikipedia pages and interact with them via two distinct modes: a research notebook for setup and a terminal-based chatbot for local interaction.

## 🚀 Features
- **Privacy First**: No data leaves your machine; everything runs locally.
- **Local LLM**: Powered by `TinyLlama-1.1B-Chat` for high efficiency on standard CPUs.
- **Local Embeddings**: Uses `BAAI/bge-small-en-v1.5` for precise vector retrieval.
- **Optimized Retrieval**: Implements a `SentenceSplitter` (chunk size 512, overlap 50) to provide high-quality context to the model.

## 🧠 Handling Hallucinations in Small Models
Because TinyLlama is a compact model (1.1B parameters), it is more prone to "hallucinating" (generating irrelevant code or random text) if not strictly guided. We have implemented three specific safeguards in `wikichat.py`:

1. **Strict System Prompting**: We use a `<|system|>` block to explicitly tell the model: *"Use the provided context ONLY. If you don't know the answer, say so. Do not generate random code."*
2. **Low Temperature (0.1)**: By setting the temperature to `0.1`, we force the model to be more predictable and factual rather than "creative" or random.
3. **Abstention Logic**: The prompt instructs the model to admit when it cannot find an answer in the local documents, preventing it from making up information.

## 📊 Dataset & Data Sourcing

### What is the dataset?
The dataset is a custom knowledge base focused on modern AI topics, including:
* **Generative Artificial Intelligence**
* **Large Language Models (LLMs)**
* **Transformer (machine learning)**
* **Diffusion Models**

### How is it downloaded?
The data is acquired via the **Wikipedia-API** in the `WikiLLM.ipynb` notebook:
1. **Automated Fetching**: The script loops through topics and pulls clean text content.
2. **Sanitization**: Page titles are converted into safe local filenames (e.g., `Large_Language_Model.txt`).
3. **Storage**: Content is saved in the `./data` directory (ignored by Git).

## 📋 Prerequisites
```bash
pip install llama-index llama-index-embeddings-huggingface llama-index-llms-huggingface transformers accelerate wikipedia-api torch
