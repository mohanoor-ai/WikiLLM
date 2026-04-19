import os
import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# 1. SETUP CONFIGURATION
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# This prompt forces the model to stay in character and use the provided context
system_prompt = """<|system|>
You are a helpful, concise AI assistant for WikiLLM. 
Your goal is to answer questions based ONLY on the provided document context.
If the user says 'hello' or 'hi', greet them politely and ask what they want to know about the local documents.
Do not generate random Python code unless specifically asked to write a script.</s>"""

# Wrapper to ensure TinyLlama understands where the user input starts
query_wrapper_prompt = PromptTemplate("<|system|>\n{system_msg}<|user|>\n{query_str}<|assistant|>")

Settings.llm = HuggingFaceLLM(
    model_name=model_id,
    tokenizer_name=model_id,
    context_window=2048,
    max_new_tokens=256,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    # Lower temperature (0.1) makes the model focused and less "creative/random"
    generate_kwargs={"temperature": 0.1, "do_sample": False}, 
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512

# 2. LOAD DATA AND BUILD INDEX
if not os.path.exists("./data"):
    print("Error: 'data' folder not found. Run the WikiLLM notebook first to download files.")
    exit()

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(
    documents, 
    transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50)]
)

# 3. INITIALIZE CHAT ENGINE
# Using 'context' mode so it uses the indexed documents to answer
chat_engine = index.as_chat_engine(chat_mode="context")

print("\n" + "="*50)
print("🤖 WikiLLM CHATBOT IS ONLINE")
print("I will answer based on your downloaded Wikipedia pages.")
print("Type 'exit' to stop.")
print("="*50 + "\n")

# 4. CHAT LOOP
while True:
    user_msg = input("You: ")
    if user_msg.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break
    
    print("Bot is thinking...")
    # The engine now uses the system prompt and context to filter out the noise
    response = chat_engine.chat(user_msg)
    print(f"\nBot: {response}\n")
