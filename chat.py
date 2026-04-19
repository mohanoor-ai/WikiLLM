import os
import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# 1. SETUP CONFIGURATION (Same as your notebook)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

Settings.llm = HuggingFaceLLM(
    model_name=model_id,
    tokenizer_name=model_id,
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": True},
    device_map="auto",
    model_kwargs={"dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512

# 2. LOAD DATA AND BUILD INDEX
print("Wait... Loading your Wikipedia knowledge base...")
if not os.path.exists("./data"):
    print("Error: 'data' folder not found. Run the notebook first to download files.")
    exit()

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(
    documents, 
    transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50)]
)

# 3. INITIALIZE CHAT ENGINE
# 'context' mode allows the bot to remember the conversation
chat_engine = index.as_chat_engine(chat_mode="context")

print("\n" + "="*50)
print("🤖 WikiLLM CHATBOT IS ONLINE")
print("You can ask questions about the AI documents in your 'data' folder.")
print("Type 'exit' or 'quit' to stop.")
print("="*50 + "\n")

# 4. CHAT LOOP
while True:
    user_msg = input("You: ")
    if user_msg.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break
    
    print("Bot is thinking...")
    response = chat_engine.chat(user_msg)
    print(f"\nBot: {response}\n")