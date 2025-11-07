# ================================================================
# query_finance_rag.py — Query your finance knowledge base (Local Ollama)
# ================================================================

import os
import subprocess
from dotenv import load_dotenv

# ==== 1️⃣ Load environment ====
load_dotenv()

# ==== 2️⃣ LangChain imports ====
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# ==== 3️⃣ Configuration ====
CHROMA_DIR = "data/chroma_finance_db"
OLLAMA_MODEL = "llama3:8b"   # change if you pulled a different model name

# ==== 4️⃣ Check if Ollama and model are available ====
try:
    # Check that Ollama CLI is reachable
    subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=True)
except Exception:
    print("⚠️  Ollama is not detected in PATH or not running.")
    print("👉  Make sure Ollama is installed and open the Ollama app before running this script.")
    exit(1)

try:
    # Check if model is downloaded
    result = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, check=True
    )
    if OLLAMA_MODEL.split(":")[0] not in result.stdout:
        print(f"⚠️  Model '{OLLAMA_MODEL}' not found.")
        print(f"👉  Please run this command first in PowerShell:")
        print(f"    ollama pull {OLLAMA_MODEL}")
        exit(1)
except Exception:
    print("⚠️  Could not check Ollama models. Make sure Ollama service is running.")
    exit(1)

# ==== 5️⃣ Load embeddings and vector database ====
print("🔍 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("💾 Loading Chroma DB...")
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ==== 6️⃣ Load local LLaMA model through Ollama ====
print(f"🤖 Connecting to local Ollama model: {OLLAMA_MODEL}")
llm = Ollama(model=OLLAMA_MODEL)

# ==== 7️⃣ Build Retrieval-QA chain ====
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

print("\n💬 Personal Finance RAG (Local Ollama) ready. Type 'exit' to quit.\n")

# ==== 8️⃣ Interactive Q&A loop ====
while True:
    query = input("🧠 Ask about your finances: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Bye!")
        break

    try:
        result = qa.invoke({"query": query})
        print(f"💡 {result['result']}\n")
    except Exception as e:
        print(f"⚠️ Error: {e}\n")

print("🔍 Sample retrieval test:")
docs = retriever.get_relevant_documents("How much did I spend in September?")
print(f"Found {len(docs)} docs")
for d in docs[:2]:
    print(d.page_content[:200], "\n---")
