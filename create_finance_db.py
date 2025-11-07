# ================================================================
# create_finance_db.py — Conversión de transacciones a embeddings
# ================================================================

import os
import torch
from time import sleep

# 🚫 Desactivar telemetría y avisos de Chroma
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# === Librerías de LangChain ===
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Compatibilidad entre versiones (sin cambiar requirements)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_chroma import Chroma

# === Parámetros ===
CSV_PATH = "data/Personal_Finance_Dataset.csv"
CHROMA_DIR = "data/chroma_finance_db"

# === Paso 1: Cargar datos ===
print("✅ Loading transactions from CSV...")
loader = CSVLoader(file_path=CSV_PATH, encoding="utf-8")
docs = loader.load()
print(f"✅ Loaded {len(docs)} transactions.")

# === Paso 2: Dividir texto ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(docs)
print(f"✅ Generated {len(texts)} text chunks.")

# === Paso 3: Cargar modelo de embeddings ===
print("⚙️ Loading model... this may take a few minutes ⏳")
torch.set_num_threads(4)  # evita bloqueos al inicializar PyTorch

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("✅ Embedding model loaded successfully.")

# === Paso 4: Crear base vectorial ===
print("💾 Creating Chroma vector database...")
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

print(f"✅ Stored {len(texts)} chunks in {CHROMA_DIR}")
print("🚀 Finance DB successfully created and persisted.")
print("🎯 You can now run `python query_finance_rag.py` to query your finance data.")
