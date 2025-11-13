# ================================================================
# create_finance_db.py — Build vector DB using processed finance data
# ================================================================

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

PROCESSED_PATH = "data/Personal_Finance_Dataset_Processed.csv"
CHROMA_DIR = "data/chroma_finance_db"

# --- 1) Load processed dataset ---
print("📄 Loading processed dataset...")
df = pd.read_csv(PROCESSED_PATH)

if "RAG_Text" not in df.columns:
    raise ValueError("❌ ERROR: 'RAG_Text' column not found. Run prepare_finance_data.py first!")

print(f"Loaded {len(df)} cleaned transactions.")

# --- 2) Convert rows into LangChain Document objects ---
print("🧩 Creating text chunks for embedding...")

documents = []
for _, row in df.iterrows():
    text = row["RAG_Text"]
    metadata = {
        "Date": row["Date"],
        "Category": row["Category"],
        "Type": row["Type"],
        "Amount": row["Amount"],
        "Year": row["Year"],
        "Month": row["Month"],
        "Quarter": row["Quarter"],
    }
    documents.append(Document(page_content=text, metadata=metadata))

print(f"Generated {len(documents)} documents for embedding.")

# --- 3) Load OpenAI Embeddings ---
print("🔵 Loading OpenAI embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 4) Create Chroma Vector DB ---
print("💾 Creating Chroma vector DB...")
os.makedirs(CHROMA_DIR, exist_ok=True)

db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

print("🚀 Vector DB created successfully!")
print(f"📁 Stored in: {CHROMA_DIR}")
