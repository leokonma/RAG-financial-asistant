# ================================================================
# inspect_finance_db.py — Inspect local Chroma store
# ================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.decomposition import PCA

CSV_PATH = "data/Personal_Finance_Dataset.csv"
CHROMA_DIR = "data/chroma_finance_db"

# === 1) Load original CSV ===
print("\n📂 Loading dataset...")
if not os.path.exists(CSV_PATH):
    print("❌ CSV not found. Check path.")
    exit()

df = pd.read_csv(CSV_PATH)
print(f"✅ {len(df)} rows, {len(df.columns)} columns\n")
print("Columns:", list(df.columns))
print(df.head())

# === 2) Load Chroma store ===
print("\n💾 Loading Chroma database...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
collection = db._collection

count = collection.count()
print(f"✅ {count} embedded entries found.")

# === 3) Show samples ===
print("\n🔍 Sample documents:")
sample = collection.get(limit=3)
for i, doc in enumerate(sample["documents"]):
    print(f"\n--- {i+1} ---")
    print(doc[:250], "...")

# === 4) Visualize embeddings (PCA) ===
print("\n📈 Visualizing embeddings...")
data = collection.get(limit=min(200, count), include=["embeddings"])
emb = np.array(data["embeddings"], dtype=object)

emb_clean = [np.array(e, dtype=float) for e in emb if e is not None and not np.isnan(e).any()]
if not emb_clean:
    print("⚠️ No valid embeddings.")
else:
    emb_2d = PCA(n_components=2).fit_transform(emb_clean)
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6, color="steelblue")
    plt.title("PCA of Finance Embeddings")
    plt.show()
