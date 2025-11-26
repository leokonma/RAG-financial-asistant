import os
import shutil
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = "data/chroma_finance_db"
DATASET_PATH = "data/Finance_Processed.csv"


def build_vectorstore():

    # 1. Load dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Dataset missing in data/Finance_Processed.csv")

    df = pd.read_csv(DATASET_PATH)

    if "RAG_Text" not in df.columns:
        raise ValueError("RAG_Text column missing!")

    texts = df["RAG_Text"].astype(str).tolist()
    ids = [f"tx_{i}" for i in range(len(texts))]

    # 2. Delete old DB
    if os.path.exists(CHROMA_DIR):
        print("🗑️ Removing old Chroma DB...")
        shutil.rmtree(CHROMA_DIR)

    # 3. Init embeddings + vector DB
    print("⚡ Creating new embeddings...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    # 4. Add documents manually (Windows safe)
    vectorstore.add_texts(texts=texts, ids=ids)

    print("📦 New vectorstore created:", CHROMA_DIR)


if __name__ == "__main__":
    build_vectorstore()
