# ================================================================
# query_finance_rag.py — RAG using OpenAI GPT-3.5 + OpenAI Embeddings
# ================================================================

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

CHROMA_DIR = "data/chroma_finance_db"

# --- Load Embeddings ---
print("🔵 Loading Chroma DB with OpenAI embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- Load LLM ---
print("🤖 Loading LLM: gpt-3.5-turbo")
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.15,
    max_tokens=350
)

# --- Build Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """
Eres un asistente experto en análisis de finanzas personales.

Reglas:
- Usa SOLO el contexto recuperado del vector DB.
- Calcula montos, totales, promedios, meses, trimestres y categorías.
- Si falta información, dilo explícitamente.
- Sé preciso y profesional.
"""),
    ("user", "{question}"),
    ("system", "Contexto recuperado:\n{context}")
])

# --- RAG chain ---
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

print("\n💬 Personal Finance RAG (OpenAI GPT-3.5) ready.\n")

# --- Loop ---
while True:
    q = input("🧠 Ask about your finances: ")

    if q.lower() in ["exit", "quit"]:
        print("👋 Bye!")
        break

    try:
        result = qa.invoke({"query": q})
        print(f"💡 {result['result']}\n")
    except Exception as e:
        print(f"⚠️ Error: {e}\n")
