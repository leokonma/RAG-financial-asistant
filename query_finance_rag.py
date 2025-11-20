# ================================================================
# query_finance_rag.py — Unified RAG Engine for Dashboard & Terminal
# Uses: OpenAI GPT-4.1-mini + OpenAI Embeddings + Chroma DB
# ================================================================

import os
from dotenv import load_dotenv

# LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --------------------------------------------------
# Global Config
# --------------------------------------------------
CHROMA_DIR = "data/chroma_finance_db"


# ================================================================
# FUNCTION: RAG Pipeline (importable from dashboard)
# ================================================================
def get_finance_rag_chain():
    """Creates and returns the RAG chain used for both dashboard and terminal."""

    # --- Load Embeddings ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # --- Load LLM (UPGRADED from GPT-3.5 to GPT-4.1-mini) ---
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.15,
        max_tokens=450
    )

    # --- Prompt Template ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
Eres un asistente experto en análisis de finanzas personales.

Reglas importantes:
- Usa SOLO el contexto recuperado del vector DB (no inventes datos).
- Calcula correctamente: totales, promedios, balances, meses, trimestres, categorías.
- Si la información no existe en los datos, dilo claramente.
- Sé directo, preciso y profesional.
"""),
        ("system", "Contexto recuperado:\n{context}"),
        ("user", "{question}")
    ])

    # --- RAG chain ---
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


# ================================================================
# OPTIONAL TERMINAL INTERFACE
# ================================================================
if __name__ == "__main__":
    print("🔵 Loading Chroma DB with OpenAI embeddings...")
    print("🤖 Loading LLM: gpt-4.1-mini")
    print("\n💬 Personal Finance RAG ready.\n")

    qa = get_finance_rag_chain()

    while True:
        q = input("🧠 Ask about your finances: ")

        if q.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        try:
            res = qa.invoke({"query": q})
            print(f"💡 {res['result']}\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")
