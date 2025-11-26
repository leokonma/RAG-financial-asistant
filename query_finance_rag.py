# ================================================================
# query_finance_rag.py — Unified RAG Engine for Dashboard & Terminal
# Adapted for new Finance_Processed.csv schema (RAG_Text field)
# ================================================================

import os
from dotenv import load_dotenv
load_dotenv()

# LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --------------------------------------------------
# Global Config
# --------------------------------------------------
CHROMA_DIR = "data/chroma_finance_db"   # <-- this stays the same


# ================================================================
# FUNCTION: Create a RAG chain for dashboard & CLI
# ================================================================
def get_finance_rag_chain():
    """Creates and returns the RAG chain (LangChain 0.3.x compliant)."""

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # ---------------------------
    # FIXED PROMPT FOR LC 0.3.x
    # ---------------------------
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
Eres un asistente experto en análisis de finanzas personales y planificación financiera.

REGLAS IMPORTANTES:
1. Usa SOLO el contexto recuperado del vector DB para datos numéricos (no inventes transacciones).
2. Puedes hacer análisis avanzados:
   - presupuestos mensuales
   - hábitos de gasto
   - recomendaciones para ahorrar
   - optimización de gasto en categorías
   - proyecciones basadas en patrones reales
3. Puedes razonar, explicar y sugerir acciones basadas en los datos (insights).
4. Si falta información, dilo claramente y explica qué se necesitaría.
5. Sé directo, práctico, profesional y útil.

Tus tareas además de responder preguntas:
- Identificar categorías donde el usuario gasta más
- Detectar meses con desequilibrio entre ingreso y gasto
- Proponer límites de gasto mensuales basados en patrones reales
- Construir planes de ahorro (ej: 10%, 20% del ingreso promedio)
- Explicar cómo mejorar la salud financiera del usuario
- Resumir comportamientos financieros

Recuerda:
El contexto recuperado son transacciones reales que debes tratar como verdad absoluta.
""")
,

        # IMPORTANT: "context" MUST be exposed explicitly for LC 0.3.x
        ("system", "Contexto recuperado:\n{context}"),

        # This is the user question
        ("user", "{input}")
    ])

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.15,
        max_tokens=400
    )

    # ---------------------------
    # NEW WAY: Build the chain MANUALLY
    # ---------------------------
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain

    # Stuff-chaining of documents into a single context chunk
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    # Full RAG chain
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    return rag_chain

# ================================================================
# OPTIONAL TERMINAL MODE
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
            res = qa.invoke({"input": q})   # <-- FIXED
            print(f"💡 {res['answer']}\n")   # <-- FIXED
        except Exception as e:
            print(f"⚠️ Error: {e}\n")
