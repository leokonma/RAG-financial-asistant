

# **Personal Finance RAG Assistant**

### *AI-powered financial insights using OpenAI Embeddings + GPT-3.5 + ChromaDB*

This project implements a **Retrieval-Augmented Generation (RAG)** system that turns your personal financial transactions into an **intelligent finance assistant** capable of answering natural language questions such as:

* *“How much did I spend in November?”*
* *“What was my total income in 2023?”*
* *“What are my top spending categories this year?”*
* *“Give me a summary of expenses by month.”*
* *“What’s my cumulative cashflow over time?”*

It uses:

✔️ **OpenAI Embeddings** (`text-embedding-3-small`)
✔️ **GPT-3.5-Turbo** for reasoning
✔️ **ChromaDB** for vector storage
✔️ **LangChain 0.3.x** (modern API)
✔️ A clean and modular 3-script pipeline

---

# 📂 **Project Structure**

```
RAG-financial-asistant/
│
├── prepare_finance_data.py          # Cleans + enriches CSV, creates RAG_Text
├── create_finance_db.py             # Builds embeddings + Chroma vector DB
├── query_finance_rag.py             # RAG assistant using GPT-3.5
│
├── data/
│   ├── Personal_Finance_Dataset.csv              # (Not tracked in Git)
│   ├── Personal_Finance_Dataset_Processed.csv    # Processed, auto-generated
│   └── chroma_finance_db/                        # Vector database
│
├── .env                               # Contains OPENAI_API_KEY (ignored)
├── .gitignore
└── README.md
```

---

# **1. Setup & Installation**

### **Clone the repo**

```bash
git clone https://github.com/leokonma/RAG-financial-asistant.git
cd RAG-financial-asistant
```

### **Create a virtual environment**

```bash
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\activate

```

### **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# **2. Environment Variables (`.env`)**

Create a `.env` file:

```
OPENAI_API_KEY="your-openai-key-here"
```

---

# **3. Step 1 — Prepare your financial data**

This script transforms your raw CSV into an enriched dataset with:

* cleaned datetime
* Year, Month, Week, Quarter
* spending vs income (signed amounts)
* cumulative balance
* Day of week
* RAG-optimized text (`RAG_Text`)

### Run:

```bash
python prepare_finance_data.py
```

This generates:

```
data/Personal_Finance_Dataset_Processed.csv
```

---

# **4. Step 2 — Build the vector database (Chroma + OpenAI Embeddings)**

Once your data is processed, generate embeddings:

```bash
python create_finance_db.py
```

This creates:

```
data/chroma_finance_db/
```

Each row is converted into a high-quality semantic embedding using
**OpenAI’s `text-embedding-3-small`** (extremely cheap + accurate).

---

# **5. Step 3 — Run the Finance RAG Assistant**

Launch the assistant:

```bash
python query_finance_rag.py
streamlit run dashboard.py
```

You will see:

```
Personal Finance RAG (OpenAI GPT-3.5) ready.
🧠 Ask about your finances:
```

Now ask anything:

* “How much did I spend in February 2022?”
* “What category has the highest expenses?”
* “Give me a monthly summary.”
* “How does my cashflow trend look?”
* “What is my total income for 2023?”

The assistant will:

1. Retrieve relevant transactions using embeddings
2. Feed them to GPT-3.5
3. Produce precise, contextual answers

---

# **How the RAG pipeline works**

### **1. Data Preparation**

`prepare_finance_data.py`
→ Clean & enrich the raw CSV
→ Convert each transaction into a structured + natural-language `RAG_Text`
→ Save processed CSV

### **2. Embedding + Vector Store**

`create_finance_db.py`
→ Convert each row into a Document
→ Embed using OpenAI
→ Store in ChromaDB

### **3. Retrieval-Augmented Generation**

`query_finance_rag.py`
→ Retrieve top-k similar transactions
→ Inject them into a ChatPromptTemplate
→ LLM generates financial reasoning over retrieved data

---

# **Why this works extremely well**

* You have **rich structured time features**
* Embeddings store semantic meaning of transactions
* GPT-3.5 does reasoning **only over retrieved facts**
* Avoid hallucinations and noise
* Very cheap to operate (fractions of a cent per query)

---

# **Privacy & Security**

* No financial files are uploaded to GitHub
* No `.env` file is stored
* All embeddings are local in Chroma
* All data stays on *your machine* except the text sent to OpenAI for analysis



# **Author**

**Leonardo Sánchez Castillo**

