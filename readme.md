

# 💸 **Personal Finance RAG Assistant — AI-Powered Financial Intelligence**

### *Your personal financial co-pilot, combining data cleaning, analytics, and Retrieval-Augmented Generation.*

This project transforms your **raw bank transactions** into a **clean, enriched, searchable financial intelligence system** powered by:

* **OpenAI GPT models**
* **Chroma vector store**
* **LangChain 0.3.x**
* **Streamlit dashboard (Notion-style redesign)**
* **Multi-bank ingestion (BG + Santander)**
* **Automatic currency conversion, categorization & enrichment**

You can ask natural-language questions such as:

> *“How much did I spend in restaurants last month?”*
> *“Show me my cumulative balance over time.”*
> *“What are my top spending categories?”*
> *“Summarize my year’s expenses.”*
> *“Where did I spend the most in Madrid?”*

---

# 🧠 **Core Features**

### ✔ Multi-bank support (Banco General + Santander)

### ✔ FX conversion (USD → EUR)

### ✔ Automatic categorization

### ✔ RAG-optimized transaction text for embeddings

### ✔ Chroma vector DB with OpenAI embeddings

### ✔ Notion-style financial dashboard

### ✔ Integrated AI assistant (ChatGPT)

### ✔ Advanced filters (category, source, amount, keyword)

### ✔ Heatmaps, donut charts, monthly breakdowns

### ✔ Full pipeline automation (`runner.py`)

---

# 📁 **Project Structure**

```
RAG-financial-assistant/
│
├── data_cleaning/
│   ├── loader.py              ← Multi-bank ingestion (BG + SD)
│   ├── normalizer.py          ← Date, amount, type, temporal features
│   ├── fx_converter.py        ← USD → EUR conversion
│   ├── categorizer.py         ← Auto-categories (Supermarket, Uber, etc.)
│   ├── enricher.py            ← Creates RAG_Text for embeddings
│   └── utils.py
│
├── build_chroma_vectorstore.py ← Rebuilds vector DB cleanly
├── query_finance_rag.py        ← RAG assistant (LLM-powered)
├── dashboard.py                ← Notion-style Streamlit dashboard
├── runner.py                   ← Full pipeline automation
│
├── data/
│   ├── BG_Transaccions.xlsx
│   ├── SD_Transaccions.xlsx
│   ├── fx_rates.csv
│   ├── Finance_Processed.csv
│   └── chroma_finance_db/
│
├── .env
├── .gitignore
└── README.md
```

---

# ⚙️ **1. Setup & Installation**

### Create virtual environment

```bash
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Add your API key

Create a `.env` file:

```
OPENAI_API_KEY="your-key-here"
```

---

# 🔄 **2. Run the Complete Data Pipeline**

This command runs **every step**:

```bash
python runner.py
```

It performs:

1. Load BG + SD bank data
2. Convert USD → EUR
3. Normalize (date, amounts, type, time features)
4. Categorize
5. Enrich with RAG_Text
6. Export `Finance_Processed.csv`

Output:

```
data/Finance_Processed.csv
```

---

# 🧱 **3. Build the Vector Database (Chroma + OpenAI Embeddings)**

Once the processed dataset is generated:

```bash
python build_chroma_vectorstore.py
```

This generates (or rebuilds):

```
data/chroma_finance_db/
```

Each transaction becomes a semantic embedding using:

**OpenAI – text-embedding-3-small**
(cheap, fast, high-quality)

---

# 🧠 **4. RAG Assistant — Ask AI about your finances**

Launch the RAG assistant:

```bash
python query_finance_rag.py
```

Example questions:

* “How much did I spend in January?”
* “What are my biggest expenses this year?”
* “Summarize my finances this month.”
* “How many Uber rides did I take?”

The assistant:

1. Retrieves relevant transactions
2. Feeds them into GPT
3. Produces context-aware financial insights

---

# 📊 **5. Notion-Style Dashboard (Streamlit)**

Run the dashboard:

```bash
streamlit run dashboard.py
```

### ✨ Features:

#### 🧮 KPIs

* Total expenses
* Total income
* Net savings

#### 📅 Monthly income vs expenses

#### 🍩 Donut chart by category

#### 🔥 Heatmap (weekday × month)

#### 🛍 Top vendors

#### 🔍 Advanced filters

* Source (BG/SD)
* Category
* Income/expense
* Amount range slider
* Keyword search in description
* Date range

#### 🤖 AI Assistant (right panel)

* Persistent side chat
* Interacts with your actual vectorstore
* Summaries, insights, budgeting help

---

# 🔍 **6. Pipeline Explained**

### **A) Loader Phase**

Multi-bank ingestion:

* Drop irrelevant columns
* Convert amounts
* Convert dates
* Clean descriptions
* Add bank metadata
* Normalize column schema

### **B) Normalization Phase**

Adds:

* Type: income / expense
* Signed amounts
* Year / Month / Day
* Day of week
* Cumulative balance

### **C) Categorization Phase**

Regex-based classifier:

* Supermarket
* Restaurants
* Uber
* Tabaco / Estanco
* Pharmacy
* Suscriptions
* Movilidad
* Otros

### **D) Enrichment Phase**

Builds:

`RAG_Text` → optimized for embeddings

Example:

```
On 2025-01-05, a expense of 14.20 EUR at "Primaprix" categorized as "Supermarket".
```

### **E) Vectorization**

ChromaDB with persistent embeddings.

### **F) RAG Query Engine**

Retrieves k=5 similar rows + GPT reasoning.

---

# 🛡 **7. Privacy & Security**

* No financial files tracked in Git
* `.env` excluded
* Embeddings stored locally
* No cloud storage
* Only text snippets are sent to OpenAI during queries
* All raw data stays on device

---

# 🏁 **8. Roadmap**

* Predictive budgeting with ML
* Personal finance anomaly detection
* Subscription manager
* Spending alerts
* Savings recommendation engine
* Export monthly PDF reports
* Mobile-friendly dashboard mode

---

