# ============================================================
# dashboard.py — Personal Finance Dashboard + AI Assistant
# FINAL VERSION — GPT-4.1-mini + Chroma + Amount_Signed
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import os

from dotenv import load_dotenv
from query_finance_rag import get_finance_rag_chain

# ============================================
# 1) Load environment variables
# ============================================
load_dotenv()

# ============================================
# 2) Load CSV dataset
# ============================================
CSV_PATH = "data/Personal_Finance_Dataset_Processed.csv"

@st.cache_data
def load_data():
    df_finance = pd.read_csv(CSV_PATH)

    # Convert date column
    df_finance["Date"] = pd.to_datetime(df_finance["Date"])

    # Normalize key columns
    df_finance = df_finance.rename(columns={
        "Date": "date",
        "Amount": "amount",
        "Category": "category",
        "Type": "type",
        "Amount_Signed": "amount_signed"
    })

    return df_finance

df = load_data()

# ============================================
# 3) Load RAG Assistant
# ============================================
qa_chain = get_finance_rag_chain()

# ============================================
# 4) Streamlit Layout
# ============================================

st.set_page_config(
    page_title="Personal Finance Dashboard + AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Personal Finance Dashboard + 🤖 AI Assistant")

col1, col2 = st.columns([3, 1])   # Left = Dashboard, Right = Assistant

# ============================================
# 5) LEFT PANEL — Dashboard
# ============================================

with col1:

    st.subheader("📁 Filters")

    # Date filter
    min_date = df["date"].min()
    max_date = df["date"].max()

    date_range = st.date_input(
        "Select Date Range",
        value=[min_date, max_date]
    )

    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = df[(df["date"] >= start) & (df["date"] <= end)]

    # KPI Metrics (using Amount_Signed)
    st.subheader("📌 Key Metrics — Using Amount_Signed")

    total_balance = filtered["amount_signed"].sum()
    avg_month = filtered.groupby(filtered["date"].dt.to_period("M"))["amount_signed"].sum().mean()
    total_expense = filtered[filtered["amount_signed"] < 0]["amount_signed"].sum()
    total_income = filtered[filtered["amount_signed"] > 0]["amount_signed"].sum()

    kp1, kp2, kp3, kp4 = st.columns(4)

    kp1.metric("Net Balance", f"${total_balance:,.2f}")
    kp2.metric("Total Income", f"${total_income:,.2f}")
    kp3.metric("Total Expense", f"${total_expense:,.2f}")
    kp4.metric("Avg Monthly Balance", f"${avg_month:,.2f}")

    st.divider()

    # Monthly Net Balance Trend
    st.subheader("📈 Monthly Net Balance Trend")

    df_month = filtered.copy()
    df_month["month"] = df_month["date"].dt.to_period("M").astype(str)

    monthly_balance = df_month.groupby("month")["amount_signed"].sum().reset_index()

    fig_line = px.line(
        monthly_balance,
        x="month",
        y="amount_signed",
        markers=True,
        title="Monthly Net Balance"
    )

    st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

    # Category Breakdown
    st.subheader("🧁 Category Breakdown")

    category_sum = filtered.groupby("category")["amount_signed"].sum().reset_index()

    fig_pie = px.pie(
        category_sum,
        names="category",
        values="amount_signed",
        hole=0.45
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # Raw Table
    st.subheader("📋 Transactions")
    st.dataframe(filtered, use_container_width=True, height=350)


# ============================================
# 6) RIGHT PANEL — AI Assistant
# ============================================

with col2:
    st.subheader("🤖 AI Finance Assistant")

    user_query = st.text_area("Ask something about your finances:")

    if st.button("Ask AI"):
        if user_query.strip() == "":
            st.warning("Please type a question.")
        else:
            try:
                answer = qa_chain.invoke({"query": user_query})
                st.success(answer["result"])
            except Exception as e:
                st.error(f"Error: {e}")
