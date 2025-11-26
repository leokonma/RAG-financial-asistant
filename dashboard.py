# ================================================================
# Personal Finance Dashboard + RAG AI Assistant
# ================================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from query_finance_rag import get_finance_rag_chain


# ================================================================
# Load Data
# ================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Finance_Processed.csv", parse_dates=["date"])
    return df


df = load_data()

st.set_page_config(page_title="💸 Finance Dashboard", layout="wide")
st.title("💸 Personal Finance Dashboard")
st.write("Real-time insights + AI assistant powered by your own transaction history.")


# ================================================================
# Sidebar Filters
# ================================================================
st.sidebar.header("🔍 Filters")

# Date Range Filter
min_date = df["date"].min()
max_date = df["date"].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

df_filtered = df[
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1]))
]

# Type Filter
type_filter = st.sidebar.multiselect(
    "Transaction Type",
    options=df["type"].unique(),
    default=df["type"].unique()
)

df_filtered = df_filtered[df_filtered["type"].isin(type_filter)]

# Category Filter
category_filter = st.sidebar.multiselect(
    "Category",
    options=sorted(df["auto_category"].dropna().unique()),
    default=sorted(df["auto_category"].dropna().unique())
)

df_filtered = df_filtered[df_filtered["auto_category"].isin(category_filter)]


# ================================================================
# KPI METRICS
# ================================================================
st.subheader("📊 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

total_expense = df_filtered[df_filtered["amount_signed"] < 0]["amount_signed"].sum()
total_income  = df_filtered[df_filtered["amount_signed"] > 0]["amount_signed"].sum()
net_savings   = df_filtered["amount_signed"].sum()

col1.metric("📉 Total Expenses", f"{total_expense:.2f} €")
col2.metric("📈 Total Income", f"{total_income:.2f} €")
col3.metric("💰 Net Savings", f"{net_savings:.2f} €")


# ================================================================
# Monthly Income vs Expenses
# ================================================================
st.subheader("📅 Monthly Income vs Expenses")

df_filtered["year_month"] = df_filtered["date"].dt.to_period("M").astype(str)

monthly = (
    df_filtered.groupby(["year_month", "type"])["amount_signed"]
    .sum()
    .reset_index()
)

fig = px.bar(
    monthly,
    x="year_month",
    y="amount_signed",
    color="type",
    title="Monthly Income vs Expenses",
    labels={"year_month": "Month", "amount_signed": "Amount (€)"},
)

st.plotly_chart(fig, use_container_width=True)


# ================================================================
# Category Breakdown
# ================================================================
st.subheader("🏷️ Spending by Category")

category_summary = (
    df_filtered.groupby("auto_category")["amount_signed"]
    .sum()
    .reset_index()
    .sort_values("amount_signed")
)

fig2 = px.bar(
    category_summary,
    x="amount_signed",
    y="auto_category",
    orientation="h",
    title="Total Spending by Category",
)

st.plotly_chart(fig2, use_container_width=True)


# ================================================================
# Raw Table
# ================================================================
st.subheader("📄 Transaction Table")

st.dataframe(
    df_filtered.sort_values("date", ascending=False),
    use_container_width=True,
    height=450
)


# ================================================================
# 🤖 AI RAG ASSISTANT
# ================================================================
st.divider()
st.subheader("🤖 Finance Assistant (RAG-Powered)")

# Load RAG chain once
@st.cache_resource
def load_rag():
    return get_finance_rag_chain()

rag_chain = load_rag()

with st.form("rag_form"):
    user_q = st.text_input(
        "Ask something about your finances:",
        placeholder="Examples: 'How much did I spend eating out this month?', 'What are my biggest expenses?', 'Summarize last week.'"
    )
    submitted = st.form_submit_button("Ask")

if submitted and user_q.strip() != "":
    with st.spinner("Thinking..."):
        try:
            response = rag_chain.invoke({"input": user_q})
            st.markdown(f"### 💬 Answer:\n{response['answer']}")
        except Exception as e:
            st.error(f"⚠️ Error running RAG: {e}")
