# ================================================================
# PERSONAL FINANCE DASHBOARD — Notion/Minimal Redesign
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from query_finance_rag import get_finance_rag_chain


# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="💸 Personal Finance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal style
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .sidebar .sidebar-content { padding-top: 2rem; }
        .stChatMessage { background-color: #f7f7f7 !important; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)


# ================================================================
# LOAD DATA
# ================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Finance_Processed.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()


# ================================================================
# SIDEBAR — ADVANCED FILTERS (Notion aesthetic)
# ================================================================
st.sidebar.title("🔧 Filters")

# DATE RANGE
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "📆 Date range",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

df_filtered = df[
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1]))
]

# TYPE FILTER
type_filter = st.sidebar.multiselect(
    "💼 Transaction type",
    df["type"].unique(),
    default=df["type"].unique()
)
df_filtered = df_filtered[df_filtered["type"].isin(type_filter)]

# CATEGORY FILTER
category_filter = st.sidebar.multiselect(
    "🏷 Category",
    sorted(df["auto_category"].unique()),
    default=sorted(df["auto_category"].unique())
)
df_filtered = df_filtered[df_filtered["auto_category"].isin(category_filter)]

# SOURCE FILTER (NEW)
source_filter = st.sidebar.multiselect(
    "🏦 Bank source",
    df["source"].unique(),
    default=df["source"].unique()
)
df_filtered = df_filtered[df_filtered["source"].isin(source_filter)]

# AMOUNT RANGE FILTER (NEW)
min_amt, max_amt = float(df["amount"].min()), float(df["amount"].max())
amount_range = st.sidebar.slider(
    "💰 Amount range",
    min_amt, max_amt, (min_amt, max_amt)
)
df_filtered = df_filtered[
    (df_filtered["amount"] >= amount_range[0]) &
    (df_filtered["amount"] <= amount_range[1])
]

# TEXT SEARCH FILTER (NEW)
text_search = st.sidebar.text_input("🔍 Search in description")
if text_search.strip() != "":
    df_filtered = df_filtered[df_filtered["description"].str.contains(text_search, case=False)]


# ================================================================
# LAYOUT — TWO COLUMNS
# ================================================================
left, right = st.columns([0.68, 0.32])


# ================================================================
# LEFT COLUMN — DASHBOARD
# ================================================================
with left:
    st.title("📊 Financial Overview")

    # KPIs
    col1, col2, col3 = st.columns(3)
    total_expense = df_filtered[df_filtered["amount_signed"] < 0]["amount_signed"].sum()
    total_income = df_filtered[df_filtered["amount_signed"] > 0]["amount_signed"].sum()
    net_savings = df_filtered["amount_signed"].sum()

    col1.metric("📉 Total Expenses", f"{total_expense:.2f} €")
    col2.metric("📈 Total Income", f"{total_income:.2f} €")
    col3.metric("💰 Net Savings", f"{net_savings:.2f} €")

    st.markdown("---")

    # Monthly trends
    df_filtered["ym"] = df_filtered["date"].dt.to_period("M").astype(str)
    monthly = df_filtered.groupby(["ym", "type"])["amount_signed"].sum().reset_index()

    fig = px.bar(monthly, x="ym", y="amount_signed", color="type",
                title="📅 Monthly Income vs Expenses", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Category Donut
    cat_sum = df_filtered.groupby("auto_category")["amount_signed"].sum().abs()
    fig2 = px.pie(cat_sum, names=cat_sum.index, values=cat_sum.values,
                  title="🏷 Spending Breakdown by Category",
                  hole=0.5)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # HEATMAP (day vs month)
    df_filtered["weekday"] = df_filtered["date"].dt.day_name()
    df_filtered["month_name"] = df_filtered["date"].dt.month_name()

    heat = df_filtered.groupby(["weekday", "month_name"])["amount_signed"].sum().reset_index()

    # Fix weekday order for beautiful heatmap
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat["weekday"] = pd.Categorical(heat["weekday"], categories=order, ordered=True)

    # Correct pivot syntax
    pivot = heat.pivot(index="weekday", columns="month_name", values="amount_signed").fillna(0)

    fig3 = px.imshow(
        pivot,
        title="🔥 Heatmap — Spending Pattern (Day vs Month)",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig3, width="stretch")



# ================================================================
# RIGHT COLUMN — AI ASSISTANT (FIXED PANEL)
# ================================================================
with right:
    st.header("🤖 AI Finance Assistant")

    @st.cache_resource
    def load_rag():
        return get_finance_rag_chain()

    rag = load_rag()

    user_q = st.text_area("Ask me anything about your finances:", height=120)
    if st.button("💬 Ask"):
        with st.spinner("Analyzing your finances..."):
            try:
                res = rag.invoke({"input": user_q})
                st.markdown("### 💡 Answer:")
                st.write(res["answer"])
            except Exception as e:
                st.error(f"Error: {e}")

