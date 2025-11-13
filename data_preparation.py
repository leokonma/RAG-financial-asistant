# ================================================================
# prepare_finance_data.py — Clean & enrich personal finance CSV
# ================================================================

import pandas as pd
import os

RAW_PATH = "data/Personal_Finance_Dataset.csv"
OUTPUT_PATH = "data/Personal_Finance_Dataset_Processed.csv"

def prepare_finance_data():
    print("📄 Loading raw dataset...")
    df = pd.read_csv(RAW_PATH)

    # --- 1) Convert Date column ---
    print("⏱ Converting Date column to datetime...")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=["Date"])

    # --- 2) Temporal features ---
    print("📆 Generating time features...")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%B")
    df["Day"] = df["Date"].dt.day
    df["Week"] = df["Date"].dt.isocalendar().week
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.day_name()
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

    # --- 3) Signed amounts ---
    print("💰 Normalizing income/expense amounts...")
    df["Amount_Signed"] = df.apply(
        lambda row: -row["Amount"] if str(row["Type"]).lower() == "expense" else row["Amount"],
        axis=1
    )

    # --- 4) Cumulative balance ---
    print("📈 Calculating cumulative balance over time...")
    df = df.sort_values("Date")
    df["Cumulative_Balance"] = df["Amount_Signed"].cumsum()

    # --- 5) Combined finance text (for better embedding quality) ---
    print("🧠 Building combined text field for RAG...")
    df["RAG_Text"] = (
        "On " + df["Date"].dt.strftime("%Y-%m-%d") +
        ", there was a " + df["Type"] +
        " of $" + df["Amount"].astype(str) +
        " in category '" + df["Category"] + "'." +
        " Additional details: " + df["Transaction Description"] +
        ". Month: " + df["Month_Name"] +
        ", Year: " + df["Year"].astype(str) +
        ", Day of week: " + df["DayOfWeek"] + "."
    )

    # --- 6) Save processed file ---
    print(f"💾 Saving processed dataset to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n🚀 Finance dataset processed successfully!")
    print(f"Saved to: {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    prepare_finance_data()
