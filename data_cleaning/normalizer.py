import pandas as pd   # <-- ESTA LINEA FALTABA !
def normalize(df):
    df = df.copy()

    # --- 1) Ensure correct dtypes ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["description"] = df["description"].astype(str).str.strip()

    # --- 2) Create signed amount (already signed from BG/SD) ---
    df["amount_signed"] = df["amount"]

    # --- 3) Create transaction type ---
    df["type"] = df["amount_signed"].apply(
        lambda x: "income" if x >= 0 else "expense"
    )

    # --- 4) Time features ---
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["dayofweek"] = df["date"].dt.day_name()

    # --- 5) Sort ---
    df = df.sort_values("date").reset_index(drop=True)

    # --- 6) Cumulative balance ---
    df["cumulative_balance"] = df["amount_signed"].cumsum()

    return df
