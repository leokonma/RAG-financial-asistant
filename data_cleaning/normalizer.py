# ============================================================
# normalizer.py — Clean dates, amounts and compute time features
# ============================================================

import pandas as pd
from .utils import clean_date, clean_amount


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ------------------------------------------------
    # 0. Robust renaming: handle Date/DATE/etc.
    # ------------------------------------------------
    rename_map = {}
    for col in df.columns:
        low = str(col).lower().strip()
        if low == "date":
            rename_map[col] = "date"
        elif low == "amount":
            rename_map[col] = "amount"
        elif low in ["description", "transaction description"]:
            rename_map[col] = "description"
        elif low == "source":
            rename_map[col] = "source"
        elif low == "currency":
            rename_map[col] = "currency"
        elif low == "balance":
            rename_map[col] = "balance"
    if rename_map:
        df = df.rename(columns=rename_map)

    # ----------------------------------------
    # 1. Normalize dates
    # ----------------------------------------
    if "date" not in df.columns:
        raise KeyError("normalize(): expected a 'date' column in the dataframe.")
    df["date"] = df["date"].apply(clean_date)

    # ----------------------------------------
    # 2. Normalize amounts
    # ----------------------------------------
    if "amount" not in df.columns:
        raise KeyError("normalize(): expected an 'amount' column in the dataframe.")
    df["amount"] = df["amount"].apply(clean_amount)

    # ----------------------------------------
    # 3. Transaction type + signed amount
    # ----------------------------------------
    if "type" not in df.columns:
        # Infer from sign
        df["type"] = df["amount"].apply(
            lambda x: "income" if (pd.notna(x) and x >= 0) else "expense"
        )

    # Amount is already signed (we did this in loader),
    # but we keep a separate column for clarity.
    df["amount_signed"] = df["amount"]

    # ----------------------------------------
    # 4. Time features
    # ----------------------------------------
    df["year"] = df["date"].dt.year
    df["month_name"] = df["date"].dt.month_name()
    df["dayofweek"] = df["date"].dt.day_name()

    # ----------------------------------------
    # 5. Sort by date
    # ----------------------------------------
    df = df.sort_values("date")

    # ----------------------------------------
    # 6. Cumulative balance (computed)
    # ----------------------------------------
    df["cumulative_balance"] = df["amount_signed"].fillna(0).cumsum()

    return df
