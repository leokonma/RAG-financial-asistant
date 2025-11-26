# ============================================================
# enricher.py — Create RAG text field
# ============================================================


import pandas as pd

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure required columns exist
    required_cols = ["date", "description", "amount_signed", "auto_category"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

    # Fix types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["description"] = df["description"].fillna("No description")
    df["auto_category"] = df["auto_category"].fillna("Otros")

    # Convert amount to float cleanly
    df["amount_signed"] = pd.to_numeric(df["amount_signed"], errors="coerce").fillna(0)

    def make_text(row):
        tr_type = "expense" if row["amount_signed"] < 0 else "income"
        amount = abs(row["amount_signed"])
        date_str = row["date"].strftime("%Y-%m-%d")

        return (
            f"On {date_str}, a {tr_type} of {amount:.2f} EUR "
            f"at '{row['description']}' "
            f"in category '{row['auto_category']}'."
        )

    df["RAG_Text"] = df.apply(make_text, axis=1)
    return df
