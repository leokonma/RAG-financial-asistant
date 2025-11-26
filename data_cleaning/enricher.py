import pandas as pd   # <-- ESTA LINEA FALTABA !
def enrich(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["description"] = df["description"].fillna("No description").astype(str).str.strip()
    df["auto_category"] = df["auto_category"].fillna("Otros").astype(str).str.strip()
    df["amount_signed"] = pd.to_numeric(df["amount_signed"], errors="coerce").fillna(0)

    def make_text(row):
        tr_type = "expense" if row["amount_signed"] < 0 else "income"
        amt = abs(row["amount_signed"])
        date_str = row["date"].strftime("%Y-%m-%d")
        return (
            f"On {date_str}, a {tr_type} of {amt:.2f} EUR "
            f"at '{row['description']}' categorized as '{row['auto_category']}'."
        )

    df["RAG_Text"] = df.apply(make_text, axis=1)
    return df
