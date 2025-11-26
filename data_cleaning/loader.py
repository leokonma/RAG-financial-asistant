import pandas as pd
import re
from datetime import datetime

# =====================================================
# UTILITIES
# =====================================================

def clean_amount(x):
    """Convert '1.234,56' or '−3,20' or '-3.20' → float."""
    if pd.isna(x):
        return None
    x = str(x).strip()
    x = x.replace("−", "-")
    x = x.replace(",", ".")
    try:
        return float(x)
    except:
        return None


def clean_date(x):
    """Clean date in various formats."""
    if pd.isna(x):
        return None
    try:
        return pd.to_datetime(x, errors="coerce")
    except:
        return None


def _drop_unnamed(df):
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]

# =====================================================
# BG LOADER  (Banco General Panamá)
# =====================================================

def load_BG(path):
    print("📘 Loading BG...")

    df_raw = pd.read_excel(path, header=None)

    # Find header row dynamically
    header_row = df_raw[
        df_raw.apply(lambda r: r.astype(str).str.contains("Fecha|Descripción|Saldo", case=False).any(), axis=1)
    ].index[0]

    df = pd.read_excel(path, header=header_row)
    df = _drop_unnamed(df)
    df.columns = [str(c).strip() for c in df.columns]

    # Find amount column (BG does not label debit/credit correctly)
    amount_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ["débito", "debito", "crédito", "credito", "importe", "monto"]):
            amount_col = col
            break

    if amount_col is None:
        raise ValueError("❌ No 'Débito/Crédito/Importe/Monto' column found in BG file.")

    df["amount_raw"] = df[amount_col].apply(clean_amount)

    # Detect sign from description text
    def detect_sign(row):
        desc = str(row.get("description", "")).lower()

        income_kw = ["yappy de", "transferencia de", "depósito", "deposito", "interes", "abono"]
        expense_kw = ["compra", "pago", "cargo", "itbms", "super", "estanco", "beer",
                      "google", "spotify", "farmacia", "vinted", "kebab", "uber"]

        if any(k in desc for k in income_kw):
            return row["amount_raw"]

        if any(k in desc for k in expense_kw):
            return -row["amount_raw"]

        # default: assume expense
        return -row["amount_raw"]

    df["amount"] = df.apply(detect_sign, axis=1)

    # Clean date
    if "date" in df.columns:
        df["date"] = df["date"].apply(clean_date)
    else:
        # find the column automatically
        for col in df.columns:
            if "fecha" in col.lower():
                df["date"] = df[col].apply(clean_date)
                break

    df["currency"] = "EUR"
    df["source"] = "BG"

    keep = ["date", "description", "amount", "currency", "source"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=["date"])

    return df

# =====================================================
# SD LOADER  (Santander España)
# =====================================================

def load_SD(path):
    print("📕 Loading SD...")

    raw = pd.read_excel(path, header=None)
    header_row = 6

    df = pd.read_excel(path, header=header_row)
    df = _drop_unnamed(df)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "Fecha operación": "date",
        "Fecha operacion": "date",
        "Fecha valor": "date_value",
        "Concepto": "description",
        "Importe": "import",
        "Saldo": "balance",
        "Divisa": "currency",
    }

    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    if "import" not in df.columns:
        raise ValueError("❌ Santander file missing 'Importe' field.")

    df["amount"] = df["import"].apply(clean_amount)
    df["date"] = df["date"].apply(clean_date)

    df["currency"] = "EUR"
    df["source"] = "SD"

    keep = ["date", "description", "amount", "currency", "source", "balance"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna(subset=["date"])

    return df

# =====================================================
# COMBINED LOADER
# =====================================================

def load_all(bg_path: str, sd_path: str) -> pd.DataFrame:
    print("📥 Loading raw datasets...")

    bg = load_BG(bg_path)
    sd = load_SD(sd_path)

    df = pd.concat([bg, sd], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)

    return df
