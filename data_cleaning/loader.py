import pandas as pd
import re

# =====================================================
# BG LOADER  (Banco General Panamá)
# =====================================================

def load_BG(path):
    print("📘 Loading BG (clean version)...")

    # 1. Load real table starting at row 8
    df = pd.read_excel(path, skiprows=7, header=0)

    # 2. Drop irrelevant columns (confirmed from notebook)
    df = df.drop(columns=[
        "Unnamed: 1",
        "Referencia",
        "Transacción",
        "Unnamed: 7",
        "Saldo total"
    ], errors="ignore")

    # Remaining columns now:
    # Fecha | Descripción | Débito | Crédito

    # 3. Collapse debit + credit into single amount
    def collapse_amount(row):
        if pd.notna(row.get("Débito")):
            return row["Débito"]
        if pd.notna(row.get("Crédito")):
            return row["Crédito"]
        return None

    df["amount"] = df.apply(collapse_amount, axis=1)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # 4. Convert Fecha
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    # 5. Clean description
    df["Descripción"] = df["Descripción"].astype(str).str.strip()

    # 6. Add metadata
    df["currency"] = "USD"   # Later converted to EUR in convert_usd_to_eur
    df["source"] = "BG"

    # 7. Rename AFTER cleaning
    df = df.rename(columns={
        "Fecha": "date",
        "Descripción": "description"
    })

    # 8. Final column order
    df = df[["date", "description", "amount", "currency", "source"]]

    # 9. Remove invalid rows
    df = df.dropna(subset=["date", "amount"])

    # 10. Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =====================================================
# SD LOADER  (Santander España)
# =====================================================

def load_SD(path):
    print("📕 Loading SD (clean version)...")

    # 1. Load real table starting at row 7
    df = pd.read_excel(path, skiprows=6, header=0)

    # Raw columns:
    # Fecha operación | Fecha valor | Concepto | Importe | Saldo | Divisa

    # ---- CONVERT FIRST ----

    # 2. Convert amount
    def clean_amount_raw(x):
        if pd.isna(x):
            return None
        x = str(x)
        x = x.replace("EUR", "").replace("€", "")
        x = x.replace("−", "-")
        x = x.replace(",", ".")
        x = x.strip()
        match = re.findall(r"-?\d+\.?\d*", x)
        return float(match[0]) if match else None

    df["Importe"] = df["Importe"].apply(clean_amount_raw)
    df["Importe"] = pd.to_numeric(df["Importe"], errors="coerce")

    # 3. Convert date
    df["Fecha operación"] = pd.to_datetime(df["Fecha operación"], dayfirst=True, errors="coerce")

    # 4. Clean description
    df["Concepto"] = df["Concepto"].astype(str).str.strip()

    # 5. Clean currency
    df["Divisa"] = df["Divisa"].astype(str).str.strip()

    # 6. Add metadata
    df["source"] = "SD"

    # 7. Drop useless columns BEFORE renaming
    df = df.drop(columns=["Fecha valor", "Saldo"], errors="ignore")

    # ---- RENAME LATER ----

    df = df.rename(columns={
        "Fecha operación": "date",
        "Concepto": "description",
        "Importe": "amount",
        "Divisa": "currency"
    })

    # ---- FINAL ORDER ----

    df = df[["date", "description", "amount", "currency", "source"]]

    # Filter invalid rows
    df = df.dropna(subset=["date", "amount"])

    # Sort
    df = df.sort_values("date").reset_index(drop=True)

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
