import pandas as pd
import re

# ============================================================
# SAFE STRING CLEANER
# ============================================================
def safe_strip(x):
    """Safely strip strings."""
    if isinstance(x, str):
        return x.strip()
    return x


# ============================================================
# DATE CLEANER
# ============================================================
def clean_date(x):
    """
    Normalize inconsistent date formats from Santander (SD) and BG.
    Accepts formats like:
    - 26/11/2025
    - 26/11/2025 | 16:38:45
    - 2025-11-26
    """
    if pd.isna(x):
        return pd.NaT

    if isinstance(x, str):
        # Remove time if present
        x = x.split("|")[0].strip()

    return pd.to_datetime(x, dayfirst=True, errors="coerce")


# ============================================================
# AMOUNT CLEANER
# ============================================================
def clean_amount(x):
    """
    Normalize European and USD amounts:
    - Converts commas to dots
    - Removes currency symbols
    - Handles negative signs
    - Converts empty / "−3,20" / "-3.20" / "3,20 EUR" etc.
    """
    if pd.isna(x):
        return None

    if isinstance(x, str):
        x = x.replace("EUR", "").replace("$", "").replace("USD", "")
        x = x.replace("−", "-")         # special minus
        x = x.replace(",", ".")         # comma to dot
        x = x.strip()

        # Extract numbers using regex
        match = re.findall(r"-?\d+\.?\d*", x)
        if match:
            try:
                return float(match[0])
            except:
                return None

    if isinstance(x, (int, float)):
        return float(x)

    return None


# ============================================================
# TEXT NORMALIZER
# ============================================================
def clean_description(x):
    """Normalize transaction descriptions."""
    if pd.isna(x):
        return ""
    return str(x).strip()


# ============================================================
# COLUMN NAME STANDARDIZER
# ============================================================
def standardize_column_names(df):
    """
    Converts messy column names (SD & BG) into a normalized set:
    Target names:
        - date
        - description
        - amount
        - balance
        - source
    """
    mapping = {}

    for col in df.columns:
        col_clean = col.lower().strip()

        if "fecha" in col_clean:
            mapping[col] = "date"
        elif "concepto" in col_clean or "descrip" in col_clean:
            mapping[col] = "description"
        elif "importe" in col_clean or "débito" in col_clean or "credito" in col_clean:
            mapping[col] = "amount"
        elif "saldo" in col_clean:
            mapping[col] = "balance"
        else:
            mapping[col] = col  # keep original name

    df = df.rename(columns=mapping)
    return df
