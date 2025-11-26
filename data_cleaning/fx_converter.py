# ============================================================
# fx_converter.py — Offline USD→EUR converter
# ============================================================

import pandas as pd
import os

CACHE_PATH = "data/fx_rates.csv"

def load_fx_rates():
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            "FX rates file not found. Expected at data/fx_rates.csv"
        )

    fx = pd.read_csv(CACHE_PATH, parse_dates=["Date"])
    fx["USD"] = pd.to_numeric(fx["USD"], errors="coerce")
    fx = fx.dropna()
    return fx


def usd_to_eur(amount, date, fx):
    if amount is None or pd.isna(amount):
        return None

    date = pd.to_datetime(date, errors="coerce")
    if pd.isna(date):
        return None

    # Find closest previous rate
    row = (
        fx.loc[fx["Date"] <= date]
        .sort_values("Date")
        .tail(1)["USD"]
    )

    if row.empty:
        return None

    rate = float(row.iloc[0])
    if rate == 0:
        return None

    return float(amount) / rate


def convert_usd_to_eur(df):
    df = df.copy()

    if "source" not in df.columns:
        return df

    mask = df["source"] == "BG"

    if mask.sum() == 0:
        df["currency"] = "EUR"
        return df

    fx = load_fx_rates()

    df["amount_eur"] = df.apply(
        lambda row: usd_to_eur(row["amount"], row["date"], fx)
        if row["source"] == "BG"
        else row["amount"],
        axis=1,
    )

    df["amount"] = df["amount_eur"]
    df = df.drop(columns=["amount_eur"])
    df["currency"] = "EUR"

    return df
