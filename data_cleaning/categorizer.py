# ============================================================
# categorizer.py — Assign automatic category based on regex
# ============================================================

import re
import pandas as pd

def normalize_text(x):
    return str(x).lower().strip()

CATEGORY_PATTERNS = {
    "Supermercado": [
        r"mercadona", r"eroski", r"primaprix", r"bm\b", r"dia\b",
        r"fruter", r"super 99", r"super xtra"
    ],
    "Restaurantes": [
        r"restaurant", r"restaurante", r"kebab", r"arepa", r"doner",
        r"fusion market", r"good burger", r"pizz"
    ],
    "Bares / Cafés": [
        r"cafe", r"cafetea", r"macchiato", r"bar\b", r"cerve", r"pub"
    ],
    "Entretenimiento": [
        r"cine", r"golem", r"evento", r"festival", r"club", r"zentral",
        r"concierto"
    ],
    "Moda / Ropa": [
        r"zara", r"bershka", r"pull", r"moda", r"outlet"
    ],
    "Compras Online": [
        r"amazon", r"amzn", r"vinted", r"temu", r"zara\.com"
    ],
    "Gimnasio / Deporte": [
        r"vivagym", r"ayuntamiento", r"piscina", r"strong club"
    ],
    "Cuidado Personal": [
        r"peluquer", r"beauty", r"barber", r"estética"
    ],
    "Suscripciones": [
        r"spotify", r"openai", r"apple\.com", r"itunes", r"amazon prime",
        r"google one"
    ],
    "Salud": [
        r"farmacia", r"clinic", r"denti"
    ],
    "Transporte / Viajes": [
        r"ride on", r"uber", r"taxi", r"aero", r"trainline",
        r"mytrip", r"renfe", r"blablacar"
    ],
    "Transferencias": [
        r"bizum", r"yappy", r"transferencia\b"
    ],
    "MEC": [
        r"entre cuentas", r"sanchez castillo.*sanchez castillo",
        r"leonardo enrique.*leonardo"
    ],
    "ATM / Efectivo": [
        r"atm", r"ingreso\s+contra\s+cuenta"
    ],
}


def assign_category(desc):
    text = normalize_text(desc)
    for cat, patterns in CATEGORY_PATTERNS.items():
        for p in patterns:
            if re.search(p, text):
                return cat
    return "Otros"

def categorize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Find the best description-like column
    desc_col = None
    for c in df.columns:
        if str(c).lower() in ["description", "transaction description"]:
            desc_col = c
            break

    if desc_col is None:
        raise KeyError("categorize(): no description column found in dataframe.")

    # FIX: must be "auto_category" (lowercase)
    df["auto_category"] = df[desc_col].apply(assign_category)

    return df


