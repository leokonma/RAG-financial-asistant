from data_cleaning import load_all, normalize, categorize, enrich, convert_usd_to_eur
import os

OUTPUT_PATH = "data/Finance_Processed.csv"


def run_pipeline():
    print("📥 Loading raw datasets...")
    df = load_all(
        "data/BG_Transaccions.xlsx",
        "data/SD_Transaccions.xlsx",
    )

    print("💱 Converting USD → EUR...")
    df = convert_usd_to_eur(df)

    print("🧼 Normalizing data...")
    df = normalize(df)

    print("🏷️ Categorizing transactions...")
    df = categorize(df)

    print("📈 Enriching for RAG...")
    df = enrich(df)

    print("💾 Saving final dataset...")
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()
