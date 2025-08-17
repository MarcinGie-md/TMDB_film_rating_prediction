import os
import pandas as pd
from pathlib import Path

RAW_MOVIES = Path("data/raw/tmdb_5000_movies.csv")
RAW_CREDITS = Path("data/raw/tmdb_5000_credits.csv")
PROC_DIR = Path("data/processed")
PROC_MOVIES = PROC_DIR / "movies_clean.csv"
PROC_CREDITS = PROC_DIR / "credits_clean.csv"

def ensure_paths():
    if not RAW_MOVIES.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {RAW_MOVIES}")
    if not RAW_CREDITS.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {RAW_CREDITS}")
    PROC_DIR.mkdir(parents=True, exist_ok=True)

def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Usuń mało-przydatną kolumnę z dużą liczbą braków
    if "homepage" in df.columns:
        df = df.drop(columns=["homepage"])

    # 2) release_date -> datetime (braki jako NaT)
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    # 3) overview -> braki jako pusty string
    if "overview" in df.columns:
        df["overview"] = df["overview"].fillna("")

    # 4) runtime -> braki medianą (po bezpiecznej konwersji do liczby)
    if "runtime" in df.columns:
        df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
        median_runtime = df["runtime"].median()
        df["runtime"] = df["runtime"].fillna(median_runtime)

    return df

def copy_credits(df: pd.DataFrame) -> pd.DataFrame:
    # Na tym etapie bez zmian – zostawiamy „as-is”.
    # (Później ewentualnie zrobimy parsowanie JSON-ów: cast/crew.)
    return df

def report(df: pd.DataFrame, name: str):
    print(f"\n=== [{name}] AFTER CLEANING ===")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    missing = df.isna().sum().sort_values(ascending=False)
    pct = (missing / len(df) * 100).round(2)
    rep = pd.DataFrame({"missing": missing, "missing_%": pct})
    # Pokaż tylko te kolumny, gdzie są braki
    rep = rep[rep["missing"] > 0]
    if rep.empty:
        print("Braki: BRAK")
    else:
        print("Braki (kolumny z NaN):")
        print(rep)

def main():
    ensure_paths()

    movies = pd.read_csv(RAW_MOVIES)
    credits = pd.read_csv(RAW_CREDITS)

    movies_clean = clean_movies(movies)
    credits_clean = copy_credits(credits)

    # Zapis
    movies_clean.to_csv(PROC_MOVIES, index=False)
    credits_clean.to_csv(PROC_CREDITS, index=False)

    # Raport końcowy
    report(movies_clean, "MOVIES_CLEAN")
    report(credits_clean, "CREDITS_CLEAN")

    print(f"\nZapisano:\n- {PROC_MOVIES}\n- {PROC_CREDITS}")
    print("\nDONE: EDA STEP 2 (clean basic)")

if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 120)
    main()
