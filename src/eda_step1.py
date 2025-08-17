import os
import pandas as pd
import numpy as np

MOVIES_PATH = os.path.join("data", "raw", "tmdb_5000_movies.csv")
CREDITS_PATH = os.path.join("data", "raw", "tmdb_5000_credits.csv")

def basic_overview(df: pd.DataFrame, name: str):
    print(f"\n=== [{name}] BASIC OVERVIEW ===")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print("Columns:", list(df.columns))
    print("\nDtypes:")
    print(df.dtypes.sort_index())

    print("\nHead (3 rows):")
    print(df.head(3))

def missing_report(df: pd.DataFrame, name: str):
    print(f"\n=== [{name}] MISSING VALUES ===")
    missing = df.isna().sum().sort_values(ascending=False)
    pct = (missing / len(df) * 100).round(2)
    rep = pd.DataFrame({"missing": missing, "missing_%": pct})
    print(rep[rep["missing"] > 0])

def detect_json_like(df: pd.DataFrame, name: str):
    print(f"\n=== [{name}] JSON/ARRAY-LIKE COLUMNS (heurystyka) ===")
    json_like_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            s = df[col].dropna().astype(str)
            if len(s) == 0:
                continue
            # udział wartości zaczynających się od '[' lub '{'
            starts = s.str.strip().str.startswith(("[", "{")).mean()
            if starts > 0.5:  # prosta heurystyka
                json_like_cols.append((col, round(starts, 2)))
    if json_like_cols:
        for col, ratio in json_like_cols:
            print(f"- {col}: {ratio*100}% wierszy wygląda na listę/JSON")
    else:
        print("Brak wyraźnie JSON-owych kolumn wg heurystyki.")

def detect_binary_like(df: pd.DataFrame, name: str, max_unique: int = 2):
    print(f"\n=== [{name}] BINARY-LIKE COLUMNS (≤{max_unique} unikalnych) ===")
    candidates = []
    for col in df.columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        unique_vals = pd.unique(non_null)
        if len(unique_vals) <= max_unique:
            candidates.append((col, len(unique_vals), unique_vals[:10]))
    if candidates:
        for col, k, vals in candidates:
            print(f"- {col}: {k} unikalnych (przykłady: {list(vals)})")
    else:
        print("Brak oczywistych binarnych kolumn.")

def detect_numeric_as_text(df: pd.DataFrame, name: str, sample_rows: int = 500):
    print(f"\n=== [{name}] NUMERIC-AS-TEXT (heurystyka) ===")
    suspects = []
    for col in df.columns:
        if df[col].dtype == "object":
            s = df[col].dropna().astype(str)
            if s.empty:
                continue
            sample = s.sample(min(sample_rows, len(s)), random_state=42)
            # policz udział wartości, które da się zrzutować na liczbę
            coerced = pd.to_numeric(sample, errors="coerce")
            ratio_numeric = coerced.notna().mean()
            if 0.6 <= ratio_numeric < 1.0:  # jeśli większość wygląda na liczby
                suspects.append((col, round(ratio_numeric, 2)))
    if suspects:
        for col, ratio in suspects:
            print(f"- {col}: ~{ratio*100}% wartości wygląda na numeryczne (zapisane jako tekst)")
    else:
        print("Brak oczywistych kolumn numerycznych zapisanych jako tekst.")

def check_dates(df: pd.DataFrame, name: str, date_cols=("release_date",)):
    print(f"\n=== [{name}] DATE PARSE CHECK ===")
    for col in date_cols:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            ok_ratio = parsed.notna().mean()
            print(f"- {col}: parsowalne ~{round(ok_ratio*100,2)}% wartości")
        else:
            print(f"- {col}: kolumna nie występuje")

def main():
    # --- Wczytanie danych ---
    if not os.path.exists(MOVIES_PATH):
        raise FileNotFoundError(f"Nie znaleziono pliku: {MOVIES_PATH}")
    if not os.path.exists(CREDITS_PATH):
        raise FileNotFoundError(f"Nie znaleziono pliku: {CREDITS_PATH}")

    movies = pd.read_csv(MOVIES_PATH)
    credits = pd.read_csv(CREDITS_PATH)

    # --- Raport dla MOVIES ---
    basic_overview(movies, "MOVIES")
    missing_report(movies, "MOVIES")
    detect_json_like(movies, "MOVIES")
    detect_binary_like(movies, "MOVIES")
    detect_numeric_as_text(movies, "MOVIES")
    check_dates(movies, "MOVIES", date_cols=("release_date",))

    # --- Raport dla CREDITS ---
    basic_overview(credits, "CREDITS")
    missing_report(credits, "CREDITS")
    detect_json_like(credits, "CREDITS")
    detect_binary_like(credits, "CREDITS")
    detect_numeric_as_text(credits, "CREDITS")
    # w credits zwykle nie ma dat, więc pomijamy

    print("\n=== DONE: EDA STEP 1 (diagnoza) ===")
    print("Wnioski na kolejny krok: które kolumny chcemy konwertować/usunąć?")

if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 120)
    main()
