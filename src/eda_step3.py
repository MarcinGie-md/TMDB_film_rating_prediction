import os
import ast
import pandas as pd
from pathlib import Path
from typing import Any, List, Optional

PROC_MOVIES_IN = Path("data/processed/movies_clean.csv")
PROC_CREDITS_IN = Path("data/processed/credits_clean.csv")
PROC_OUT = Path("data/processed/movies_features.csv")

def ensure_paths():
    if not PROC_MOVIES_IN.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {PROC_MOVIES_IN}")
    if not PROC_CREDITS_IN.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {PROC_CREDITS_IN}")
    PROC_OUT.parent.mkdir(parents=True, exist_ok=True)

def safe_parse(value: Any) -> Any:
    """
    Parsuje tekst reprezentujący listę/dict do obiektu Pythona.
    Zwraca oryginalną wartość, jeśli nie jest stringiem JSON-owatym.
    """
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") or value.startswith("{"):
            try:
                return ast.literal_eval(value)
            except Exception:
                return None
    return value

def extract_genres(row_val: Any) -> List[str]:
    """
    Oczekuje listy słowników z polami 'id' i 'name'.
    Zwraca listę nazw gatunków.
    """
    parsed = safe_parse(row_val)
    if isinstance(parsed, list):
        names = []
        for item in parsed:
            if isinstance(item, dict) and "name" in item:
                names.append(str(item["name"]))
        return names
    return []

def main_genre_from_list(genres_list: List[str]) -> Optional[str]:
    return genres_list[0] if genres_list else None

def extract_first_n_cast(cast_val: Any, n: int = 3) -> List[Optional[str]]:
    """
    Zwraca listę [actor_1, actor_2, actor_3] (lub mniej, jeśli brak).
    Oczekuje listy słowników z kluczami w stylu {'name': '...'}.
    """
    parsed = safe_parse(cast_val)
    out = []
    if isinstance(parsed, list):
        for i in range(min(n, len(parsed))):
            name = None
            item = parsed[i]
            if isinstance(item, dict) and "name" in item:
                name = str(item["name"])
            out.append(name)
    # dopełnij None do długości n
    while len(out) < n:
        out.append(None)
    return out

def extract_director(crew_val: Any) -> Optional[str]:
    """
    Szuka w crew osoby z job == 'Director' i zwraca jej 'name'.
    """
    parsed = safe_parse(crew_val)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and item.get("job") == "Director":
                name = item.get("name")
                return str(name) if name is not None else None
    return None

def drop_tagline(df: pd.DataFrame) -> pd.DataFrame:
    if "tagline" in df.columns:
        return df.drop(columns=["tagline"])
    return df

def build_features(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    # 1) Movies: usuń tagline
    movies = drop_tagline(movies).copy()

    # 2) Genres -> genres_list + main_genre
    if "genres" in movies.columns:
        genres_list = movies["genres"].apply(extract_genres)
        movies["genres_list"] = genres_list
        movies["main_genre"] = genres_list.apply(main_genre_from_list)

    # 3) Credits: actor_1..3 + director
    credits = credits.copy()
    if "cast" in credits.columns:
        actors_123 = credits["cast"].apply(lambda x: extract_first_n_cast(x, n=3))
        credits["actor_1"] = actors_123.apply(lambda xs: xs[0])
        credits["actor_2"] = actors_123.apply(lambda xs: xs[1])
        credits["actor_3"] = actors_123.apply(lambda xs: xs[2])

    if "crew" in credits.columns:
        credits["director"] = credits["crew"].apply(extract_director)

    # 4) Join po id / movie_id
    on_left = "id" if "id" in movies.columns else None
    on_right = "movie_id" if "movie_id" in credits.columns else None
    if not on_left or not on_right:
        raise KeyError("Brakuje kolumn łączących: movies.id lub credits.movie_id")

    # wybierz tylko potrzebne kolumny z credits, by nie duplikować dużych JSON-ów
    credits_small = credits[[on_right, "actor_1", "actor_2", "actor_3", "director"]].copy()

    merged = movies.merge(credits_small, left_on=on_left, right_on=on_right, how="left")
    merged = merged.drop(columns=[on_right])  # nie potrzebujemy duplicate key
    return merged

def report(df: pd.DataFrame):
    print("\n=== FEATURES SUMMARY ===")
    cols = ["genres", "genres_list", "main_genre", "actor_1", "actor_2", "actor_3", "director"]
    existing = [c for c in cols if c in df.columns]
    print("Podgląd kolumn cech:", existing)
    print("\nHead (3 rows):")
    print(df[existing].head(3))

def main():
    ensure_paths()
    movies = pd.read_csv(PROC_MOVIES_IN)
    credits = pd.read_csv(PROC_CREDITS_IN)

    features = build_features(movies, credits)

    features.to_csv(PROC_OUT, index=False)
    print(f"Zapisano: {PROC_OUT}")

    report(features)
    print("\nDONE: EDA STEP 3 (features)")

if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 120)
    main()
