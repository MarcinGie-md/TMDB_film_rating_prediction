import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# Ścieżki
DATA_IN = Path("data/processed/movies_features.csv")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

# Ustawienia
TOP_K = 50
RANDOM_STATE = 42

NUM_COLS = ["budget", "revenue", "runtime", "popularity", "vote_count", "release_year"]
CAT_COLS = ["main_genre", "actor_1", "actor_2", "actor_3", "director"]
TARGET = "vote_average"
TITLE_COL = "title"  # użyte do raportu predictions

def ensure_paths() -> None:
    if not DATA_IN.exists():
        raise FileNotFoundError(f"Brak pliku: {DATA_IN}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_IN)
    # release_date może w CSV być stringiem -> zamień na datetime, potem wyciągnij rok
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year
    else:
        raise KeyError("Brak kolumny 'release_date' potrzebnej do 'release_year'.")
    return df

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["vote_count"] > 90].copy()
    df = df[df["release_year"].notna()].copy()
    df["release_year"] = df["release_year"].astype(int)
    return df

def compute_topk(series: pd.Series, k: int) -> List[str]:
    return series.value_counts().head(k).index.tolist()

def apply_topk(series: pd.Series, topk: List[str], unknown_label="Unknown", other_label="Other") -> pd.Series:
    s = series.fillna(unknown_label).astype(str)
    return np.where(s.isin(topk), s, other_label)

def prepare_cats_with_topk(
    train_df: pd.DataFrame, other_df: pd.DataFrame, cat_cols: List[str], k: int
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Wylicza TOP-K na train_df i stosuje mapowanie do train_df i other_df.
    Zwraca (train_df2, other_df2, topk_map).
    """
    topk_map: Dict[str, List[str]] = {}
    train2 = train_df.copy()
    other2 = other_df.copy()
    for c in cat_cols:
        topk = compute_topk(train2[c].fillna("Unknown").astype(str), k)
        topk_map[c] = topk
        train2[c] = apply_topk(train2[c], topk)
        other2[c] = apply_topk(other_df[c], topk)
    return train2, other2, topk_map

def build_pipeline(alpha: float) -> Pipeline:
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ],
        remainder="drop",
    )
    model = Ridge(alpha=alpha)
    pipe = Pipeline(steps=[("preproc", preproc), ("model", model)])
    return pipe

def evaluate_mae(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    y_pred = pipe.predict(X)
    return mean_absolute_error(y, y_pred)

def main():
    ensure_paths()
    df = load_data()
    df = filter_data(df)

    # Przygotowanie bazowych kolumn kategorycznych — wypełnij braki już teraz
    for c in CAT_COLS:
        if c not in df.columns:
            raise KeyError(f"Brak wymaganej kolumny kategorycznej: {c}")
        df[c] = df[c].fillna("Unknown").astype(str)

    # Podział: najpierw test 15%, potem z reszty walidacja ≈ 10% całości
    trainval_df, test_df = train_test_split(
        df, test_size=0.15, random_state=RANDOM_STATE, shuffle=True
    )
    # val ma stanowić ~10% całości -> w relacji do 85% reszty to 10/85
    val_rel = 10 / 85
    train_df, val_df = train_test_split(
        trainval_df, test_size=val_rel, random_state=RANDOM_STATE, shuffle=True
    )

    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # TOP-K na train; zastosuj na train i val
    train_df, val_df, topk_map = prepare_cats_with_topk(train_df, val_df, CAT_COLS, TOP_K)

    # Zestawy dla strojenia
    X_train, y_train = train_df[NUM_COLS + CAT_COLS], train_df[TARGET]
    X_val, y_val = val_df[NUM_COLS + CAT_COLS], val_df[TARGET]

    # Krok 1: gruby przegląd alpha
    coarse_alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]
    results = []
    for a in coarse_alphas:
        pipe = build_pipeline(alpha=a).fit(X_train, y_train)
        mae = evaluate_mae(pipe, X_val, y_val)
        results.append((a, mae))
    best_alpha, best_mae = min(results, key=lambda t: t[1])
    print(f"[Coarse] best alpha={best_alpha} (val MAE={best_mae:.4f})")

    # Krok 2: doprecyzowanie w okolicy najlepszego
    multipliers = [1/3, 1/2, 1, 2, 3]
    refine_alphas = sorted(set(max(1e-8, best_alpha*m) for m in multipliers))
    results_refined = []
    for a in refine_alphas:
        pipe = build_pipeline(alpha=a).fit(X_train, y_train)
        mae = evaluate_mae(pipe, X_val, y_val)
        results_refined.append((a, mae))
    best_alpha_refined, best_mae_refined = min(results_refined, key=lambda t: t[1])
    print(f"[Refine] best alpha={best_alpha_refined} (val MAE={best_mae_refined:.4f})")

    # Po wyborze alpha -> łączymy train + val, ponownie wyznaczamy TOP-K (więcej danych, nadal bez testu)
    train_full = pd.concat([train_df, val_df], axis=0)
    test_df = test_df.copy()

    # Zastosuj topk na pełnym trainie oraz test (nowe topk z train_full)
    train_full, test_df, topk_map_full = prepare_cats_with_topk(train_full, test_df, CAT_COLS, TOP_K)

    # Finalne zestawy
    X_train_full, y_train_full = train_full[NUM_COLS + CAT_COLS], train_full[TARGET]
    X_test, y_test = test_df[NUM_COLS + CAT_COLS], test_df[TARGET]

    # Finalny pipeline z najlepszym alpha
    final_alpha = float(best_alpha_refined)
    final_pipe = build_pipeline(alpha=final_alpha).fit(X_train_full, y_train_full)

    # Predykcje test i metryki
    y_pred = final_pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    # Baseline: średnia z train_full
    baseline_mean = float(y_train_full.mean())
    y_pred_base = np.full_like(y_test, fill_value=baseline_mean, dtype=float)
    mae_base = mean_absolute_error(y_test, y_pred_base)
    improvement_pct = 100.0 * (mae_base - mae) / mae_base if mae_base > 0 else np.nan

    # Raporty per genre
    test_out = pd.DataFrame({
        "title": test_df.get(TITLE_COL, pd.Series(index=test_df.index, dtype=str)),
        "main_genre": test_df["main_genre"],
        "y_true": y_test,
        "y_pred": y_pred,
    })
    test_out["abs_error"] = (test_out["y_true"] - test_out["y_pred"]).abs()

    mae_by_genre = test_out.groupby("main_genre")["abs_error"].mean().sort_values().rename("mae").reset_index()

    # Zapis artefaktów
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics = {
        "alpha": final_alpha,
        "sizes": {"train": int(len(train_full)), "test": int(len(test_df))},
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "baseline_mean": baseline_mean,
        "mae_baseline": float(mae_base),
        "improvement_pct": float(improvement_pct),
        "topk": {"K": TOP_K, "columns": CAT_COLS},
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # mae_by_genre.csv
    mae_by_genre.to_csv(REPORTS_DIR / "mae_by_genre.csv", index=False)

    # predictions_test.csv
    test_out.to_csv(REPORTS_DIR / "predictions_test.csv", index=False)

    # feature_space.txt (liczba kolumn po One-Hot)
    # dopasuj preproc, aby policzyć wymiar przestrzeni cech:
    preproc_only = final_pipe.named_steps["preproc"]
    if hasattr(preproc_only, "get_feature_names_out"):
        feat_names = preproc_only.get_feature_names_out()
        feat_count = len(feat_names)
    else:
        # fallback
        feat_count = -1
        feat_names = []
    with open(REPORTS_DIR / "feature_space.txt", "w", encoding="utf-8") as f:
        f.write(f"Feature count after preprocessing: {feat_count}\n")
        if feat_names is not None and len(feat_names) > 0:
            f.write("First 50 feature names (preview):\n")
            for name in list(feat_names)[:50]:
                f.write(f"- {name}\n")

    # Zapis modelu
    joblib.dump(final_pipe, MODELS_DIR / "ridge.pkl")

    print("\n=== RESULTS ===")
    print(f"Best alpha (refined): {final_alpha}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2:   {r2:.4f}")
    print(f"Baseline MAE: {mae_base:.4f}  |  Improvement: {improvement_pct:.2f}%")
    print("Saved: reports/metrics.json, mae_by_genre.csv, predictions_test.csv, feature_space.txt, models/ridge.pkl")

if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 140)
    main()
