# TMDB_film_rating_prediction

Celem projektu jest udokumentowanie prostego i czytelnego procesu tworzenia modelu do przewidywania ocen filmów (`vote_average`) na danych TMDB — z widocznym wykorzystaniem (git, EDA, ewaluacji, metryk).

---

## Założenia
- **Prosty pipeline**: surowe dane → czyszczenie → parsowanie cech → trening → metryki.
- **Filtr danych**: do modelu trafiają filmy z `vote_count > 90`.
- **Podział danych**: `train/val/test ≈ 75% / 10% / 15%` (najpierw wydzielany test = 15%).
- **Model**: Ridge Regression (regularizacja L2) z dwukrokowym strojeniem `alpha` na walidacji.

---

## Dane

Wejściowe pliki CSV znajdują się w `data/raw/`:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Pliki pośrednie trafiają do `data/processed/`.

---

## Zrealizowane etapy

**Step 1 — Diagnoza (`src/eda_step1.py`)**  
Przegląd typów, braków, rozpoznanie kolumn list/JSON.

**Step 2 — Clean (`src/eda_step2.py`)**  
Usunięte: kolumny o niskiej wartości; ujednolicone typy (m.in. `release_date`), uzupełnione drobne braki.  
Zapis: `data/processed/movies_clean.csv`, `data/processed/credits_clean.csv`.

**Step 3 — Cechy (`src/eda_step3.py`)**  
Parsowanie i łączenie danych:
- `genres` → `genres_list` + `main_genre`,
- `cast` → `actor_1`, `actor_2`, `actor_3`,
- `crew` → `director`,  
Merge po identyfikatorze filmu.  
Zapis: `data/processed/movies_features.csv`.

---

## Cechy używane w modelu

- **Numeryczne**: `budget`, `revenue`, `runtime`, `popularity`, `vote_count`, `release_year` (z `release_date`).  
- **Kategoryczne**: `main_genre`.  
- **Obsada/crew**: `actor_1`, `actor_2`, `actor_3`, `director`.  
- One-Hot dla **TOP-K=50** najczęstszych wartości w kolumnach kategorycznych (reszta = `Other`, braki = `Unknown`).  
- Numeryczne standaryzowane w pipeline (StandardScaler).

---

## Model i walidacja

- **Model**: Ridge Regression.  
- **Strojenie `alpha`** (bez K-fold, na walidacji):
  1) szeroki przegląd: `[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]`  
  2) doprecyzowanie wokół najlepszego (np. ×{1/3, 1/2, 1, 2, 3})  
- Finalny model trenowany na **train+val (≈85%)**, oceniany na **test (15%)**.

---

## Wyniki (Ridge)

Podział po filtrze `vote_count > 90`: train≈75%, val≈10%, test≈15%.  
Wybrane `alpha`: **5.0** (dwukrokowe strojenie na walidacji).

| Metryka | Test |
|---|---:|
| **MAE** | **0.474** |
| RMSE | 0.604 |
| R² | 0.462 |
| MAE baseline (średnia) | 0.665 |
| Poprawa vs baseline | **+28.73%** |

Artefakty:
- `reports/metrics.json` — metryki, `alpha`, rozmiary zbiorów
- `reports/mae_by_genre.csv` — MAE w podziale na `main_genre`
- `reports/predictions_test.csv` — `title`, `y_true`, `y_pred`, `abs_error`
- `reports/feature_space.txt` — liczba kolumn po One-Hot

---

## Uruchomienie (terminal)

```bash
# klonowanie
git clone https://github.com/MarcinGie-md/TMDB_film_rating_prediction.git
cd TMDB_film_rating_prediction

# środowisko
# Windows
python -m venv .venv
.venv\Scripts\activate
# macOS/Linux
# python3 -m venv .venv
# source .venv/bin/activate

pip install -r requirements.txt

# EDA / featuryzacja
python src/eda_step1.py
python src/eda_step2.py
python src/eda_step3.py

```bash
# trening i raporty
python src/train_ridge.py
```

Struktura repo (skrót)
TMDB_film_rating_prediction/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ models/
├─ reports/
├─ src/
│  ├─ eda_step1.py
│  ├─ eda_step2.py
│  ├─ eda_step3.py
│  └─ train_ridge.py
├─ .gitignore
├─ README.md
└─ requirements.txt
