# TMDB_film_rating_prediction

Celem projektu jest zbudowanie prostego i czytelnego procesu przewidywania ocen filmów (`vote_average`) na danych TMDB, w sposób atrakcyjny z perspektywy rekrutacji na stanowiska juniorskie w obszarze AI/ML. Repozytorium akcentuje porządek pracy, terminal-first, EDA, świadomy dobór cech, regularizację oraz klarowne raportowanie metryk.

---

## Założenia i styl pracy
- Minimalizm i przejrzystość: małe, logiczne etapy, każdy z jasno opisanym celem.
- Terminal-first: wszystkie kroki uruchamiane z konsoli.
- Reprodukowalność: skrypty działają w izolowanym środowisku (`venv`).
- Ewaluacja > fajerwerki: stawiamy na zrozumiały model i klarowne metryki zamiast złożoności.

---

## Dane i filtr
Pliki źródłowe znajdują się w `data/raw/`.  
Na etapie modelowania uwzględniamy wyłącznie filmy z **`vote_count > 90`**.  
Pojedyncze braki `release_date` usuwamy przed featuryzacją.

---

## Zrealizowane etapy

### Step 1 — Diagnoza (`src/eda_step1.py`)
- Przegląd kształtu i typów danych, braków oraz kolumn w formacie list/JSON.
- Wnioski do dalszego czyszczenia i parsowania.

### Step 2 — Clean (`src/eda_step2.py`)
- Usunięto kolumny o niskiej wartości informacyjnej i/lub z dużym odsetkiem braków.
- Ujednolicenie typów (`release_date` → datetime), uzupełnienia drobnych braków.
- Zapis: `data/processed/movies_clean.csv`, `data/processed/credits_clean.csv`.

### Step 3 — Cechy (`src/eda_step3.py`)
- Parsowanie struktur:
  - `genres` → `genres_list` (lista nazw), `main_genre` (pierwszy gatunek),
  - `cast` → `actor_1`, `actor_2`, `actor_3` (pierwsze trzy nazwiska),
  - `crew` → `director`.
- Połączenie danych po identyfikatorze filmu.
- Zapis: `data/processed/movies_features.csv`.

---

## Cechy używane w modelu (plan)
- **Numeryczne:** `budget`, `revenue`, `runtime`, `popularity`, `vote_count`, `release_year` (z `release_date`).
- **Kategoryczne:** `main_genre` (One-Hot).
- **Obsada/crew:** `actor_1`, `actor_2`, `actor_3`, `director` — One-Hot **dla TOP-K=50** najczęstszych wartości w każdej kolumnie; pozostałe jako `Other`.  
- Braki kategorii → `Unknown`. Numeryczne standaryzowane w pipeline.

---

## Model i walidacja (plan)
- **Model:** Ridge Regression (regularizacja `L2`).
- **Podział danych:** **train/val/test ≈ 75% / 10% / 15%**  
  - najpierw wydzielany test = 15% (stały `random_state`),  
  - z pozostałych 85% wydzielane val = 10% całego zbioru.
- **Strojenie `alpha` (bez K-fold, na walidacji):**
  1. Przegląd szeroki: `[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]`
  2. Doprecyzowanie wokół najlepszego (np. ×{1/3, 1/2, 1, 2, 3})
- Po wyborze `alpha`: trening na **train+val (85%)**, końcowa ocena na **teście (15%)**.

---

## Metryki i raportowanie (plan)
- **Główna metryka:** MAE (łatwa interpretacja w skali 0–10).
- **Dodatkowe:** RMSE, R².
- **Baseline:** predykcja średniej z treningu; raportujemy **MAE_baseline** i **% poprawy** względem baseline’u.
- **Agregacje:** MAE per `main_genre` (test), dla krótkiego porównania jakości w segmentach.

**Artefakty po treningu:**
- `reports/metrics.json` — MAE, RMSE, R², baseline, wybrane `alpha`
- `reports/mae_by_genre.csv`
- `reports/predictions_test.csv` — `title`, `y_true`, `y_pred`, `abs_error`
- `reports/feature_space.txt` — liczba kolumn po One-Hot
- `models/ridge.pkl` — zapisany model (opcjonalnie)

---

## Struktura repo (skrót)
TMDB_film_rating_prediction/
├─ data/
│ ├─ raw/
│ └─ processed/
├─ models/
├─ reports/
├─ src/
│ ├─ eda_step1.py
│ ├─ eda_step2.py
│ └─ eda_step3.py
├─ .gitignore
├─ README.md
└─ requirements.txt

---

## Uruchomienie (terminal)
```powershell
# środowisko
cd C:\Users\Administrator\TMDB_film_rating_prediction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# EDA / featuryzacja
python src\eda_step1.py
python src\eda_step2.py
python src\eda_step3.py

# Trening (zostanie dodany)
# python src\train_ridge.py
