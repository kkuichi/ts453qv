# Prediktívna analýza dát v oblasti spotreby elektrickej energie v domácnostiach

Diplomová práca – TUKE FEI Košice, 2026

Tento repozitár obsahuje praktickú časť diplomovej práce zameranej na predikciu hodnôt napätia a prúdu z dát smart meterov a detekciu anomálií prostredníctvom šiestich rôznych modelov strojového učenia.

## Štruktúra repozitára

```
.
├── Data/
│   ├── Druhe_data/      # vstupné dáta (tuke_poruchy_vzorky)
│   └── Export_data/     # spracované segmenty + obrázky z analýzy
├── Modely/              # výstupy z jednotlivých modelov (vytvárajú sa automaticky)
└── Workbook/            # Jupyter notebooky, analýza, predspracovanie a modely
```

### `Data/`

- **`Druhe_data/`** – vstupné dáta:
  - `tuke_poruchy_vzorky.parquet` – merania elektromerov (trojfázové napätie a prúd, 10-minútová granularita)
  - `tuke_poruchy_vzorky.csv` – evidencia porúch s dátumovými rozsahmi
- **`Export_data/`** – výstup z `Workbook.ipynb`:
  - `forecast_10min.parquet`, `forecast_30min.parquet`, `forecast_1h.parquet` – spracované dáta v troch granularitách
  - `segments.csv` – metadáta o použiteľných segmentoch
  - `Obrazky/` – grafy a vizualizácie z `Analyza.ipynb`

### `Modely/`

Pre každý model a granularitu sa automaticky vytvorí samostatný podpriečinok, napríklad:

```
Modely/
├── SARIMA/
│   ├── sarima_1h/
│   └── sarima_30min/
├── VAR/
│   ├── var_10min/
│   ├── var_1h/
│   └── var_30min/
├── LightGBM/
│   ├── lightgbm_10min/
│   ├── lightgbm_1h/
│   └── lightgbm_30min/
├── XGBoost/
│   ├── xgboost_10min/
│   ├── xgboost_1h/
│   └── xgboost_30min/
├── TemporalCNN/
│   ├── tcnn_1h/
│   └── tcnn_30min/
└── Transformer/
    ├── transformer_1h/
    └── transformer_30min/
```

### `Workbook/`

Všetky Jupyter notebooky:

| Súbor | Popis |
|---|---|
| `Workbook.ipynb` | Predspracovanie dát – načítanie, čistenie, segmentácia, agregácia a export |
| `Analyza.ipynb` | Exploratívna analýza dát – štatistiky, denné/sezónne profily, korelácie |
| `MODEL_1_SARIMA_*.ipynb` | SARIMA – klasický štatistický model pre časové rady |
| `MODEL_2_VAR_*.ipynb` | VAR – vektorová autoregresia (modeluje všetkých 6 cieľových premenných naraz) |
| `MODEL_3_LightGBM_*.ipynb` | LightGBM – gradient boosting framework |
| `MODEL_4_XGBoost_*.ipynb` | XGBoost – gradient boosting framework |
| `MODEL_5_TemporalCNN_*.ipynb` | Temporal CNN – konvolučná neurónová sieť pre časové rady (Bai et al., 2018) |
| `MODEL_6_Transformer_*.ipynb` | Transformer – encoder architektúra (Vaswani et al., 2017) |

## Dátový pipeline

Vstupné dáta prechádzajú pipelinom v `Workbook.ipynb`:

1. **Načítanie** parquet súboru s meraniami a CSV súboru s evidenciou porúch
2. **Priradenie** príznaku `Chyba` podľa dátumových rozsahov porúch
3. **Filtrovanie** VN (vysokonapäťových) elektromerov – pracujeme len s NN sieťou
4. **Oprava** chýbajúcich hodnôt (lineárna interpolácia s viacúrovňovým fallbackom)
5. **Odstránenie outlierov** metódou IQR (k = 3.0)
6. **Segmentácia** – rozdelenie na súvislé úseky bez prerušení a zmien stavu
7. **Agregácia** na 30-minútovú a 1-hodinovú granularitu
8. **Export** do parquet súborov pre tréning modelov

## Použité knižnice

### Základné

- `pandas`, `numpy` – práca s dátami
- `matplotlib`, `seaborn` – vizualizácie
- `scikit-learn` – metriky, škálovanie
- `scipy` – štatistické funkcie

### Modely

- `statsmodels` – SARIMA, VAR
- `lightgbm` – LightGBM
- `xgboost` – XGBoost
- `torch` – PyTorch pre Temporal CNN a Transformer
- `optuna` – hyperparameter tuning

### Pomocné

- `pyarrow` – parquet formát
- `tqdm` – progress bary

## Inštalácia

Odporúča sa Python 3.10 alebo novší. Závislosti sa dajú nainštalovať cez pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install statsmodels lightgbm xgboost optuna
pip install torch
pip install pyarrow tqdm
```

Pre tréning hlbokých modelov je odporúčaná NVIDIA GPU s podporou CUDA – beh na CPU je výrazne pomalší.

## Spustenie

Notebooky sa spúšťajú v nasledujúcom poradí:

1. **`Workbook.ipynb`** – vygeneruje spracované dáta do `Data/Export_data/`. Toto je nutné spustiť ako prvé.
2. **`Analyza.ipynb`** – exploratívna analýza, vytvorí grafy v `Data/Export_data/Obrazky/`. Voliteľné, ale odporúčané.
3. **Modely** – ľubovoľný `MODEL_*.ipynb`. Modely sú nezávislé, dajú sa spúšťať v ľubovoľnom poradí. Výsledky sa ukladajú do `Modely/<NazovModelu>/<nazov_granularita>/`.

Notebooky predpokladajú nasledujúcu štruktúru pracovného adresára:

```
Workbook/    ← notebook sa spúšťa odtiaľto
Data/
Modely/
```

Cesty v notebookoch sú relatívne (`../Data/...`, `../Modely/...`).

## Modely – spoločná logika

Všetky modely majú jednotnú štruktúru:

- **Chronologický split** segmentov 70/15/15 (train/val/test) – train je najstarší, test najnovší
- **Tréning** výhradne na čistých dátach (Chyba = 0)
- **Forecasting** s viacerými horizontmi predikcie (krátky → dlhý)
- **Anomaly detection** – model trénovaný na čistých dátach, prah `mean + 3σ` z reziduí na čistých dátach, detekcia na zmiešaných dátach (Chyba = 0 + Chyba = 1)
- **Štandardizované metriky**: MAE, RMSE, R² pre forecasting; Precision, Recall, F1, ROC-AUC, PR-AUC pre anomaly detection

## Cieľové premenné

Šesť premenných z trojfázovej sústavy:

- `u1_norm`, `u2_norm`, `u3_norm` – napätia jednotlivých fáz [V]
- `i1_norm`, `i2_norm`, `i3_norm` – prúdy jednotlivých fáz [A]

## Granularity

| Granularita | Krokov za deň | Použité modely |
|---|---|---|
| 10 min | 144 | VAR, LightGBM, XGBoost |
| 30 min | 48 | všetky |
| 1 h | 24 | všetky |
