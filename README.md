# Peru Disaster-Risk Classifiers — Distributed PySpark / Spark MLlib

A **distributed Spark MLlib** reimplementation of the household **flood** and **landslide** risk
classifiers from:

> Quiliche, R., Maçaira, P., Araujo Baião, F., & Leiras, A. (2025).
> *Identifying final demand points for aid in the aftermath of sudden-onset climate-related disasters
> in Peru: A supervised learning approach.*
> **International Journal of Disaster Risk Reduction, 127, 105593.** https://doi.org/10.1016/j.ijdrr.2025.105593

The original study used single-node `XGBoost` + Bayesian search (`scikit-optimize`) + `SHAP`.
This repository rebuilds the same pipeline to run **distributed on Databricks** (tested on **Free Edition /
serverless**) using the native Spark ML stack.

---

## Where the data comes from (ETL)

The CSVs in this repo are the **end** of an ETL pipeline described in the paper. The steps above the
dashed line were done in the original study; this repo starts from the train/test matrices.

```mermaid
flowchart TD
    A["ENAHO national household survey, INEI<br/>2018, 2019, 2020, 2021<br/>stratified random sample, urban and rural"]
    A1["Household coordinates"]
    A2["Self-report: house affected by a<br/>natural disaster in the last 12 months<br/>(disaster type NOT recorded)"]
    A3["112 vulnerability features<br/>economic, social, health, geographical"]

    B1["MINAM flood-prone urban areas<br/>(incl. tsunamis)"]
    B2["Dottori et al. 2016<br/>global river flood hazard map"]
    B3["GFDRR Global Landslide Hazard Map<br/>rainfall-triggered, 1980-2018"]
    B4["CENEPRED historical records<br/>(verification)"]
    FZ["Flood hazard zones"]
    LZ["Landslide hazard zones"]

    C{"Spatial join<br/>household point in hazard zone?"}
    D["flood_risk = 1<br/>affected AND inside a flood zone"]
    E["landslide_risk = 1<br/>affected AND inside a landslide zone"]
    N["0 otherwise"]
    F["Modeling table<br/>101,115 household-years<br/>112 features + 4 ID columns"]
    G["Train: 2018-2020<br/>72,952 rows"]
    H["Test: 2021 hold-out<br/>28,163 rows"]
    I["Spark MLlib pipeline in this repo<br/>assemble, scale, weight, GBT, tune, evaluate"]

    A --> A1
    A --> A2
    A --> A3
    B1 --> FZ
    B2 --> FZ
    B3 --> LZ
    B4 -.-> LZ
    A1 --> C
    FZ --> C
    LZ --> C
    A2 --> D
    A2 --> E
    C --> D
    C --> E
    C --> N
    D --> F
    E --> F
    N --> F
    A3 --> F
    F -->|temporal split| G
    F -->|temporal split| H
    G -.-> I
    H -.-> I
```

**Why the spatial join is needed:** ENAHO asks whether the house was affected by a natural disaster
in the last 12 months (drought, storm, plague, flood, etc.) but does not record *which* hazard.
Intersecting each household's coordinates with hazard zones attributes the event to floods or
landslides.

**Hazard-zone sources**
- **Floods:** Ministerio del Ambiente (MINAM) flood-prone urban areas, including tsunamis, joined with
  the global flood hazard map of Dottori et al. (2016) to cover rural river floods.
- **Landslides:** Global Landslide Hazard Map from the Global Facility for Disaster Reduction and
  Recovery (GFDRR) — mean annual rainfall-triggered landslide hazard, 1980-2018 — verified against
  historical data from CENEPRED.

**Why the classes are imbalanced:** a household is positive only if it both reports being affected and
sits inside a hazard zone. Few households meet both conditions (1.20% for floods and 5.45% for
landslides over the whole 2018-2021 sample), so positives are a small minority:

| Label | Train positives (2018-2020) | Test positives (2021) |
|---|---|---|
| `flood_risk` | 1,020 of 72,952 (1.4%) | 198 of 28,163 (0.7%) |
| `landslide_risk` | 3,931 of 72,952 (5.4%) | 1,581 of 28,163 (5.6%) |

This is why the pipeline uses cost-sensitive weights (`n_neg / n_pos`) and imbalance-aware metrics
(GIBA, G-Mean, MCC) instead of plain accuracy.

## What's here

```
peru-disaster-risk-spark/
├── databricks_distributed_pipeline.ipynb   # the full pipeline (import into Databricks)
├── Databases/TrainTestSplit/               # modeling matrices (temporal split: train < 2021, test = 2021)
│   ├── xf_train.csv  yf_train.csv  xf_test.csv  yf_test.csv   # floods   (112 features + 4 ID cols)
│   └── xl_train.csv  yl_train.csv  xl_test.csv  yl_test.csv   # landslides
├── CITATION.cff
└── requirements.txt
```

> The `x*` files hold 4 ID columns (`conglome, vivienda, hogar, year`) + 112 features.
> The `y*` files hold a single label column (`flood_risk` / `landslide_risk`) aligned **by row order**.

## How the distributed version maps to the original

| Original (pandas / sklearn) | This repo (Spark MLlib) |
|---|---|
| `pd.read_csv` | `spark.read.csv` |
| `RobustScaler` (sklearn) | `pyspark.ml.feature.RobustScaler` (`withCentering=True`) |
| feature matrix `X` | `VectorAssembler` → one `features` vector |
| `scale_pos_weight` | per-row `weightCol` (cost-sensitive learning) |
| `XGBClassifier` | `GBTClassifier` (distributed gradient boosting) |
| `BayesSearchCV` (skopt) | `TrainValidationSplit` + `ParamGridBuilder` (grid search) |
| GIBA / G-Mean / specificity | confusion matrix via Spark `groupBy` |
| SHAP | `.featureImportances` + Spark partial-dependence |
| `pickle.dump` | `model.write().save(...)` |

**Why the tuner changed:** PySpark has no Bayesian hyperparameter search — `pyspark.ml.tuning` provides
only grid/random search (`CrossValidator`, `TrainValidationSplit`). A single-node sklearn tuner cannot
drive a distributed Spark estimator, so the search had to move to the cluster-native one.

## Results from the Databricks run

The committed notebook includes the outputs of a full run on **Databricks Free Edition (serverless)**
for `LABEL = "flood_risk"`. Grid search over 4 combinations selected `maxDepth = 4`, `maxIter = 60`.
Metrics on the 2021 hold-out test set (28,163 households, 198 positives):

| Metric | Value |
|---|---|
| AUC | 0.8145 |
| Sensitivity (recall) | 0.6717 |
| Specificity | 0.8312 |
| Balanced accuracy | 0.7515 |
| G-Mean | 0.7472 |
| GIBA (α = 0.9) | 0.6400 |
| MCC | 0.1113 |
| Precision | 0.0274 |
| F1 | 0.0527 |

Confusion matrix: TP = 133, FN = 65, FP = 4,720, TN = 23,245.

**Reading these numbers:** the model is tuned to avoid false negatives (unmet demand), so it finds
about two thirds of the affected households at the cost of many false alarms. With only 0.7% positives
in the test year, precision is low even at 83% specificity. The top features by importance are
`altitude` (0.32) and per-capita expenditure `gpc` (0.12).

Landslides have not been run yet in the committed notebook — set `LABEL = "landslide_risk"` and re-run.

**A note on scale:** the data is about 50 MB, so Spark gives no speed advantage here over single-node
sklearn. The point of this repo is the port to a distributed framework and the platform constraints
it surfaces, not a scale-out speedup.

## Run it on Databricks

1. **Import the notebook** — Workspace → ⋮ → **Import → File** → `databricks_distributed_pipeline.ipynb`.
2. **Upload the data to a Volume** — Catalog → your catalog/schema → **Create → Volume** (e.g. `thesis_data`),
   then upload the 8 CSVs from `Databases/TrainTestSplit/` into
   `/Volumes/<catalog>/<schema>/thesis_data/Databases/TrainTestSplit/`.
   Alternatively, upload the CSVs to a Workspace folder and let the notebook's first code cell create
   the Volume and copy them in — edit `SRC` in that cell to your own folder.
3. **Set `CATALOG` / `SCHEMA` / `VOLUME`** in the notebook's cell 0 (defaults: `workspace` / `default` /
   `thesis_data`). Pick floods or landslides with `LABEL`.
4. **Run top to bottom** on Serverless.

### Serverless notes (baked into the notebook)
- **Scratch path:** MLlib needs `SPARKML_TEMP_DFS_PATH` set to a Volume folder — cell 1 does this.
- **Column names:** region dummies with spaces are sanitized (Delta forbids `` ,;{}()\n\t=``).
- **Model-cache limit:** Spark Connect caps the per-session model cache at 1 GB, so the notebook uses
  `TrainValidationSplit` (fewer models than k-fold CV). If you hit `ML_CACHE_SIZE_OVERFLOW`, run
  `dbutils.library.restartPython()` and re-run cells 1 → 3 → 4.

## Requirements

Spark MLlib is **built into the Databricks runtime** — nothing to install for the core pipeline.
`requirements.txt` lists only optional extras (e.g. `xgboost` for the `SparkXGBClassifier` variant,
which needs a classic cluster). See it for details.

## Citation

See `CITATION.cff`. Please cite the paper (DOI above) if you use this code or data.

## License

All rights reserved by the authors unless a license file is added. Contact the authors before reuse.
