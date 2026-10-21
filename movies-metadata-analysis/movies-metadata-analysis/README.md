#  Movies Metadata — EDA, Clustering & Revenue Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/movies-metadata-analysis/blob/main/notebooks/movies_meta.ipynb)
![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans%20%7C%20Agglomerative-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-ff6f00)


An end-to-end data-science project on the **Movies Metadata** dataset (45,466 movies, 24 columns):

1. **Exploratory Data Analysis & feature engineering** — including *hidden* missing values (zeros that actually mean "not recorded").
2. **Clustering** movies by `budget`, `runtime` and `vote_average` with **K-Means** and **Hierarchical (Ward)** clustering.
3. **Neural network** that predicts `log(revenue)` from engineered features.

A full written report (Persian) is available in [`report/report_metamovies.pdf`](report/report_metamovies.pdf).

---

##  Repository structure

```
movies-metadata-analysis/
├── notebooks/
│   └── movies_meta.ipynb          # full analysis (EDA → clustering → NN)
├── data/
│   └── movies_metadata.csv        # dataset (add it yourself, see data/README.md)
├── outputs/
│   ├── df_cluster_final.csv       # clustered data (budget, vote_average, runtime, log_budget, runtime_winsor, cluster_label)
│   ├── df_cluster_final.pkl       # same, as pickle
│   └── kmeans_final_model.pkl     # trained KMeans (k=5)
├── report/
│   └── report_metamovies.pdf      # project report (Persian)
├── images/                        # figures used in this README
├── requirements.txt
└── README.md
```

---

## Getting started
### Option A — Google Colab
Click the **Open in Colab** badge above and run the notebook top-to-bottom.
The first cell installs the extra packages, clones this repo (edit `REPO_URL` in that cell) and finds the dataset in `data/`.

### Option B — Local

```bash
git clone https://github.com/hataminahid/movies-metadata-analysis.git
cd movies-metadata-analysis
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put movies_metadata.csv inside data/
jupyter notebook notebooks/movies_meta.ipynb
```

### Use the saved clustering model

```python
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("outputs/df_cluster_final.csv")
features = ["log_budget", "runtime_winsor", "vote_average"]

# the scaler was not saved, so re-fit it on the same data used for training
scaler = StandardScaler().fit(df[features])

kmeans = joblib.load("outputs/kmeans_final_model.pkl")
new_movie = pd.DataFrame([[np.log1p(30_000_000), 110, 7.2]], columns=features)
print(kmeans.predict(scaler.transform(new_movie)))   # -> cluster id (0-4)
```

---

##  Methodology

### 1. EDA & missing data
- 13 duplicate rows removed; 6 numeric columns analysed (`budget, revenue, runtime, vote_average, vote_count, popularity`) → 45,190 rows after dropping true `NaN`s.
- **Hidden missing values:** ~80 % of `budget` and ~84 % of `revenue` are `0`, which means *not recorded*, not "free movie". Zeros were therefore treated as missing (median imputation + missing-indicator flags where prediction is the goal).
- Skewed features (`budget`, `revenue`, `vote_count`, `popularity`) → `log1p`; `runtime` outliers → winsorization.
- Little's MCAR test (via `pyampute`) was run on the zero-as-missing columns.

![Missing values](images/01_missing_values.png)
![Correlation](images/02_correlation_matrix.png)

### 2. Clustering (`budget`, `runtime`, `vote_average`)
- Features: `log_budget`, `runtime_winsor`, `vote_average`, standardised with `StandardScaler`. `revenue` is deliberately **not** used (avoids leakage).
- k = 2…10 evaluated with the elbow method and silhouette score.

![KMeans elbow/silhouette](images/04_kmeans_elbow_silhouette.png)

**K-Means, k = 5** (silhouette 0.343; k = 6 scores 0.357 but is only marginally better, so the simpler model was kept):

| Cluster | Movies | log_budget | runtime (min) | vote_average | Interpretation |
|:-:|--:|--:|--:|--:|---|
| 0 | 24,527 | 15.87 | 96.9 | 6.59 | Typical mid-length, well-rated films |
| 1 | 595 | 7.57 | 86.7 | 5.90 | Very low budget |
| 2 | 2,717 | 15.83 | 34.2 | 6.58 | Short films |
| 3 | 13,140 | 15.85 | 93.1 | 4.63 | Poorly rated films |
| 4 | 4,211 | 16.29 | 149.6 | 6.66 | Long, high-budget, well-rated films |

![KMeans 3D](images/05_kmeans_3d_clusters.png)

**Hierarchical (Ward), k = 5** gave a comparable structure (silhouette ≈ 0.312) and made overlaps among mid-budget movies easier to see; K-Means was preferred for its simplicity and speed.

![Hierarchical 3D](images/07_hierarchical_3d_clusters.png)

### 3. Neural network — revenue prediction
- **Target:** `log1p(revenue)`.
- **Features (27):** `log_budget`, `log_vote_count`, `log_popularity`, `has_collection`, `budget_missing`, `is_US_production`, top-10 genres (one-hot), top-10 languages + "Other" (one-hot).
- **Split:** 70 / 15 / 15 (train / validation / test), `StandardScaler` fitted on train only.
- **Model:** `Dense(64, relu) → Dense(32, relu) → Dense(1)`, Adam, MSE loss, early stopping (patience 6).

| Metric (test set) | Value |
|---|---|
| MSE (log scale) | 13.89 |
| R² | 0.621 |

Strongest correlations with `log_revenue`: `log_vote_count` (0.64), `log_popularity` (0.58), `is_US_production` (0.27), `log_budget` (0.19), `has_collection` (0.18); `budget_missing` is strongly negative (−0.62).

---

##  Notes & limitations
- Movies with unrecorded revenue (`0`) are kept in the regression and become `log1p(0) = 0`. Because unrecorded budget and unrecorded revenue tend to co-occur, the `budget_missing` flag carries a lot of the predictive signal, so **R² = 0.62 should not be read as "how well revenue of released films is predicted"**. Restricting the model to films with a recorded revenue is a natural next step.
- Cluster labels from K-Means and Hierarchical clustering are independent (label `0` in one is unrelated to label `0` in the other).
- `kmeans_final_model.pkl` was saved with scikit-learn 1.7.2; other versions may emit a warning when loading.
- The `StandardScaler` used for clustering was not persisted (see the snippet above).

##  Tech stack
Python · pandas · NumPy · SciPy · scikit-learn · TensorFlow/Keras · matplotlib · seaborn · missingno · pyampute

##  License
Add a license of your choice (e.g. MIT) as `LICENSE`. The dataset belongs to its original authors — check its license on Kaggle.
