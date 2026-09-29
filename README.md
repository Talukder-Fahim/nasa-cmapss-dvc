# NASA C-MAPSS Turbofan RUL Prediction Pipeline (MLOps)

An end-to-end reproducible MLOps pipeline for predicting Remaining Useful Life (RUL) of turbofan jet engines using the NASA C-MAPSS benchmark dataset.

## 🛠️ Architecture & Tech Stack
* **Version Control:** Git & GitHub
* **Data & Model Lineage:** DVC (Data Version Control) with local remote backup
* **Feature Store:** Feast (Parquet offline store & SQLite online store)
* **Modeling:** Scikit-Learn `RandomForestRegressor` with engine-level `GroupShuffleSplit`
* **Monitoring & Drift:** Evidently AI (Visual HTML drift report & performance degradation metrics)

## 📁 Project Structure
```text
CMaps/
├── .dvc/                   # DVC configuration & cache tracking
├── data/
│   ├── raw/                # Tracked by DVC (train_FD001.txt, train_FD002.txt, etc.)
│   └── processed/          # Preprocessed tables & test splits
├── feature_repo/           # Feast feature repository
│   ├── data/               # Engine features in Parquet format
│   ├── feature_store.yaml  # Feature store configuration
│   ├── feature_definitions.py # Entity & FeatureView schemas
│   └── test_online.py      # Online feature retrieval test
├── src/
│   ├── preprocess.py       # RUL computation, sensor filtering, scaling
│   ├── train.py            # GroupShuffleSplit training logic
│   ├── evaluate.py         # RMSE & MAE metric evaluation
│   ├── prepare_feast_data.py # Parquet export for Feast
│   └── drift_check.py      # Data drift analysis & FD002 evaluation
├── dvc.yaml                # DVC pipeline stage graph
├── params.yaml             # Hyperparameter & pipeline config
├── metrics.json            # Model evaluation metrics
└── drift_report.html       # Evidently AI visual drift report
