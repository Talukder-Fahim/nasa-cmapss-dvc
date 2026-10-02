# 🚀 NASA C-MAPSS Turbofan Engine RUL Prediction & MLOps Pipeline

> **An end-to-end Machine Learning & MLOps pipeline for predicting the Remaining Useful Life (RUL) of turbofan engines using NASA C-MAPSS data.**

The project combines **Machine Learning, DVC-based reproducibility, centralized configuration, automated evaluation, and data drift monitoring** into a single pipeline.

---

## 🎯 01  Problem Statement

Predictive maintenance is essential for preventing unexpected failures in aircraft engines.

The primary objective of this project is to **predict the Remaining Useful Life (RUL)** of turbofan engines using historical sensor readings.

> 💡 **Goal:** Estimate how many operational cycles an engine has remaining before reaching failure.

Accurate RUL prediction can support:

*  Predictive maintenance planning
*  Engine health monitoring
*  Early failure detection
*  Data-driven maintenance decisions

---

## 📊 02  Dataset

This project uses the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset.

Two datasets are used for different purposes:

| Dataset   | Purpose                                     |
| --------- | ------------------------------------------- |
| **FD001** |  Baseline training & evaluation           |
| **FD002** |  Data drift & distribution-shift analysis |

### Why two datasets?

**FD001** provides the baseline environment for developing and evaluating the model.

**FD002** introduces different operational conditions, allowing the pipeline to investigate whether the underlying data distribution changes when the operating environment changes.

---

## 🤖 03  Machine Learning Model

###  Random Forest Regressor

The project uses a **Random Forest Regressor** for RUL prediction.

Random Forest is suitable for this task because it can model **non-linear relationships** between multiple engine sensor readings and the target RUL value.

###  Hyperparameter Management

Instead of hard-coding parameters inside Python scripts, project parameters are centrally managed through:

```text
params.yaml
```

Important configurable parameters include:

```text
n_estimators
max_depth
test_size
rul_cap
```

This makes experimentation and reproducibility easier.

---

# 🏗️ 04  Project Structure

```text
CMaps/
│
├── 📁 data/
│   ├── 📁 raw/
│   │   └── Raw C-MAPSS dataset files (.txt)
│   │
│   └── 📁 processed/
│       └── Processed CSV files & engineered features
│
├── 📁 src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── drift_check.py
│
├── ⚙️ dvc.yaml
├── ⚙️ params.yaml
├── 📦 requirements.txt
└── 📖 README.md
```

### 📌 Source Files

| File             | Responsibility                                         |
| ---------------- | ------------------------------------------------------ |
| `preprocess.py`  |  Data cleaning, RUL generation & feature engineering |
| `train.py`       |  Model training & artifact generation                |
| `evaluate.py`    |  Model evaluation & metric generation                |
| `drift_check.py` |  Data drift monitoring using Evidently AI            |
| `dvc.yaml`       |  DVC pipeline orchestration                          |
| `params.yaml`    |  Centralized project configuration                   |

---

# 🔄 05  MLOps Pipeline

The complete workflow is orchestrated using **Data Version Control (DVC)**.

```text
        📂 Raw Data
             │
             ▼
      🧹 PREPROCESS
             │
             ▼
      🤖 TRAIN MODEL
             │
             ▼
       📈 EVALUATE
             │
             ▼
       🔍 DRIFT CHECK
```

---

##  5.1  Preprocess

**Script:** `src/preprocess.py`

The preprocessing stage prepares the raw C-MAPSS data for machine learning.

### Main operations

*  Load raw sensor data
*  Clean and organize the dataset
*  Generate the RUL target
*  Apply RUL capping
*  Perform feature engineering
*  Generate processed datasets

Output:

```text
data/processed/
```

---

## 🤖 5.2  Train

**Script:** `src/train.py`

The training stage uses the processed dataset to train the **Random Forest Regressor**.

Model configuration is loaded from:

```text
params.yaml
```

This keeps model configuration separate from the actual training logic.

### Pipeline

```text
Processed Data
      ↓
Parameter Configuration
      ↓
Random Forest Regressor
      ↓
Trained Model
```

---

## 📈 5.3  Evaluate

**Script:** `src/evaluate.py`

The trained model is evaluated using the test dataset.

### Evaluation Metrics

| Metric   | Purpose                                        |
| -------- | ---------------------------------------------- |
| **RMSE** | Measures the magnitude of prediction errors    |
| **MAE**  | Measures the average absolute prediction error |

The generated metrics are stored in:

```text
metrics.json
```

---

## 🔍 5.4  Drift Check

**Script:** `src/drift_check.py`

The drift monitoring stage uses **Evidently AI** to analyze changes in the data distribution.

The comparison is performed between:

```text
FD001 → Baseline
FD002 → New Operational Condition
```

The resulting report is generated as:

```text
drift_report.html
```

This provides an interactive view of feature-level distribution changes.

---

# ⚙️ 06  Centralized Configuration

All important project parameters are maintained in:

```text
params.yaml
```

Example:

```yaml
train:
  n_estimators: ...
  max_depth: ...

preprocess:
  rul_cap: ...

split:
  test_size: ...
```

### ✨ Benefits

* 🔁 Reproducible experiments
* ⚙️ Easy configuration changes
* 🧪 Easier experimentation
* 🧹 Cleaner source code
* 📦 Better project maintainability

---

# 🔄 07  DVC Pipeline

The pipeline is defined in:

```text
dvc.yaml
```

DVC manages:

* Pipeline stages
* Dependencies
* Parameters
* Outputs
* Reproducibility

### Pipeline Flow

```text
┌───────────────┐
│  PREPROCESS   │
└───────┬───────┘
        ↓
┌───────────────┐
│     TRAIN     │
└───────┬───────┘
        ↓
┌───────────────┐
│   EVALUATE    │
└───────┬───────┘
        ↓
┌───────────────┐
│ DRIFT CHECK   │
└───────────────┘
```

---

# 💻 08  Installation & Setup

## ① Clone the Repository

```bash
git clone https://github.com/Talukder-Fahim/nasa-cmapss-dvc.git
cd CMaps
```

## ② Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ 09  Run the Pipeline

Execute the complete pipeline using:

```bash
dvc repro
```

DVC automatically executes the required stages based on the defined dependencies and parameters.

### End-to-End Execution

```text
 Raw C-MAPSS Data
        ↓
 Preprocessing
        ↓
 Model Training
        ↓
 Evaluation
        ↓
 Drift Analysis
        ↓
 Reports & Metrics
```

---

# 🧪 10  View DVC Experiments

To inspect and compare DVC experiments:

```bash
dvc exp show --no-pager
```

This allows different parameter configurations and their resulting metrics to be compared.

---

# 📦 11  Pipeline Outputs

###  Processed Data

```text
data/processed/
```

Contains cleaned datasets and engineered features.

###  Model Artifacts

The training stage generates the trained Random Forest model and related artifacts.

###  Evaluation Metrics

```text
metrics.json
```

Contains:

```text
RMSE
MAE
```

###  Drift Report

```text
drift_report.html
```

Provides an interactive analysis of distribution changes between the baseline and new operational dataset.

---

# 📈 12  Results

## 🤖 Model Performance

The Random Forest model is evaluated on the test split using:

* **RMSE**
* **MAE**

The resulting metrics are automatically recorded in:

```text
metrics.json
```

This makes the evaluation process reproducible and suitable for experiment tracking.

---

## 🔍 Data Drift Analysis

The pipeline compares:

```text
FD001  ───────────────►  FD002
Baseline                 Distribution Shift
```

The generated `drift_report.html` provides an interactive analysis of how feature distributions change between the two operational conditions.

---

# 🧰 13  Technologies Used

| Technology          | Purpose                          |
| ------------------- | -------------------------------- |
|  **Python**       | Core development                 |
|  **Pandas**       | Data processing                  |
|  **NumPy**        | Numerical operations             |
|  **Scikit-learn** | Machine Learning                 |
|  **DVC**          | Pipeline & experiment management |
|  **Evidently AI** | Data drift monitoring            |
|  **PyYAML**       | Configuration management         |
|  **NASA C-MAPSS** | Turbofan engine dataset          |

---

# 🧠 14  MLOps Concepts Demonstrated

This project demonstrates a practical machine learning lifecycle:

```text
                 ┌─────────────────────┐
                 │    NASA C-MAPSS     │
                 │       Dataset       │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │    PREPROCESSING    │
                 │ Cleaning + Features │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │      TRAINING       │
                 │   Random Forest     │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │     EVALUATION      │
                 │     RMSE + MAE      │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │   DRIFT MONITORING  │
                 │     Evidently AI    │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │   REPORT & METRICS  │
                 └─────────────────────┘
```

### Key MLOps Components

| MLOps Concept                 | Implementation    |
| ----------------------------- | ----------------- |
|  Data & Pipeline Management | DVC               |
|  Configuration Management   | `params.yaml`     |
|  Reproducibility            | `dvc.yaml`        |
|  Data Processing            | Python            |
|  Feature Engineering        | Python            |
|  Model Training             | Random Forest     |
|  Model Evaluation           | RMSE & MAE        |
|  Experiment Tracking        | DVC Experiments   |
|  Drift Monitoring           | Evidently AI      |
|  Reporting                  | HTML Drift Report |

---

# 🚀 15  End-to-End Workflow

```text
                    ✈️ NASA C-MAPSS
                         │
                         ▼
                   📂 Raw Data
                         │
                         ▼
                ┌─────────────────┐
                │    PREPROCESS │
                │                 │
                │ • Cleaning      │
                │ • RUL Creation  │
                │ • RUL Capping   │
                │ • Feature Eng.  │
                └────────┬────────┘
                         │
                         ▼
                   Processed Data
                         │
                         ▼
                ┌─────────────────┐
                │    🤖 TRAIN     │
                │                 │
                │ Random Forest   │
                └────────┬────────┘
                         │
                         ▼
                    Trained Model
                         │
                         ▼
                ┌─────────────────┐
                │   📈 EVALUATE   │
                │                 │
                │ • RMSE          │
                │ • MAE           │
                └────────┬────────┘
                         │
                         ▼
                     metrics.json


              🔵 FD001        🟠 FD002
              Baseline    Distribution Shift
                  │              │
                  └──────┬───────┘
                         ▼
                ┌─────────────────┐
                │  🔍 DRIFT CHECK │
                │                 │
                │  Evidently AI   │
                └────────┬────────┘
                         │
                         ▼
                📊 drift_report.html
```

---

# 🏁 16  Conclusion

This project implements an end-to-end **RUL Prediction & MLOps Pipeline** using the NASA C-MAPSS turbofan engine dataset.

It combines:

>  **Machine Learning**
>  **Pipeline Reproducibility**
>  **Centralized Configuration**
>  **Automated Evaluation**
>  **Data Drift Monitoring**

The project goes beyond simply training a regression model by incorporating practical MLOps components such as **DVC pipeline orchestration, experiment tracking, centralized parameter management, and Evidently-based drift monitoring**.

Using **FD001 as the baseline** and **FD002 for distribution-shift analysis** provides a structured environment for studying how changes in operational data can affect an ML pipeline.

---

## ⭐ Project Highlights

```text
 RUL Prediction
 Random Forest Regression
 DVC Pipeline
 Parameterized Experiments
 RMSE & MAE Evaluation
 Evidently Data Drift Monitoring
 Interactive HTML Reporting
 Reproducible ML Workflow
```

> **From raw sensor data → preprocessing → model training → evaluation → drift monitoring — the entire ML workflow is automated through a reproducible pipeline.**
