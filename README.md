# System Threat Forecaster V2 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Model-orange)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-purple)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-yellow)

**System Threat Forecaster V2** is a production grade, end-to-end Machine Learning pipeline designed to predict, the probability and risk of Windows system malware and security vulnerabilities from system data.

Test it out live here: [**LIVE DEMO**](https://system-threat-forecaster-v2.onrender.com/)

---

## 🚀 Key Features

* **Feature Engineering (`data_processor.py`)**: Automatically parses complex version strings into component parts and patch deltas, extracts date components, calculates hardware performance ratios (RAM per core, storage ratios), computes security configuration vulnerability indexes, and applies frequency encodings.
* **Automated Hyperparameter Tuning (`trainer.py`)**: Integrates **Optuna** for efficient cross-validation hyperparameter optimization, persisting the best parameters automatically.
* **Robust Cross-Validation & Interpretability**: Employs Stratified K-Fold training using **LightGBM** coupled with **SHAP** tree explainers and out-of-fold confusion matrix evaluations.
* **Validated Model Performance**: Achieves robust cross-validation benchmark results with an **Overall OOF ROC-AUC : 0.68015** and an **Overall Accuracy : 0.63018**.
* **Modular Pipeline Architecture**: Clean separation of concerns with dedicated ingestion, validation, processing, training, and inference pipelines.
* **Industry Ready User Interface (`index.html`, `results.html`, `analytics.html`)**: Features glassmorphic dark-mode dashboards, interactive dropzones, animated probability distribution histograms with threshold lines, full-dataset search capabilities, and a **"Stats for Geeks"** model telemetry view.

---

## 🗂️ Project Structure

```text
System_Threat_Forecaster_V2/
│
├── app/
│   ├── main.py               # FastAPI web server and routing
│   └── templates/
│       ├── index.html        # Main SOC command center upload portal
│       ├── analytics.html    # Model related statistics.
│       └── results.html      # Interactive threat analysis & search dashboard
├── artifacts/
│   ├── metrics/              # Evaluation plots (SHAP, confusion matrix) & best params
│   ├── models/               # Serialized model binaries (.pkl)
│   └── predictions/          # Generated inference CSV outputs
│
├── data/                     # Training and testing datasets
├── engine/                   # Core pipeline logic & utilities
│   ├── data_processor.py     # Feature engineering & memory optimization
│   ├── exception.py          # Custom error traceback handler
│   ├── logger.py             # Centralized timestamped logging
│   ├── trainer.py            # Optuna tuning, LightGBM CV & SHAP
│   ├── utils.py              # YAML reader & joblib serialization
│   └── validator.py          # Schema and target integrity checks
│
├── pipeline/                 # End-to-end pipeline orchestration
│   ├── predict_pipeline.py   # Production inference pipeline
│   └── train_pipeline.py     # Full training execution pipeline
│
├── config.yaml               # Centralized configuration mapping
├── train.py                  # Training entry point script
└── requirements.txt          # Project dependencies
```

---

## ⚙️ Installation, Setup & Usage

### Step 1: Clone the Repository
```bash
git clone https://github.com/adityambati/System_Threat_Forecaster_V2.git
cd System_Threat_Forecaster_V2
```
#### Step 2: Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Model Training Pipeline
To execute data ingestion, schema validation, feature engineering, Optuna tuning, and K-Fold cross-validation:
```bash
python train.py
```
(This trains the model, saves artifacts into artifacts/, and generates submission.csv).

### Step 5: Launch the Web Dashboard
Start the local FastAPI application:
```bash
uvicorn main:app --reload
```
Open your browser and navigate to http://127.0.0.1:8000 to access the dashboard.

---

## Model Performance & Intelligence
* **Algorithm**: LightGBM Classifier (gbdt)
* **Validation Strategy**: Stratified K-Fold Cross-Validation
* **Decision Alert Threshold**: >= 0.35 Threat Probability
* **Explainability**: SHAP global feature impact analysis integrated directly into the Stats for Geeks dashboard panel.

---

## 🙋‍♂️ Author

**Aditya Ambati**  
🎓 B.S in Data Science @ IIT Madras  
🔗 [GitHub Profile](https://github.com/adityambati)
