# SpaceX Launch Predictor & Cost Optimizer

A full end-to-end ML system that predicts the probability of a SpaceX Falcon 9 booster landing successfully, enriches launch data with real weather conditions, and exposes the results through an interactive Streamlit web application with a business ROI calculator.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Setup & Installation](#-setup--installation)
  - [Option A - Local Python Environment](#option-a-local-python-environment)
  - [Option B - Docker (Recommended)](#option-b-docker-recommended)
- [Step 1 - Run the ML Pipeline](#step-1-run-the-ml-pipeline)
- [Step 2 - Launch the Web App](#step-2-launch-the-web-app)
- [How to Use the App](#-how-to-use-the-app)
- [AWS S3 Configuration](#-aws-s3-configuration)
- [Streamlit Secrets](#-streamlit-secrets)
- [Model Details](#-model-details)
- [Statistical Analysis Performed](#-statistical-analysis-performed)
- [Fallback: Synthetic Data Mode](#-fallback-synthetic-data-mode)

---

## Project Overview

This project has two components that work together:

| Component | File | Role |
|---|---|---|
| **ML Pipeline** | `SpaceX.py` | Data ingestion → cleaning → statistical analysis → model training → upload to S3 |
| **Web App** | `app.py` | Interactive Streamlit UI that downloads the trained model from S3 and serves predictions |

The pipeline attempts to pull **live data** from the SpaceX public API (v4) and the **Open-Meteo weather archive** API. If the live API is unavailable, it automatically falls back to generating 200 synthetic launch records so the pipeline can always run end-to-end.

---

## Architecture

```
SpaceX API ──┐
             ├──► get_spacex_data()
Open-Meteo ──┘         │
                        ▼
               save_to_sqlite()
                        │
                        ▼
               master_spacex_weather.db  (SQLite)
                        │
              ┌─────────┴──────────┐
              ▼                          ▼
      cleanup_database()   statistical_Analysis()
              │
              ▼
      encoding_modeling()
       (XGBoost + GridSearchCV + SHAP)
              │
              ▼
        best_model.pkl ──► AWS S3 Bucket
                                  │
                                  ▼
                           app.py (Streamlit)
                        downloads model on startup
```

---

## Project Structure

```
SpaceX/
├── SpaceX.py            # Full ML pipeline (data → training → S3 upload)
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
├── docker-compose.yml   # Multi-container orchestration
```

After running the pipeline, two additional files will be created locally:

```
├── master_spacex_weather.db   # SQLite database with launch records
└── best_model.pkl             # Trained XGBoost model (also uploaded to S3)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost (via scikit-learn Pipeline) |
| Hyperparameter Tuning | GridSearchCV + TimeSeriesSplit |
| Explainability | SHAP (TreeExplainer) |
| Model Reporting | EMReport (PyPI) |
| Data Storage | SQLite |
| Model Storage | AWS S3 |
| Web Framework | Streamlit |
| Containerization | Docker + Docker Compose |
| Weather Data | Open-Meteo Archive API |
| Launch Data | SpaceX REST API v4 |

---

## Prerequisites

- Python **3.9+**
- Docker & Docker Compose (for Option B)
- An **AWS account** with an S3 bucket (to persist and serve the model)
- Internet access (for SpaceX API + Open-Meteo API calls)

---

## Setup & Installation

### Option A — Local Python Environment

```bash
# 1. Clone the repository
git clone https://github.com/alkiviadisss/SpaceX.git
cd SpaceX

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Option B — Docker (Recommended)

No Python setup required. Docker handles everything.

```bash
# 1. Clone the repository
git clone https://github.com/alkiviadisss/SpaceX.git
cd SpaceX

# 2. Create a .env file with your AWS credentials (see AWS section below)
cp .env.example .env   # or create manually

# 3. Build and start the app
docker-compose up --build
```

The app will be available at **http://localhost:8501**.

---

## Step 1 - Run the ML Pipeline

> This step trains the model and uploads it to S3. It only needs to be run **once**.

```bash
python SpaceX.py
```

The pipeline executes 4 sequential steps:

1. **Data Collection** - Calls the SpaceX API for the latest launch, enriches it with weather data from Open-Meteo, and stores everything in `master_spacex_weather.db`.
2. **Cleanup** - Fills any missing `payload_mass_kg` values using the average for the same booster version.
3. **Statistical Analysis** - Runs descriptive and inferential tests (t-tests, chi-square, ANOVA, Pearson correlation) and prints results to the console.
4. **Encoding & Modeling** - Trains an XGBoost classifier inside a scikit-learn Pipeline, tunes it with GridSearchCV, evaluates it, runs SHAP explainability, saves `best_model.pkl`, and prompts you to upload it to S3.

During the S3 upload step you will be asked interactively:

```
AWS Access Key ID:
AWS Secret Access Key:
S3 Bucket Name (e.g. spacex-ml-models):
Region (e.g. eu-central-1) [Default: eu-central-1]:
```

You can press Enter to use environment variables instead of typing credentials.

---

## Step 2 - Launch the Web App

### Local

```bash
streamlit run app.py
```

### Docker

```bash
docker-compose up
```

Open your browser at **http://localhost:8501**.

---

## How to Use the App

The app has a **sidebar** for inputs and three **tabs** for outputs.

### Sidebar - Mission Parameters

| Input | Type | Description |
|---|---|---|
| Launch Site | Dropdown | KSC LC 39A, CCSFS SLC 40, VAFB SLC 4E |
| Booster Version | Dropdown | B1058, B1060, B1062, B1067 |
| Orbit | Dropdown | LEO, ISS, GTO, VLEO |
| Payload Mass (kg) | Slider | 2,000 – 16,000 kg |
| Temperature (°C) | Slider | 0 – 40 °C |
| Wind Speed (km/h) | Slider | 0 – 50 km/h |
| Visibility (m) | Slider | 1,000 – 10,000 m |

### Tab 1 - Prediction

Shows the predicted **probability of a successful booster landing** as a percentage, alongside a success/failure badge.

### Tab 2 - Explainability (SHAP)

Displays a SHAP Waterfall plot showing exactly which features pushed the model's prediction up or down and by how much.

### Tab 3 - Business & ROI Calculator

Enter three financial parameters to compute the **Expected Financial Value (EV)** of the launch:

| Input | Default |
|---|---|
| Payload Value ($M) | $150M |
| Launch Cost ($M) | $67M |
| Failure Cost ($M) | $50M |

The calculator outputs the EV and a launch recommendation (proceed / abort).

---

## AWS S3 Configuration

The trained model is stored at:

```
s3://<your-bucket-name>/production/best_model.pkl
```

The app downloads it automatically on startup using the credentials you provide.

### Setting credentials via environment variables

```bash
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_BUCKET_NAME=your_bucket_name
export AWS_REGION=eu-central-1
```

### Setting credentials via `.env` file (for Docker Compose)

Create a `.env` file in the project root:

```env
AWS_ACCESS_KEY_ID=your_key_id
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_BUCKET_NAME=your_bucket_name
```

Docker Compose will inject these automatically into the container (see `docker-compose.yml`).

---

## Streamlit Secrets

For the deployed web app, credentials are read from Streamlit's secrets manager. Create the file `.streamlit/secrets.toml`:

```toml
AWS_ACCESS_KEY_ID = "your_key_id"
AWS_SECRET_ACCESS_KEY = "your_secret_key"
AWS_BUCKET_NAME = "your_bucket_name"
AWS_REGION = "eu-central-1"
```

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Framework | scikit-learn Pipeline |
| Preprocessing | OneHotEncoder (categorical) + StandardScaler (numerical) |
| Tuning | GridSearchCV (81 parameter combinations) |
| Validation | TimeSeriesSplit (3 folds) — prevents data leakage |
| Explainability | SHAP TreeExplainer |
| Output artifact | `best_model.pkl` (saved with joblib) |

**Features used:**

- `launch_site` (categorical)
- `booster_version` (categorical)
- `orbit` (categorical)
- `payload_mass_kg` (numerical)
- `temp_c` (numerical)
- `wind_speed_kmh` (numerical)
- `visibility_m` (numerical)

**Target variable:** `landing_outcome` (binary — contains "True" = success)

**Hyperparameter grid searched:**

```python
{
    'classifier__max_depth':      [2, 3, 4],
    'classifier__learning_rate':  [0.01, 0.05, 0.1],
    'classifier__n_estimators':   [50, 100, 150],
    'classifier__subsample':      [0.7, 0.8, 1.0]
}
```

---

## Statistical Analysis Performed

The pipeline runs the following tests automatically and prints results to the console:

| Test | Variables | Hypothesis |
|---|---|---|
| Welch's t-test | Wind speed vs landing outcome | Does wind speed affect landing success? |
| Chi-square | Launch site vs landing outcome | Does launch site affect landing success? |
| One-way ANOVA | Payload mass per orbit type | Does payload mass affect success per orbit? |
| Welch's t-test | Visibility vs landing outcome | Does visibility affect landing success? |
| Chi-square | Low visibility (<5000 m) vs landing outcome | Does low visibility specifically increase failure rate? |
| Welch's t-test | Payload mass vs landing outcome | Does payload mass affect landing success? |
| Pearson Correlation | Flight number vs cumulative success rate | Is SpaceX learning over time? |
| Pearson Correlation | Flight number vs cumulative payload mass | Has payload capacity grown over time? |
| Chi-square (GoF) | Launch month distribution | Are launches seasonal? |

---

## Fallback: Synthetic Data Mode

If the SpaceX API is unreachable, the pipeline automatically generates **200 synthetic launches** spanning from January 2018 onward, with realistic distributions for all features. Weather-based failure logic is embedded:

- Wind speed > 22 km/h → higher failure probability
- Visibility < 4,000 m → higher failure probability

This ensures the full pipeline (including training and the Streamlit app) can run completely offline or in environments without API access.

---

## Dependencies

```
streamlit==1.30.0
pandas==2.1.0
scikit-learn==1.3.0
xgboost==2.0.0
shap==0.44.0
boto3==1.34.0
joblib==1.3.2
matplotlib==3.8.2
scipy==1.11.4
requests==2.31.0
EMReport==0.1.3
```

Install via:

```bash
pip install -r requirements.txt
```
*For this project i used my own Evaluation Metric Library (EMReport), that standardizes supervised and unsupervised model diagnostics into a single, consistent API, that is also Verified and Published in PYPI Community.*

---

## Author

**Alkiviadis Agrogiannhs**  
Data Scientist & Machine Learning Engineer  
[LinkedIn](https://www.linkedin.com/in/alkiviadis-agrogiannhs/)
[Email](mailto:alkiviadisagrogiannhs@gmail.com)
[GitHub](https://github.com/alkiviadisss)

---
