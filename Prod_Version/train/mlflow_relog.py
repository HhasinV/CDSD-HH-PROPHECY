"""
Relog manuel d'un MLflow run à partir du pickle existant.
À usage one-shot pour rattraper un MLflow loupé.
"""
import pickle
import mlflow
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# ───────────────────────────────────────────────
# CHARGEMENT DU .env
# ───────────────────────────────────────────────
load_dotenv()

# Config MLflow (corrige l'URL ici)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT   = "experiment-smart-reassort-xgboost-forecast-30d-v6"
MLFLOW_RUN_NAME     = "smart-reassort-30d-airflow-relog"

# Charger le pickle
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Pickle chargé : {len(data['feature_cols'])} features")

# Métriques connues du dernier run (depuis tes logs Airflow)
metrics = {
    "mae_test": 2.485,
    "wmape_test": 47.5,
    "acc3_test": 76.3,
    "mae_train": 2.175,
    "wmape_train": 46.4,
    "cv_mae_mean": 2.615,
    "cv_mae_std": 0.371,
}

# Log MLflow
print(f"Connexion MLflow : {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
    mlflow.log_params(data["best_params"])
    mlflow.log_param("forecast_horizon", data["forecast_horizon"])
    mlflow.log_param("n_features", len(data["feature_cols"]))
    mlflow.log_param("features_list", ", ".join(data["feature_cols"]))
    
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    
    run_id = mlflow.active_run().info.run_id
    print(f"✓ MLflow run créé : {run_id}")

print("✓ Terminé")