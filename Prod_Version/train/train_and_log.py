"""
Smart Reassort — Train + MLflow log + Pickle
==============================================
Optuna tuning + train final + métriques + CV + log MLflow + pickle
Source : final_main.ipynb (cellules 4 fin, 5, 6, 7, 8, 14, 15)

Entrée  : newcode/exports/sales_train_ready.csv  (35 colonnes, ~20K lignes)
Sortie  : newcode/train/model.pkl
          + MLflow run sur le tracking server

Usage standalone :
    python newcode/train/train_and_log.py

Skip MLflow (utile en dev) :
    SKIP_MLFLOW=true python newcode/train/train_and_log.py

Usage Airflow :
    from newcode.train.train_and_log import run_train
    run_train(input_path="...", model_path="...")
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from dotenv import load_dotenv

# ───────────────────────────────────────────────
# CHARGEMENT DU .env
# ───────────────────────────────────────────────
load_dotenv()

# ───────────────────────────────────────────────
# LOGGING
# ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ───────────────────────────────────────────────
# CONSTANTES MÉTIER
# ───────────────────────────────────────────────
FORECAST_HORIZON = 30
TEST_DAYS = 60
OPTUNA_TRIALS = 120

FEATURES_TO_REMOVE = ["total_sales_7d", "total_sales_30d", "rolling_mean_90", "quarter"]

# Chemins par défaut
DEFAULT_INPUT      = "/home/henintsoa/Documents/ProjetFInalJedha/newcode/exports/sales_train_ready.csv"
DEFAULT_MODEL_PATH = "/home/henintsoa/Documents/ProjetFInalJedha/newcode/train/model.pkl"

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT   = "experiment-smart-reassort-xgboost-forecast-30d-v6"
MLFLOW_RUN_NAME     = "smart-reassort-30d-airflow"

# Skip MLflow (utile en dev / si HF est down)
SKIP_MLFLOW = os.getenv("SKIP_MLFLOW", "false").lower() in ("true", "1", "yes")


# ═══════════════════════════════════════════════
# SPLIT + FEATURES
# ═══════════════════════════════════════════════

def prepare_split(df_agg: pd.DataFrame) -> dict:
    """
    Split train/test temporel + sélection des features.
    """
    cutoff_date = df_agg["sale_date"].max() - pd.Timedelta(days=TEST_DAYS)
    
    feature_cols = [
        c for c in df_agg.columns
        if c not in ["sale_date", "target_30d", "target_log", "product_id"]
    ]
    feature_cols = [c for c in feature_cols if c not in FEATURES_TO_REMOVE]
    
    train_agg = df_agg[df_agg["sale_date"] <= cutoff_date]
    test_agg  = df_agg[df_agg["sale_date"] >  cutoff_date]
    
    X_train      = train_agg[feature_cols]
    y_train_log  = train_agg["target_log"]
    y_train_real = train_agg["target_30d"]
    X_test       = test_agg[feature_cols]
    y_test       = test_agg["target_30d"]
    
    train_weights = 1 + np.log1p(y_train_real)
    
    # Pour Optuna CV (sur l'ensemble du df_agg trié)
    df_agg_sorted = df_agg.sort_values("sale_date").reset_index(drop=True)
    X_all         = df_agg_sorted[feature_cols]
    y_all_log     = np.log1p(df_agg_sorted["target_30d"])
    y_all_real    = df_agg_sorted["target_30d"]
    
    # TimeSeriesSplit : 4 folds avec test_size = 20%
    test_size = int(len(X_all) * 0.20)
    tscv = TimeSeriesSplit(n_splits=4, test_size=test_size)
    
    logger.info(
        f"  Train : {len(train_agg):,} | Test : {len(test_agg):,} | "
        f"Features : {len(feature_cols)} (retiré {FEATURES_TO_REMOVE})"
    )
    
    return {
        "X_train": X_train, "y_train_log": y_train_log, "y_train_real": y_train_real,
        "X_test": X_test, "y_test": y_test, "train_weights": train_weights,
        "test_agg": test_agg, "feature_cols": feature_cols,
        "df_agg_sorted": df_agg_sorted, "X_all": X_all,
        "y_all_log": y_all_log, "y_all_real": y_all_real, "tscv": tscv,
    }


# ═══════════════════════════════════════════════
# OPTUNA
# ═══════════════════════════════════════════════

def run_optuna(splits: dict, n_trials: int = OPTUNA_TRIALS) -> dict:
    """Tuning hyperparamètres avec Optuna + CV TimeSeriesSplit."""
    X_all      = splits["X_all"]
    y_all_log  = splits["y_all_log"]
    y_all_real = splits["y_all_real"]
    tscv       = splits["tscv"]
    
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 1200),
            "max_depth":        trial.suggest_int("max_depth", 4, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0, 5.0),
            "random_state":     42, "n_jobs": -1,
        }
        
        mae_scores = []
        for train_idx, test_idx in tscv.split(X_all):
            X_tr, X_te    = X_all.iloc[train_idx], X_all.iloc[test_idx]
            y_tr          = y_all_log.iloc[train_idx]
            y_te_real     = y_all_real.iloc[test_idx]
            w             = 1 + np.log1p(y_all_real.iloc[train_idx])
            
            m = xgb.XGBRegressor(**params)
            m.fit(X_tr, y_tr, sample_weight=w, verbose=False)
            preds_log = m.predict(X_te)
            preds = np.maximum(0, np.expm1(preds_log))
            mae_scores.append(mean_absolute_error(y_te_real, preds))
        
        return np.mean(mae_scores)
    
    logger.info(f"  Tuning Optuna ({n_trials} essais)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    logger.info(f"  ✓ Best CV MAE : {study.best_value:.3f}")
    
    return best_params


# ═══════════════════════════════════════════════
# TRAIN FINAL + MÉTRIQUES
# ═══════════════════════════════════════════════

def train_final_model(splits: dict, best_params: dict) -> xgb.XGBRegressor:
    """Entraîne le modèle final sur le train set avec les best params."""
    logger.info("  Entraînement final...")
    model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(
        splits["X_train"],
        splits["y_train_log"],
        sample_weight=splits["train_weights"],
        verbose=False,
    )
    return model


def compute_test_metrics(model: xgb.XGBRegressor, splits: dict) -> dict:
    """MAE, WMAPE, Acc±3 sur train et test."""
    y_pred       = np.maximum(0, np.expm1(model.predict(splits["X_test"])))
    y_pred_train = np.maximum(0, np.expm1(model.predict(splits["X_train"])))
    
    y_test       = splits["y_test"]
    y_train_real = splits["y_train_real"]
    
    mae        = mean_absolute_error(y_test, y_pred)
    mae_train  = mean_absolute_error(y_train_real, y_pred_train)
    wmape      = np.sum(np.abs(y_test.values - y_pred)) / np.sum(y_test.values) * 100
    wmape_train = np.sum(np.abs(y_train_real.values - y_pred_train)) / np.sum(y_train_real.values) * 100
    errors     = np.abs(y_test.values - y_pred)
    acc_3      = (errors <= 3).mean() * 100
    
    logger.info(
        f"  Test : MAE={mae:.3f} | WMAPE={wmape:.1f}% | Acc±3={acc_3:.1f}% | "
        f"Train MAE={mae_train:.3f}"
    )
    
    return {
        "mae": mae, "mae_train": mae_train,
        "wmape": wmape, "wmape_train": wmape_train,
        "acc_3": acc_3,
    }


def run_cross_validation(splits: dict, best_params: dict) -> pd.DataFrame:
    """Cross-validation finale 4 folds pour métriques de robustesse."""
    X_all      = splits["X_all"]
    y_all_log  = splits["y_all_log"]
    y_all_real = splits["y_all_real"]
    tscv       = splits["tscv"]
    
    logger.info("  Cross-validation finale (4 folds)...")
    cv_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        X_tr, X_te = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_tr       = y_all_log.iloc[train_idx]
        y_te_real  = y_all_real.iloc[test_idx]
        w          = 1 + np.log1p(y_all_real.iloc[train_idx])
        
        fold_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        fold_model.fit(X_tr, y_tr, sample_weight=w, verbose=False)
        preds = np.maximum(0, np.expm1(fold_model.predict(X_te)))
        
        fold_mae   = mean_absolute_error(y_te_real, preds)
        fold_wmape = np.sum(np.abs(y_te_real.values - preds)) / np.sum(y_te_real.values) * 100
        fold_acc3  = (np.abs(y_te_real.values - preds) <= 3).mean() * 100
        
        cv_results.append({"fold": fold+1, "mae": fold_mae, "wmape": fold_wmape, "acc_3": fold_acc3})
        logger.info(f"    Fold {fold+1} : MAE={fold_mae:.3f}, WMAPE={fold_wmape:.1f}%, Acc±3={fold_acc3:.1f}%")
    
    cv_df = pd.DataFrame(cv_results)
    logger.info(
        f"  → CV MAE : {cv_df['mae'].mean():.3f} ± {cv_df['mae'].std():.3f}"
    )
    return cv_df


# ═══════════════════════════════════════════════
# MLFLOW LOG
# ═══════════════════════════════════════════════

def log_to_mlflow(
    best_params: dict,
    metrics: dict,
    cv_df: pd.DataFrame,
    feature_cols: list,
    n_products: int,
) -> str:
    """Log les params + métriques vers MLflow. Retourne le run_id."""
    
    # Skip si demandé
    if SKIP_MLFLOW:
        logger.warning("  ⚠ MLflow skip (SKIP_MLFLOW=true)")
        return "skipped"
    
    logger.info(f"  Connexion MLflow : {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        # Params
        mlflow.log_params(best_params)
        mlflow.log_param("forecast_horizon", FORECAST_HORIZON)
        mlflow.log_param("test_days", TEST_DAYS)
        mlflow.log_param("optuna_trials", OPTUNA_TRIALS)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("features_removed", ", ".join(FEATURES_TO_REMOVE))
        mlflow.log_param("n_products", n_products)
        mlflow.log_param("sample_weight", "1+log1p")
        mlflow.log_param("target_transform", "log1p")
        mlflow.log_param("cv_method", "TimeSeriesSplit_4fold_20pct")
        mlflow.log_param("features_list", ", ".join(feature_cols))
        
        # Métriques test
        mlflow.log_metric("mae_test", metrics["mae"])
        mlflow.log_metric("wmape_test", metrics["wmape"])
        mlflow.log_metric("acc3_test", metrics["acc_3"])
        mlflow.log_metric("mae_train", metrics["mae_train"])
        mlflow.log_metric("wmape_train", metrics["wmape_train"])
        
        # Métriques CV
        mlflow.log_metric("cv_mae_mean", cv_df["mae"].mean())
        mlflow.log_metric("cv_mae_std", cv_df["mae"].std())
        mlflow.log_metric("cv_wmape_mean", cv_df["wmape"].mean())
        mlflow.log_metric("cv_wmape_std", cv_df["wmape"].std())
        mlflow.log_metric("cv_acc3_mean", cv_df["acc_3"].mean())
        mlflow.log_metric("cv_acc3_std", cv_df["acc_3"].std())
        
        run_id = mlflow.active_run().info.run_id
    
    logger.info(f"  ✓ MLflow run : {run_id}")
    return run_id


# ═══════════════════════════════════════════════
# PICKLE
# ═══════════════════════════════════════════════

def save_pickle(
    model: xgb.XGBRegressor,
    feature_cols: list,
    best_params: dict,
    output_path: str,
) -> str:
    """Sauvegarde le modèle + métadonnées en pickle."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "best_params": best_params,
            "forecast_horizon": FORECAST_HORIZON,
        }, f)
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    logger.info(f"  ✓ Pickle : {output_path} ({size_mb:.2f} Mo)")
    return output_path


# ═══════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════

def run_train(input_path: str, model_path: str) -> dict:
    """
    Pipeline complet : split → Optuna → train → metrics → CV → MLflow → pickle.
    """
    # ── 1. Chargement ──
    logger.info(f"Lecture de {input_path}...")
    df_agg = pd.read_csv(input_path, encoding="utf-8-sig")
    df_agg["sale_date"] = pd.to_datetime(df_agg["sale_date"])
    logger.info(f"  → {len(df_agg):,} lignes, {len(df_agg.columns)} colonnes")
    
    n_products = df_agg["product_id"].nunique()
    
    # ── 2. Split + features ──
    splits = prepare_split(df_agg)
    
    # ── 3. Optuna ──
    best_params = run_optuna(splits, n_trials=OPTUNA_TRIALS)
    
    # ── 4. Train final ──
    model = train_final_model(splits, best_params)
    
    # ── 5. Métriques test ──
    metrics = compute_test_metrics(model, splits)
    
    # ── 6. CV finale ──
    cv_df = run_cross_validation(splits, best_params)
    
    # ── 7. MLflow ──
    run_id = log_to_mlflow(
        best_params=best_params,
        metrics=metrics,
        cv_df=cv_df,
        feature_cols=splits["feature_cols"],
        n_products=n_products,
    )
    
    # ── 8. Pickle ──
    save_pickle(model, splits["feature_cols"], best_params, model_path)
    
    logger.info("✓ Pipeline train terminé")
    return {
        "run_id": run_id,
        "model_path": model_path,
        "metrics": metrics,
        "cv_mae_mean": cv_df["mae"].mean(),
    }


# ═══════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    run_train(
        input_path=os.getenv("INPUT_PATH", DEFAULT_INPUT),
        model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
    )