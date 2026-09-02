"""
Smart Reassort — Preprocessing final (étape 4)
================================================
Features avancées + agrégation 30j + improvements + log-transform
Source : final_main.ipynb (cellules 1-4)

Entrée  : newcode/exports/sales_step4_b2c.csv  (41 colonnes, ~187K lignes)
Sortie  : newcode/exports/sales_train_ready.csv  (dataset agrégé prêt pour le train)

⚠ Cellule 2 (days_since_last_sale) est lente : 3-5 min sur 187K lignes
   (boucle Python imbriquée, fidèle au notebook).

Usage standalone :
    python newcode/train/preprocess_final.py

Usage Airflow :
    from newcode.train.preprocess_final import run_preprocess_final
    run_preprocess_final(input_path="...", output_path="...")
"""

import os
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────
# CONSTANTES MÉTIER
# ───────────────────────────────────────────────
FORECAST_HORIZON = 30   # On prédit la demande sur 30 jours
TEST_DAYS = 60          # Période de test (60 derniers jours)

DEFAULT_INPUT  = "/home/henintsoa/Documents/ProjetFInalJedha/newcode/exports/sales_step4_b2c.csv"
DEFAULT_OUTPUT = "/home/henintsoa/Documents/ProjetFInalJedha/newcode/exports/sales_train_ready.csv"


# ═══════════════════════════════════════════════
# FONCTIONS DE TRANSFORMATION
# ═══════════════════════════════════════════════

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features avancées du final_main.ipynb (cellule 2) :
      - days_since_last_sale : jours depuis la dernière vente
      - sale_frequency_30d   : fréquence de vente sur 30j
      - max_sale_30d         : pic de vente sur 30j
      - total_sales_7d / 30d : somme glissante
      - avg_price            : prix moyen par produit
      - product_age_days     : ancienneté depuis première vente
    """
    df = df.copy()
    products = df["product_id"].unique()
    
    # ── days_since_last_sale (boucle lente, fidèle au notebook) ──
    logger.info(f"  Calcul days_since_last_sale sur {len(products)} produits (lent)...")
    df["days_since_last_sale"] = 0
    
    for i, pid in enumerate(products):
        if (i + 1) % 200 == 0:
            logger.info(f"    {i+1}/{len(products)} produits...")
        
        mask = df["product_id"] == pid
        product_df = df.loc[mask]
        sale_dates = product_df.loc[product_df["qty_sold"] > 0, "sale_date"]
        
        if len(sale_dates) == 0:
            df.loc[mask, "days_since_last_sale"] = 999
            continue
        
        for idx in product_df.index:
            current_date = df.loc[idx, "sale_date"]
            past_sales = sale_dates[sale_dates < current_date]
            if len(past_sales) > 0:
                df.loc[idx, "days_since_last_sale"] = (current_date - past_sales.max()).days
            else:
                df.loc[idx, "days_since_last_sale"] = 999
    
    # ── Features rolling supplémentaires ──
    logger.info("  Features rolling supplémentaires (frequency, max, totals)...")
    df["sale_frequency_30d"] = df.groupby("product_id")["qty_sold"].transform(
        lambda x: (x.shift(1) > 0).rolling(window=30, min_periods=1).sum()
    )
    df["max_sale_30d"] = df.groupby("product_id")["qty_sold"].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).max()
    )
    df["total_sales_7d"] = df.groupby("product_id")["qty_sold"].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).sum()
    )
    df["total_sales_30d"] = df.groupby("product_id")["qty_sold"].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).sum()
    )
    
    # ── Prix moyen et âge produit ──
    logger.info("  Prix moyen et âge produit...")
    avg_price = df[df["prix_unitaire"] > 0].groupby("product_id")["prix_unitaire"].mean()
    df["avg_price"] = df["product_id"].map(avg_price).fillna(0)
    
    first_sale = df[df["qty_sold"] > 0].groupby("product_id")["sale_date"].min()
    df["first_sale_date"] = df["product_id"].map(first_sale)
    df["product_age_days"] = (df["sale_date"] - df["first_sale_date"]).dt.days
    df["product_age_days"] = df["product_age_days"].fillna(0).clip(lower=0)
    df = df.drop(columns=["first_sale_date"])
    
    # Nettoyage inf/NaN
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df


def aggregate_forecast_windows(
    df: pd.DataFrame,
    forecast_horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Crée des fenêtres glissantes (tous les 7 jours) avec target = somme des 30 jours suivants.
    Cellule 3 du notebook.
    
    Returns:
        df_agg : DataFrame agrégé (1 ligne = 1 (produit, jour anchor))
    """
    products = df["product_id"].unique()
    windows_data = []
    
    logger.info(f"  Agrégation {forecast_horizon}j sur {len(products)} produits...")
    
    for i, pid in enumerate(products):
        if (i + 1) % 200 == 0:
            logger.info(f"    {i+1}/{len(products)} produits...")
        
        product_df = df[df["product_id"] == pid]
        if len(product_df) < forecast_horizon + 7:
            continue
        
        # Tous les 7 jours, on crée une fenêtre
        for j in range(0, len(product_df) - forecast_horizon, 7):
            cr = product_df.iloc[j]
            cd = cr["sale_date"]
            
            # Target : somme des qty_sold sur les 30 jours suivants
            fm = (product_df["sale_date"] > cd) & \
                 (product_df["sale_date"] <= cd + pd.Timedelta(days=forecast_horizon))
            fs = product_df.loc[fm, "qty_sold"].sum()
            
            windows_data.append({
                "product_id": pid,
                "category_id": cr["category_id"],
                "sale_date": cd,
                "month": cr["month"],
                "quarter": cr["quarter"],
                "day_of_week": cr["day_of_week"],
                "is_weekend": cr["is_weekend"],
                "is_christmas_period": cr["is_christmas_period"],
                "is_summer": cr["is_summer"],
                "is_back_to_school": cr["is_back_to_school"],
                "is_black_friday_period": cr["is_black_friday_period"],
                "rolling_mean_7": cr["rolling_mean_7"],
                "rolling_mean_14": cr["rolling_mean_14"],
                "rolling_mean_30": cr["rolling_mean_30"],
                "rolling_mean_60": cr["rolling_mean_60"],
                "rolling_mean_90": cr["rolling_mean_90"],
                "rolling_std_7": cr["rolling_std_7"],
                "rolling_std_30": cr["rolling_std_30"],
                "trend_7_30": cr["trend_7_30"],
                "trend_30_90": cr["trend_30_90"],
                "lag_7": cr["lag_7"],
                "lag_14": cr["lag_14"],
                "lag_28": cr["lag_28"],
                "days_since_last_sale": cr["days_since_last_sale"],
                "sale_frequency_30d": cr["sale_frequency_30d"],
                "max_sale_30d": cr["max_sale_30d"],
                "total_sales_7d": cr["total_sales_7d"],
                "total_sales_30d": cr["total_sales_30d"],
                "avg_price": cr["avg_price"],
                "product_age_days": cr["product_age_days"],
                "target_30d": fs,
            })
    
    df_agg = pd.DataFrame(windows_data)
    df_agg["sale_date"] = pd.to_datetime(df_agg["sale_date"])
    return df_agg


def apply_improvements(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Cellule 4 du notebook :
      - Retire les zéros du dataset agrégé (target_30d > 0)
      - Crée les features d'interaction
      - Log-transform de la target
    """
    df_agg = df_agg.copy()
    
    # ── Retirer les zéros ──
    avant = len(df_agg)
    df_agg = df_agg[df_agg["target_30d"] > 0].copy()
    logger.info(f"  Retrait zéros : {avant:,} → {len(df_agg):,} lignes")
    
    # ── Features d'interaction ──
    df_agg["volume_x_freq"] = (
        df_agg["rolling_mean_30"] * df_agg["sale_frequency_30d"]
    )
    df_agg["recent_vs_old"] = (
        df_agg["total_sales_7d"] / df_agg["total_sales_30d"].replace(0, 1)
    )
    df_agg["demand_stability"] = (
        df_agg["rolling_std_30"] / df_agg["rolling_mean_30"].replace(0, 1)
    )
    logger.info("  Features interaction : volume_x_freq, recent_vs_old, demand_stability")
    
    # ── Log-transform de la target ──
    df_agg["target_log"] = np.log1p(df_agg["target_30d"])
    logger.info("  Log-transform target")
    
    return df_agg


# ═══════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════

def run_preprocess_final(input_path: str, output_path: str) -> str:
    """
    Pipeline complet : features avancées + agrégation + improvements.
    
    Args:
        input_path: sales_step4_b2c.csv (sortie de filter_b2c.py)
        output_path: sales_train_ready.csv (prêt pour le train)
    
    Returns:
        Chemin du CSV produit.
    """
    # ── 1. Chargement ──
    logger.info(f"Lecture de {input_path}...")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = df.sort_values(["product_id", "sale_date"]).reset_index(drop=True)
    logger.info(
        f"  → {len(df):,} lignes, {df['product_id'].nunique()} produits"
    )
    
    # ── 2. Features avancées ──
    logger.info("Calcul des features avancées...")
    df = add_advanced_features(df)
    
    # ── 3. Agrégation en fenêtres de 30 jours ──
    logger.info("Agrégation en fenêtres glissantes...")
    df_agg = aggregate_forecast_windows(df, forecast_horizon=FORECAST_HORIZON)
    logger.info(f"  Agrégé : {len(df_agg):,} lignes")
    
    # ── 4. Improvements (zéros, interactions, log-transform) ──
    logger.info("Application des improvements...")
    df_agg = apply_improvements(df_agg)
    
    # ── 5. Sauvegarde ──
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_agg.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    logger.info(
        f"✓ Sauvegardé : {output_path} "
        f"({size_mb:.1f} Mo, {len(df_agg):,} lignes, {len(df_agg.columns)} colonnes)"
    )
    
    return output_path


# ═══════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    run_preprocess_final(
        input_path=os.getenv("INPUT_PATH",  DEFAULT_INPUT),
        output_path=os.getenv("OUTPUT_PATH", DEFAULT_OUTPUT),
    )