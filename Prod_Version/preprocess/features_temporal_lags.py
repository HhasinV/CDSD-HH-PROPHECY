"""
Smart Reassort — Preprocessing étape 1
=========================================
Features temporelles + Lags (équivalent first_main.ipynb)

Entrée  : exports/sales_history_clean.csv  (sortie de dbt, 8 colonnes)
Sortie  : exports/sales_step1_temporal_lags.csv  (29 colonnes)

Features ajoutées :
  - 15 features temporelles (day_of_week, month, is_weekend, is_holiday, ...)
  - 6 lags par produit (lag_1, lag_3, lag_7, lag_14, lag_21, lag_28)

Usage standalone :
    python newcode/preprocess/01_features_temporal_lags.py

Usage Airflow (plus tard) :
    from newcode.preprocess.features_temporal_lags import run_features
    run_features(input_path="...", output_path="...")
"""

import os
import logging
from datetime import timedelta

import pandas as pd

# ───────────────────────────────────────────────
# LOGGING
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────
# CONSTANTES MÉTIER
# ───────────────────────────────────────────────
LAGS = [1, 3, 7, 14, 21, 28]

# Jours fériés fixes (Madagascar + universels)
FIXED_HOLIDAYS = [
    (1, 1),    # Nouvel An
    (3, 8),    # Journée de la femme
    (3, 29),   # Jour des martyrs
    (5, 1),    # Fête du travail
    (6, 26),   # Fête nationale Madagascar
    (8, 15),   # Assomption
    (11, 1),   # Toussaint
    (12, 25),  # Noël
]


# ═══════════════════════════════════════════════
# FONCTIONS DE TRANSFORMATION
# ═══════════════════════════════════════════════

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute 15 features temporelles basées sur sale_date.
    
    Args:
        df: DataFrame avec une colonne sale_date (datetime).
    
    Returns:
        DataFrame enrichi avec 15 nouvelles colonnes.
    """
    df = df.copy()
    
    # ── Features de base depuis la date ──
    df["day_of_week"]  = df["sale_date"].dt.dayofweek                        # 0=Lundi, 6=Dimanche
    df["day_of_month"] = df["sale_date"].dt.day                              # 1-31
    df["week_of_year"] = df["sale_date"].dt.isocalendar().week.astype(int)   # 1-52
    df["month"]        = df["sale_date"].dt.month                            # 1-12
    df["quarter"]      = df["sale_date"].dt.quarter                          # 1-4
    df["year"]         = df["sale_date"].dt.year
    
    # ── Indicateurs basés sur jour de semaine / mois ──
    df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)             # samedi + dimanche
    df["is_month_start"]  = (df["day_of_month"] <= 5).astype(int)            # 5 premiers jours
    df["is_month_end"]    = (df["day_of_month"] >= 25).astype(int)           # 6 derniers jours
    
    # ── Jours fériés ──
    df["is_holiday"] = 0
    for month, day in FIXED_HOLIDAYS:
        mask = (df["sale_date"].dt.month == month) & (df["sale_date"].dt.day == day)
        df.loc[mask, "is_holiday"] = 1
    
    # ── Veille de jour férié ──
    holiday_dates = set(
        df.loc[df["is_holiday"] == 1, "sale_date"].dt.date
    )
    df["is_pre_holiday"] = df["sale_date"].apply(
        lambda x: 1 if (x.date() + timedelta(days=1)) in holiday_dates else 0
    )
    
    # ── Périodes spéciales ──
    df["is_christmas_period"] = (
        (df["sale_date"].dt.month == 12) & (df["sale_date"].dt.day >= 15)
    ).astype(int)
    
    df["is_summer"] = df["sale_date"].dt.month.isin([6, 7, 8]).astype(int)
    
    df["is_back_to_school"] = (
        (df["sale_date"].dt.month == 9) & (df["sale_date"].dt.day <= 15)
    ).astype(int)
    
    df["is_black_friday_period"] = (
        (df["sale_date"].dt.month == 11) & (df["sale_date"].dt.day >= 20)
    ).astype(int)
    
    return df


def add_lag_features(df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
    """
    Ajoute les lags de qty_sold par produit.
    
    ⚠ Le DataFrame DOIT être trié par (product_id, sale_date) AVANT
       sinon les shifts donneront des résultats faux.
    
    Args:
        df: DataFrame trié par (product_id, sale_date).
        lags: liste des décalages en jours. Default = [1, 3, 7, 14, 21, 28].
    
    Returns:
        DataFrame enrichi avec une colonne par lag.
    """
    if lags is None:
        lags = LAGS
    
    df = df.copy()
    
    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = df.groupby("product_id")["qty_sold"].shift(lag)
        n_nan = df[col_name].isna().sum()
        logger.info(
            f"  {col_name:8s} — NaN : {n_nan:,} ({n_nan/len(df)*100:.1f}%)"
        )
    
    return df


# ═══════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════

def run_features(input_path: str, output_path: str) -> str:
    """
    Pipeline complet : features temporelles + lags.
    
    Args:
        input_path: CSV d'entrée (sortie de dbt).
        output_path: CSV de sortie.
    
    Returns:
        Chemin du CSV produit (utile pour chaîner les tasks Airflow).
    """
    # ── 1. Chargement ──
    logger.info(f"Lecture de {input_path}...")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    logger.info(
        f"  → {len(df):,} lignes, {df['product_id'].nunique()} produits, "
        f"{len(df.columns)} colonnes"
    )
    
    # ── 2. Tri impératif avant les lags ──
    df = df.sort_values(["product_id", "sale_date"]).reset_index(drop=True)
    
    # ── 3. Features temporelles ──
    logger.info("Calcul des 15 features temporelles...")
    df = add_temporal_features(df)
    
    # ── 4. Lags ──
    logger.info(f"Calcul des {len(LAGS)} lags : {LAGS}")
    df = add_lag_features(df, lags=LAGS)
    
    # ── 5. Sauvegarde ──
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    logger.info(
        f"✓ Sauvegardé : {output_path} "
        f"({size_mb:.1f} Mo, {len(df):,} lignes, {len(df.columns)} colonnes)"
    )
    
    return output_path


# ═══════════════════════════════════════════════
# ENTRY POINT (standalone)
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    run_features(
        input_path="/home/henintsoa/Documents/ProjetFInalJedha/newcode/exports/sales_history_clean.csv",
        output_path="/home/henintsoa/Documents/ProjetFInalJedha/newcode/exports/sales_step1_temporal_lags.csv",
    )