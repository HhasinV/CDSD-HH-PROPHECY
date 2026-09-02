"""
Smart Reassort — Preprocessing étape 2 : Rolling means + Trends
"""
import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROLLING_WINDOWS = [7, 14, 30, 60, 90]
DEFAULT_INPUT  = "/home/henintsoa/Documents/ProjetFInalJedha/newcode/exports/sales_step1_temporal_lags.csv"
DEFAULT_OUTPUT = "/home/henintsoa/Documents/ProjetFInalJedha/newcode/exports/sales_step2_rolling.csv"


def add_rolling_features(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    if windows is None:
        windows = ROLLING_WINDOWS
    df = df.copy()
    grouped = df.groupby("product_id")["qty_sold"]
    
    for w in windows:
        df[f"rolling_mean_{w}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).std()
        )
        logger.info(f"  Fenêtre {w:>3d}j ✓")
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trend_7_30"]  = df["rolling_mean_7"]  / df["rolling_mean_30"].replace(0, np.nan)
    df["trend_30_90"] = df["rolling_mean_30"] / df["rolling_mean_90"].replace(0, np.nan)
    df["trend_7_30"]  = df["trend_7_30"].replace([np.inf, -np.inf], 0).fillna(0)
    df["trend_30_90"] = df["trend_30_90"].replace([np.inf, -np.inf], 0).fillna(0)
    return df


def run_rolling(input_path: str, output_path: str) -> str:
    logger.info(f"Lecture de {input_path}...")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    logger.info(f"  → {len(df):,} lignes, {len(df.columns)} colonnes")
    
    df = df.sort_values(["product_id", "sale_date"]).reset_index(drop=True)
    
    logger.info(f"Rolling features (windows = {ROLLING_WINDOWS})...")
    df = add_rolling_features(df, ROLLING_WINDOWS)
    
    logger.info("Trends...")
    df = add_trend_features(df)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    logger.info(f"✓ {output_path} ({size_mb:.1f} Mo, {len(df.columns)} colonnes)")
    return output_path


if __name__ == "__main__":
    run_rolling(
        input_path=os.getenv("INPUT_PATH",  DEFAULT_INPUT),
        output_path=os.getenv("OUTPUT_PATH", DEFAULT_OUTPUT),
    )