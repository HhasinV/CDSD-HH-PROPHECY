"""
Smart Reassort — Preprocessing étape 3 : Fenêtre contexte + Retrait B2B
"""
import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONTEXT_WINDOW = 7
DEFAULT_INPUT  = "/home/henintsoa/Documents/ProjetFInalJedha/newcode/exports/sales_step2_rolling.csv"
DEFAULT_OUTPUT = "/home/henintsoa/Documents/ProjetFInalJedha/newcode/exports/sales_step4_b2c.csv"


def apply_context_window(df: pd.DataFrame, window: int = CONTEXT_WINDOW) -> pd.DataFrame:
    """
    Garde uniquement les lignes dans une fenêtre ±window jours autour d'une vente.
    Supprime les zéros isolés.
    """
    df = df.copy()
    df = df.sort_values(["product_id", "sale_date"]).reset_index(drop=True)
    
    df["keep"] = False
    products = df["product_id"].unique()
    
    for i, pid in enumerate(products):
        if (i + 1) % 200 == 0:
            logger.info(f"  Fenêtre contexte : {i+1}/{len(products)} produits...")
        
        mask = df["product_id"] == pid
        product_df = df.loc[mask]
        sale_indices = product_df[product_df["qty_sold"] > 0].index
        
        for idx in sale_indices:
            pos = product_df.index.get_loc(idx)
            start = max(0, pos - window)
            end = min(len(product_df), pos + window + 1)
            window_indices = product_df.index[start:end]
            df.loc[window_indices, "keep"] = True
    
    df_filtered = df[df["keep"]].drop(columns=["keep"]).reset_index(drop=True)
    logger.info(f"  Fenêtre ±{window}j : {len(df):,} → {len(df_filtered):,} lignes")
    return df_filtered


def detect_and_remove_b2b(df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les produits B2B (ventes anormales : forte variance + gros volumes)
    et les retire.
    """
    ventes = df[df["qty_sold"] > 0]
    
    stats = ventes.groupby(["product_id", "product_name"]).agg(
        qty_moyenne=("qty_sold", "mean"),
        qty_max=("qty_sold", "max"),
        qty_std=("qty_sold", "std"),
    ).reset_index()
    
    stats["cv"] = (stats["qty_std"] / stats["qty_moyenne"]).fillna(0)
    stats["is_b2b"] = (
        ((stats["qty_moyenne"] >= 5) | (stats["qty_max"] >= 15))
        & (stats["cv"] >= 1.5)
    )
    
    b2b_ids = stats[stats["is_b2b"]]["product_id"].tolist()
    logger.info(f"  Produits B2B détectés : {len(b2b_ids)}")
    
    for name in stats[stats["is_b2b"]]["product_name"].head(10):
        logger.info(f"    - {name}")
    if len(b2b_ids) > 10:
        logger.info(f"    ... et {len(b2b_ids) - 10} autres")
    
    df_filtered = df[~df["product_id"].isin(b2b_ids)].reset_index(drop=True)
    logger.info(
        f"  Retrait B2B : "
        f"{df['product_id'].nunique()} → {df_filtered['product_id'].nunique()} produits"
    )
    return df_filtered


def run_filter_b2c(input_path: str, output_path: str) -> str:
    logger.info(f"Lecture de {input_path}...")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    logger.info(f"  → {len(df):,} lignes, {df['product_id'].nunique()} produits")
    
    logger.info(f"Application fenêtre de contexte ±{CONTEXT_WINDOW}j...")
    df = apply_context_window(df, window=CONTEXT_WINDOW)
    
    logger.info("Détection et retrait des produits B2B...")
    df = detect_and_remove_b2b(df)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    logger.info(
        f"✓ {output_path} ({size_mb:.1f} Mo, {len(df):,} lignes, "
        f"{df['product_id'].nunique()} produits B2C, {len(df.columns)} colonnes)"
    )
    return output_path


if __name__ == "__main__":
    run_filter_b2c(
        input_path=os.getenv("INPUT_PATH",  DEFAULT_INPUT),
        output_path=os.getenv("OUTPUT_PATH", DEFAULT_OUTPUT),
    )