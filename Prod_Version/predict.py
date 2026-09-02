"""
Smart Reassort — predict.py
============================
Charge le modèle XGBoost, prédit la demande 30j par produit,
et génère le tableau de réassort complet.
 
Ce fichier est appelé par l'API FastAPI (api.py).
Peut aussi être exécuté seul pour tester.
 
Usage : python predict.py
"""


import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
 
 
# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
DATA_DIR = "newcode/exports"
MODEL_PATH = "newcode/train/model.pkl"
FORECAST_HORIZON = 30
SAFETY_STOCK_DAYS = 7
DEFAULT_LEAD_TIME = 5  # jours par défaut si pas de fournisseur
 
 
# ══════════════════════════════════════════════
# 1. CHARGER LE MODÈLE
# ══════════════════════════════════════════════
def load_model(model_path: str = MODEL_PATH):
    """Charge le modèle XGBoost depuis le pickle."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_cols"], data.get("best_params", {})
 
 
# ══════════════════════════════════════════════
# 2. PRÉPARER LES FEATURES POUR LA PRÉDICTION
# ══════════════════════════════════════════════
def prepare_prediction_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Prend le dataset B2C avec toutes les features déjà calculées,
    et prépare une ligne par produit avec les features les plus récentes.
    """
    df = df.sort_values(["product_id", "sale_date"])
 
    # Pour chaque produit, prendre la dernière ligne (état le plus récent)
    latest = df.groupby("product_id").last().reset_index()
 
    # S'assurer que toutes les features existent
    for col in feature_cols:
        if col not in latest.columns:
            latest[col] = 0
 
    return latest
 
 
# ══════════════════════════════════════════════
# 3. PRÉDIRE LA DEMANDE
# ══════════════════════════════════════════════
def predict_demand(model, latest_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Prédit la demande 30j pour chaque produit."""
    X = latest_df[feature_cols]
 
    # Prédiction (le modèle prédit en log, on inverse)
    y_pred_log = model.predict(X)
    y_pred = np.maximum(0, np.expm1(y_pred_log))
    y_pred_rounded = np.round(y_pred).astype(int)
 
    latest_df = latest_df.copy()
    latest_df["ventes_prevues_30j"] = y_pred_rounded
 
    return latest_df
 
 
# ══════════════════════════════════════════════
# 4. CALCULER LE STATUT CYCLE DE VIE
# ══════════════════════════════════════════════
def compute_cycle_status(row):
    """
    Détermine le cycle de vie basé sur trend_30_90.
    Croissance / Maturité / Déclin / Obsolescence
    """
    r30 = row.get("rolling_mean_30", 0)
    r60 = row.get("rolling_mean_60", 0)
    trend = row.get("trend_30_90", 0)
    freq = row.get("sale_frequency_30d", 0)
 
    # Pas de vente récente du tout
    if r30 == 0 and freq == 0:
        return "Inactif"
 
    if trend == 0:
        return "Nouveau" if r30 > 0 else "Inactif"
    elif trend > 1.15:
        return "Croissance"
    elif trend > 0.85:
        return "Maturité"
    elif trend > 0.5:
        return "Déclin"
    else:
        return "Obsolescence"
 
 
# ══════════════════════════════════════════════
# 5. GÉNÉRER LE TABLEAU DE RÉASSORT
# ══════════════════════════════════════════════
def generate_reassort(
    predictions_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    suppliers_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Génère le tableau de réassort complet :
    - Ventes prévues (ML)
    - Stock actuel
    - Statut cycle de vie
    - Quantité à commander
    - Fournisseur recommandé
    - Date idéale de livraison
    - Alerte
    """
    today = pd.Timestamp.now()
 
    # ── Cycle de vie ──
    predictions_df["cycle_status"] = predictions_df.apply(compute_cycle_status, axis=1)
 
    # ── Demande journalière moyenne ──
    predictions_df["avg_daily_demand"] = (
        predictions_df["ventes_prevues_30j"] / FORECAST_HORIZON
    ).round(2)
 
    # ── Jointure stock ──
    result = predictions_df.merge(
        stock_df[["product_id", "qty_available"]],
        on="product_id",
        how="left",
    )
    result["qty_available"] = result["qty_available"].fillna(0)
 
    # Prix de vente = avg_price (moyenne des prix_unitaire depuis l'historique des ventes)
    result["prix_vente"] = result["avg_price"]
 
    # ── Jointure fournisseur (préféré) ──
    preferred = suppliers_df[suppliers_df["is_preferred"] == True].copy()
    result = result.merge(
        preferred[["product_id", "supplier_name", "delay_days", "supplier_price", "min_order_qty"]],
        on="product_id",
        how="left",
    )
    result["delay_days"] = result["delay_days"].fillna(DEFAULT_LEAD_TIME).astype(int)
    result["supplier_name"] = result["supplier_name"].fillna("Non renseigné")
    result["supplier_price"] = result["supplier_price"].fillna(0)
    result.rename(columns={"supplier_price": "prix_achat"}, inplace=True)
    result["min_order_qty"] = result["min_order_qty"].fillna(1).astype(int)
 
    # Marge (%)
    result["marge_pct"] = result.apply(
        lambda r: round((r["prix_vente"] - r["prix_achat"]) / r["prix_achat"] * 100, 1)
        if r["prix_achat"] > 0 else 0,
        axis=1,
    )
 
    # ── Calculs métier ──
 
    # Stock de sécurité
    result["safety_stock"] = (result["avg_daily_demand"] * SAFETY_STOCK_DAYS).round(0).astype(int)
 
    # Quantité à commander = prévision - stock + sécurité
    result["qty_to_order"] = (
        result["ventes_prevues_30j"] - result["qty_available"] + result["safety_stock"]
    ).clip(lower=0).astype(int)
 
    # Arrondir au min_order_qty
    result["qty_to_order"] = result.apply(
        lambda r: int(np.ceil(r["qty_to_order"] / max(1, r["min_order_qty"])) * r["min_order_qty"])
        if r["qty_to_order"] > 0 else 0,
        axis=1,
    )
 
    # Obsolescence → ne pas commander
    result.loc[result["cycle_status"].isin(["Obsolescence", "Inactif"]), "qty_to_order"] = 0
 
    # Couverture en jours
    result["coverage_days"] = result.apply(
        lambda r: round(r["qty_available"] / r["avg_daily_demand"], 1)
        if r["avg_daily_demand"] > 0 else 999,
        axis=1,
    )
 
    # Date idéale de livraison
    result["ideal_delivery"] = result.apply(
        lambda r: (today + pd.Timedelta(
            days=max(0, r["coverage_days"] - SAFETY_STOCK_DAYS)
        )).strftime("%Y-%m-%d")
        if r["avg_daily_demand"] > 0 and r["coverage_days"] < 999 else "—",
        axis=1,
    )
 
    # Date limite commande (livraison - délai fournisseur)
    result["order_by_date"] = result.apply(
        lambda r: (pd.Timestamp(r["ideal_delivery"]) - pd.Timedelta(
            days=r["delay_days"]
        )).strftime("%Y-%m-%d")
        if r["ideal_delivery"] != "—" else "—",
        axis=1,
    )
 
    # Coût estimé
    result["estimated_cost"] = (result["qty_to_order"] * result["prix_achat"]).round(0)
 
    # Alerte
    def get_alert(row):
        if row["cycle_status"] in ["Obsolescence", "Inactif"]:
            return "Ne pas recommander"
        elif row["cycle_status"] == "Croissance" and row["qty_to_order"] > 0:
            return "Forte demande"
        elif row["coverage_days"] < row["delay_days"]:
            return "Rupture imminente"
        elif row["coverage_days"] < row["delay_days"] + SAFETY_STOCK_DAYS:
            return "À commander"
        elif row["qty_to_order"] > 0:
            return "Stable"
        else:
            return "Stock OK"
 
    result["alert"] = result.apply(get_alert, axis=1)
 
    # Trier par priorité
    alert_order = {
        "Rupture imminente": 0,
        "Forte demande": 1,
        "À commander": 2,
        "Stable": 3,
        "Stock OK": 4,
        "Ne pas recommander": 5,
    }
    result["alert_sort"] = result["alert"].map(alert_order).fillna(6)
    result = result.sort_values(["alert_sort", "coverage_days"]).drop(columns=["alert_sort"])
 
    # Score d'obsolescence (0 = pas de risque, 1 = obsolète)
    def compute_obsolescence_score(row):
        trend = row.get("trend_30_90", 0)
        freq = row.get("sale_frequency_30d", 0)
        days_since = row.get("days_since_last_sale", 999)
        avg_demand = row.get("avg_daily_demand", 0)
        stock = row.get("qty_available", 0)
 
        # Facteur tendance (baisse = risque)
        if trend == 0:
            trend_score = 0.5
        elif trend > 1:
            trend_score = 0.0
        else:
            trend_score = min(1.0, max(0.0, 1 - trend))
 
        # Facteur fréquence (peu fréquent = risque)
        freq_score = max(0, min(1.0, 1 - freq / 15))
 
        # Facteur jours sans vente (longtemps = risque)
        days_score = min(1.0, days_since / 90)
 
        # Facteur surstock (beaucoup de stock vs demande = risque)
        if avg_demand > 0:
            coverage = stock / (avg_demand * 30)
            overstock_score = min(1.0, max(0, coverage - 1))
        else:
            overstock_score = 1.0 if stock > 0 else 0.0
 
        # Score final pondéré
        score = (0.35 * trend_score + 0.25 * freq_score +
                 0.25 * days_score + 0.15 * overstock_score)
        return round(min(1.0, max(0.0, score)), 3)
 
    result["obsolescence_score"] = result.apply(compute_obsolescence_score, axis=1)
 
    # Colonnes finales
    final_cols = [
        "product_id", "product_name", "category_id",
        "cycle_status", "obsolescence_score",
        "ventes_prevues_30j", "avg_daily_demand",
        "qty_available", "coverage_days", "safety_stock",
        "qty_to_order", "prix_vente", "prix_achat", "marge_pct",
        "supplier_name", "delay_days", "estimated_cost",
        "ideal_delivery", "order_by_date", "alert",
    ]
 
    # Ne garder que les colonnes qui existent
    final_cols = [c for c in final_cols if c in result.columns]
 
    return result[final_cols]
 
 
# ══════════════════════════════════════════════
# 6. FONCTION PRINCIPALE
# ══════════════════════════════════════════════
def run_reassort(
    data_dir: str = DATA_DIR,
    model_path: str = MODEL_PATH,
) -> pd.DataFrame:
    """
    Fonction principale appelée par l'API.
    Retourne le DataFrame du tableau de réassort.
    """
    # Charger le modèle
    model, feature_cols, _ = load_model(model_path)
 
    # Charger les données
    df = pd.read_csv(
        os.path.join(data_dir, "sales_step4_b2c.csv"),
        encoding="utf-8-sig",
    )
    df["sale_date"] = pd.to_datetime(df["sale_date"])
 
    stock_df = pd.read_csv(
        os.path.join(data_dir, "products_stock.csv"),
        encoding="utf-8-sig",
    )
 
    suppliers_df = pd.read_csv(
        os.path.join(data_dir, "suppliers.csv"),
        encoding="utf-8-sig",
    )
 
    # Ajouter les features supplémentaires
    df = df.sort_values(["product_id", "sale_date"]).reset_index(drop=True)
    products = df["product_id"].unique()
 
    # days_since_last_sale
    df["days_since_last_sale"] = 0
    for pid in products:
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
 
    avg_price = df[df["prix_unitaire"] > 0].groupby("product_id")["prix_unitaire"].mean()
    df["avg_price"] = df["product_id"].map(avg_price).fillna(0)
 
    first_sale = df[df["qty_sold"] > 0].groupby("product_id")["sale_date"].min()
    df["first_sale_date"] = df["product_id"].map(first_sale)
    df["product_age_days"] = (df["sale_date"] - df["first_sale_date"]).dt.days
    df["product_age_days"] = df["product_age_days"].fillna(0).clip(lower=0)
    df = df.drop(columns=["first_sale_date"])
 
    # Features d'interaction
    df["volume_x_freq"] = df["rolling_mean_30"] * df["sale_frequency_30d"]
    df["recent_vs_old"] = df["total_sales_7d"] / df["total_sales_30d"].replace(0, 1)
    df["demand_stability"] = df["rolling_std_30"] / df["rolling_mean_30"].replace(0, 1)
 
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
 
    # Préparer les features (dernière ligne par produit)
    latest = prepare_prediction_features(df, feature_cols)
 
    # Prédire
    predictions = predict_demand(model, latest, feature_cols)
 
    # Générer le réassort
    reassort = generate_reassort(predictions, stock_df, suppliers_df)
 
    return reassort
 
 

    
def get_category_list(data_dir: str = DATA_DIR):
    data = pd.read_csv(
        os.path.join(data_dir, "sales_step4_b2c.csv"),
        encoding="utf-8-sig",
    )
    data = data[["category_id", "category"]]
    return data.drop_duplicates().to_dict(orient="records")

# ══════════════════════════════════════════════
# MAIN (test en standalone)
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("SMART REASSORT MASSIN")
    print("=" * 60)
 
    reassort = run_reassort()
 
    # Stats
    print(f"\n {len(reassort)} produits analysés")
    print(f"\n  Répartition des alertes :")
    for alert, count in reassort["alert"].value_counts().items():
        print(f"    {alert:25s} {count:>5d}")
 
    print(f"\n  Répartition cycle de vie :")
    for status, count in reassort["cycle_status"].value_counts().items():
        print(f"    {status:20s} {count:>5d}")
 
    total_cost = reassort["estimated_cost"].sum()
    n_to_order = (reassort["qty_to_order"] > 0).sum()
    print(f"\n  Produits à commander : {n_to_order}")
    print(f"  Coût total estimé   : {total_cost:,.0f}")
 
    # Top 20
    top20 = reassort[reassort["qty_to_order"] > 0].head(20)
    print(f"\n  Top 20 produits à commander :")
    print(f"  {'Produit':<40s} {'Prévu':>6s} {'Stock':>6s} {'Cmd':>5s} {'Fournisseur':<20s} {'Alerte'}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*5} {'-'*20} {'-'*20}")
    for _, row in top20.iterrows():
        name = str(row["product_name"])[:40]
        print(f"  {name:<40s} {int(row['ventes_prevues_30j']):>6d} {int(row['qty_available']):>6d} {int(row['qty_to_order']):>5d} {str(row['supplier_name'])[:20]:<20s} {row['alert']}")
 
    # Sauvegarder
    reassort.to_csv("reassort_output.csv", index=False, encoding="utf-8-sig")
    print(f"\n Tableau sauvegardé : reassort_output.csv")