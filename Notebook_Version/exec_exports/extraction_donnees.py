"""
============================================================
Prophecy — Smart Reassort
Extraction : PostgreSQL (Odoo) → exports/sales_history_clean.csv
VERSION 3 — psycopg2 direct (comme export_stock_suppliers.py)
============================================================

Rôle : produire l'historique de ventes DENSE (grille produit × jour)
qui alimente le pipeline de features :
    01_features_temporal_lags.py → 02_features_rolling.py → 03_filter_b2c.py

Périmètre appliqué à la source :
  - ventes depuis 2023
  - hors catégorie Starlink
Le B2B n'est pas filtré ici : détecté statistiquement en aval (étape 03).

Usage :
    python extraction_donnees.py
"""

import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

print("extraction_donnees.py — version 3 (psycopg2 direct)")

# ------------------------------------------------------------
# 1. Connexion — mêmes conventions que export_stock_suppliers.py
# ------------------------------------------------------------
load_dotenv()

ODOO_URI = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", "5434")),
    "dbname":   os.getenv("PG_DB", "projet_final"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "postgres"),
}

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "exports"))
OUTPUT_CSV = OUTPUT_DIR / "sales_history_clean.csv"

# ------------------------------------------------------------
# 2. Requête ventes
#    NB : avec une connexion psycopg2 directe et pd.read_sql sans
#    paramètres, les % du ILIKE sont transmis tels quels — aucun
#    échappement nécessaire.
# ------------------------------------------------------------
SQL_VENTES = """
SELECT
    so.date_order::date AS sale_date,
    pp.id               AS product_id,
    coalesce(pt.name::jsonb ->> 'fr_FR',
             pt.name::jsonb ->> 'en_US',
             pt.name::text)              AS product_name,
    pc.complete_name                     AS category,
    pt.list_price,
    SUM(sol.product_uom_qty)             AS qty_sold
FROM sale_order_line sol
INNER JOIN sale_order       so ON so.id = sol.order_id
INNER JOIN product_product  pp ON pp.id = sol.product_id
INNER JOIN product_template pt ON pt.id = pp.product_tmpl_id
LEFT  JOIN product_category pc ON pc.id = pt.categ_id
WHERE so.state IN ('sale', 'done')
    AND so.date_order >= '2023-01-01'
    AND (pc.complete_name IS NULL
         OR pc.complete_name NOT ILIKE '%....%')
    AND pp.active = true
    AND pt.active = true
    AND pt.type = 'product'
GROUP BY 1, 2, 3, 4, 5
ORDER BY 1, 2
"""


# ------------------------------------------------------------
# 3. Densification : grille complète produit × jour
# ------------------------------------------------------------
def densifier(ventes: pd.DataFrame) -> pd.DataFrame:
    """La table des ventes ne contient que les jours AVEC ventes.
    On reconstruit tous les jours (qty_sold=0 sinon) pour que les lags
    et rolling des étapes 01/02 soient justes ; la fenêtre contexte
    (étape 03) élaguera ensuite les zéros isolés."""
    dates = pd.date_range(ventes["sale_date"].min(), ventes["sale_date"].max(), freq="D")
    produits = ventes[["product_id", "product_name", "category", "list_price"]] \
        .drop_duplicates("product_id")

    grille = (
        pd.MultiIndex.from_product(
            [produits["product_id"], dates], names=["product_id", "sale_date"]
        )
        .to_frame(index=False)
        .merge(produits, on="product_id", how="left")
        .merge(ventes[["sale_date", "product_id", "qty_sold"]],
               on=["sale_date", "product_id"], how="left")
    )
    grille["qty_sold"] = grille["qty_sold"].fillna(0)
    return grille[["sale_date", "product_id", "product_name",
                   "category", "list_price", "qty_sold"]]


# ------------------------------------------------------------
# 4. Pipeline principal
# ------------------------------------------------------------
def main():
    print(f"Connexion à Odoo ({ODOO_URI['host']}:{ODOO_URI['port']}, "
          f"base {ODOO_URI['dbname']})…")
    conn = psycopg2.connect(**ODOO_URI)

    print("Extraction des ventes (périmètre : ≥ 2023, hors Starlink)…")
    ventes = pd.read_sql(SQL_VENTES, conn, parse_dates=["sale_date"])
    conn.close()
    print(f"  → {len(ventes):,} lignes, {ventes['product_id'].nunique()} produits, "
          f"du {ventes['sale_date'].min().date()} au {ventes['sale_date'].max().date()}")

    print("Densification produit × jour…")
    df = densifier(ventes)
    part_zeros = (df["qty_sold"] == 0).mean() * 100
    print(f"  → grille : {len(df):,} lignes ({part_zeros:.0f}% de jours à zéro vente)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ {OUTPUT_CSV} ({OUTPUT_CSV.stat().st_size / 1e6:.1f} Mo)")
    print("   Étape suivante : python features_temporal_lags.py")


if __name__ == "__main__":
    main()
