"""
============================================================
Prophecy — Smart Reassort
Export Stock + Suppliers : PostgreSQL (Odoo) → exports/
============================================================

Produit les 2 CSV consommés par la partie inférence / réassort :
  - exports/products_stock.csv : produits stockables actifs + stock disponible
  - exports/suppliers.csv      : relations produit ↔ fournisseur (délais réels)

Version harmonisée avec extraction_donnees.py :
  - identifiants via .env (jamais en dur dans le code)
  - sortie dans exports/ (chemin relatif, surchargeable via OUTPUT_DIR)
  - requêtes SQL identiques à la version d'origine (éprouvées sur la base)

Usage :
    python export_stock_suppliers.py
"""

import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

print("export_stock_suppliers.py — version .env")

# ------------------------------------------------------------
# 1. Connexion
# ------------------------------------------------------------
load_dotenv()

ODOO_URI = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", "5434")),
    "dbname":   os.getenv("PG_DB", "projet_final"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "postgres"),
}

OUTPUT_DIR       = Path(os.getenv("OUTPUT_DIR", "exports"))
STOCK_OUTPUT     = OUTPUT_DIR / "products_stock.csv"
SUPPLIERS_OUTPUT = OUTPUT_DIR / "suppliers.csv"

# ------------------------------------------------------------
# 2. Requêtes SQL
# ------------------------------------------------------------

# Stock : tous les produits stockables actifs, avec le disponible
# des emplacements internes (LEFT JOIN : un produit sans stock_quant
# apparaît quand même, avec qty_available = 0)
STOCK_QUERY = """
SELECT
    pp.id AS product_id,
    coalesce(pt.name::jsonb ->> 'fr_FR',
             pt.name::jsonb ->> 'en_US',
             pt.name::text)        AS product_name,
    pc.complete_name               AS category,
    pt.list_price,
    coalesce(SUM(sq.quantity), 0)  AS qty_available
FROM product_product pp
INNER JOIN product_template pt ON pt.id = pp.product_tmpl_id
LEFT JOIN product_category pc  ON pc.id = pt.categ_id
LEFT JOIN stock_quant sq       ON sq.product_id = pp.id
LEFT JOIN stock_location sl    ON sl.id = sq.location_id
                              AND sl.usage = 'internal'
WHERE pp.active = true
    AND pt.active = true
    AND pt.type = 'product'
GROUP BY pp.id, pt.name, pc.complete_name, pt.list_price
ORDER BY pp.id
"""

# Suppliers : toutes les relations product_supplierinfo.
# is_preferred = true pour le fournisseur de plus petite séquence
# (le fournisseur « préféré » d'Odoo) — c'est son delay_days qui
# alimente le stock de sécurité : Z × σ30 × √délai
SUPPLIERS_QUERY = """
SELECT
    pp.id AS product_id,
    coalesce(pt.name::jsonb ->> 'fr_FR',
             pt.name::jsonb ->> 'en_US',
             pt.name::text)  AS product_name,
    rp.name                  AS supplier_name,
    psi.delay                AS delay_days,
    psi.price                AS supplier_price,
    psi.min_qty              AS min_order_qty,
    psi.date_start,
    psi.date_end,
    CASE
        WHEN psi.sequence = MIN(psi.sequence) OVER (PARTITION BY pp.id)
        THEN true
        ELSE false
    END AS is_preferred
FROM product_supplierinfo psi
INNER JOIN product_template pt ON pt.id = psi.product_tmpl_id
INNER JOIN product_product pp  ON pp.product_tmpl_id = pt.id
INNER JOIN res_partner rp      ON rp.id = psi.partner_id
WHERE pp.active = true
    AND pt.active = true
ORDER BY pp.id, psi.sequence
"""


# ------------------------------------------------------------
# 3. Export
# ------------------------------------------------------------
def main():
    print(f"Connexion à Odoo ({ODOO_URI['host']}:{ODOO_URI['port']}, "
          f"base {ODOO_URI['dbname']})…")
    conn = psycopg2.connect(**ODOO_URI)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stock ──
    print("\n[1/2] Export products_stock.csv…")
    df_stock = pd.read_sql(STOCK_QUERY, conn)
    print(f"  → {len(df_stock):,} produits "
          f"({(df_stock['qty_available'] > 0).sum():,} avec stock > 0)")
    df_stock.to_csv(STOCK_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"  ✓ {STOCK_OUTPUT}")

    # ── Suppliers ──
    print("\n[2/2] Export suppliers.csv…")
    df_sup = pd.read_sql(SUPPLIERS_QUERY, conn)
    n_pref = df_sup["is_preferred"].sum()
    print(f"  → {len(df_sup):,} relations produit-fournisseur "
          f"({n_pref:,} fournisseurs préférés)")
    if len(df_sup):
        print(f"  → délais : min {df_sup['delay_days'].min()} j, "
              f"médiane {df_sup['delay_days'].median():.0f} j, "
              f"max {df_sup['delay_days'].max()} j")
    df_sup.to_csv(SUPPLIERS_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"  ✓ {SUPPLIERS_OUTPUT}")

    conn.close()
    print("\n✓ Terminé — le notebook peut maintenant lire les 3 CSV du dossier exports/")


if __name__ == "__main__":
    main()
