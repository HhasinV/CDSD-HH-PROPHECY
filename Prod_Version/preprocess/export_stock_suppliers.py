"""
Smart Reassort — Export Stock + Suppliers depuis Odoo
========================================================
Exporte 2 CSV pour l'équipe inférence :
  - products_stock.csv : tous les produits actifs avec leur stock
  - suppliers.csv      : toutes les relations produit ↔ fournisseur

Source : Odoo PostgreSQL (pg17)
"""

import os
import pandas as pd
import psycopg2

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
ODOO_URI = {
    "host": "localhost",
    "port": 5434,
    "dbname": "projet_final",
    "user": "postgres",
    "password": "postgres",
}

OUTPUT_DIR        = "/home/henintsoa/Documents/ProjetFInalJedha/newcode/exports"
STOCK_OUTPUT      = f"{OUTPUT_DIR}/products_stock.csv"
SUPPLIERS_OUTPUT  = f"{OUTPUT_DIR}/suppliers.csv"


# ═══════════════════════════════════════════════
# REQUÊTES SQL
# ═══════════════════════════════════════════════

# 1) Stock : tous les produits stockables actifs
STOCK_QUERY = """
SELECT 
    pp.id AS product_id,
    coalesce(pt.name::jsonb ->> 'fr_FR', pt.name::jsonb ->> 'en_US', pt.name::text) AS product_name,
    pc.complete_name AS category,
    pt.list_price,
    coalesce(SUM(sq.quantity), 0) AS qty_available
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

# 2) Suppliers : toutes les relations product_supplierinfo
SUPPLIERS_QUERY = """
SELECT 
    pp.id AS product_id,
    coalesce(pt.name::jsonb ->> 'fr_FR', pt.name::jsonb ->> 'en_US', pt.name::text) AS product_name,
    rp.name AS supplier_name,
    psi.delay AS delay_days,
    psi.price AS supplier_price,
    psi.min_qty AS min_order_qty,
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


# ═══════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════

def main():
    print(f"Connexion à Odoo ({ODOO_URI['host']}:{ODOO_URI['port']})...")
    conn = psycopg2.connect(**ODOO_URI)
    
    # ── Stock ──
    print("\n[1/2] Export products_stock.csv...")
    df_stock = pd.read_sql(STOCK_QUERY, conn)
    print(f"  → {len(df_stock):,} produits")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_stock.to_csv(STOCK_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"  ✓ {STOCK_OUTPUT}")
    
    # ── Suppliers ──
    print("\n[2/2] Export suppliers.csv...")
    df_sup = pd.read_sql(SUPPLIERS_QUERY, conn)
    print(f"  → {len(df_sup):,} relations produit-fournisseur")
    
    df_sup.to_csv(SUPPLIERS_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"  ✓ {SUPPLIERS_OUTPUT}")
    
    conn.close()
    print("\n✓ Terminé")


if __name__ == "__main__":
    main()