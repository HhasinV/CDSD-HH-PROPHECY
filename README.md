# Prophecy — Smart Reassort

> **Le copilote intelligent du réapprovisionnement e-commerce**
> Projet final — Certification Data Full Stack & Lead Data, Jedha Bootcamp
> Auteur : **Henintsoa HASINAVALONA** — NextHope, Madagascar

---

## Le problème

Gérer un réassort manuellement, c'est naviguer entre deux écueils :

- **Surstock** → capital immobilisé (BFR gonflé), produits obsolètes, démarques forcées
- **Sous-stock** → ruptures invisibles, ventes perdues, clients déçus

Prophecy prédit la **quantité vendue par produit sur les 30 prochains jours**, puis en déduit
la **quantité à commander** :

```
Qté_à_commander = max(0, Prédiction_30j + Stock_sécurité − Stock_actuel)
Stock_sécurité  = Z × σ₃₀ × √(délai_fournisseur)     (Z = 1,65 → taux de service 95 %)
```


---

## Architecture


PostgreSQL (Odoo) ──► Airbyte (CDC) ──► dbt ──► Python (features) ──► XGBoost + Optuna
                                                                          
Streamlit ◄── FastAPI ◄── S3 ◄── MLflow (versionnage) ◄──────────────────

Orchestration : Airflow (pipeline en 8 étapes, réentraînement périodique)

## Feature engineering — 34 features

| Famille | Détail |
|---|---|
| **6 lags** | J-1, J-3, J-7, J-14, J-21, J-28 — de l'actualité au rythme mensuel (28 = 4 semaines : effet paie) |
| **Rolling mean × 5** | Fenêtres 7 / 14 / 30 / 60 / 90 j — le niveau de demande à 5 échelles |
| **Rolling std × 5** | La « nervosité » du produit — alimente aussi le stock de sécurité |
| **2 trends** | moy7/moy30 et moy30/moy90 → croissance vs obsolescence, mode vs fond |
| **15 temporelles** | Jours fériés de Madagascar, veille de férié, effet paie, Noël, rentrée, Black Friday… |



## Modèle

- **XGBoost** — objectif **`count:poisson`** : la demande est un *comptage* (masse de zéros
  + petites valeurs). L'objectif quadratique par défaut sur-prédisait de +50 % (biais concentré
  sur les petits volumes) ; Poisson a ramené le biais sous 3 % et réduit l'erreur de 25 %.
- **Optuna** — 120 trials, recherche bayésienne (TPE), validation temporelle.
- **Split temporel** — entraînement sur le passé, test sur les 90 derniers jours.


## Résultats (jeu de test réel)

| Métrique | Valeur | Lecture |
|---|---|---|
| **MAE** | **3,03 articles** | Erreur moyenne par produit et par mois |
| **WMAPE** | **≈ 84,8 %** | Total des articles d'écart ÷ total des articles vendus — niveau attendu sur une demande intermittente (~3 ventes/mois/produit) |
| **Tolérance ± 3** | **86,6 %** | Prédictions opérationnellement justes |
| Biais | −2,7 % | Ni surstock ni rupture systémique |



## Axes d'amélioration

1. **Segmenter avant de modéliser**
2. **LightGBM en challenger** sur les mêmes splits temporels
3. **Historique de stock** (`stock_move`) → distinguer « zéro demande » et « rupture »
4. **Monitoring de dérive** MLflow → réentraînement automatique via Airflow


