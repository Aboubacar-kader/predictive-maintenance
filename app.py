"""
app.py — Point d'entrée de l'application Predictive Maintenance Pro.

Ce fichier est intentionnellement slim :
  - Configuration Streamlit
  - Chargement des données & pipeline ML (via les modules)
  - Sidebar (filtres)
  - Hero banner
  - Dispatch des onglets vers les modules pages/

Toute la logique métier et de visualisation est dans config/, modules/ et pages/.
"""

import warnings

import streamlit as st

# ── Config en tout premier (avant tout autre appel Streamlit) ─
st.set_page_config(
    page_title="Predictive Maintenance Pro",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Imports internes ─────────────────────────────────────────
from modules.data_loader import load_data
from modules.feature_engineering import compute_equipment_stats, prepare_ml_features
from modules.ml_model import build_model, enrich_equipment_stats
from modules.styles import hero_banner, inject_css
from pages import equipment_health, overview, planning, predictive_model, reports

# Suppression ciblée uniquement des avertissements tiers connus et non actionnables.
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")


def build_sidebar(n_equip: int) -> None:
    """Sidebar épurée — informations générales uniquement."""
    with st.sidebar:
        st.markdown("## ⚙️ Maintenance Pro")
        st.markdown("---")
        st.markdown("**📦 Jeu de données**")
        st.metric("Équipements total", n_equip)
        st.markdown("---")
        st.caption("Période : Jan 2020 – Déc 2024")


def run_pipeline(
    df_equip: pd.DataFrame,
    df_maint: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pipeline complet :
      1. Calcul des statistiques par équipement
      2. Préparation des features ML
      3. Entraînement XGBoost (mis en cache)
      4. Enrichissement du DataFrame avec scores de risque et planning
    """
    eq_stats        = compute_equipment_stats(df_maint, df_equip)
    X, y            = prepare_ml_features(eq_stats)
    model_res       = build_model(X, y)
    eq_stats_full   = enrich_equipment_stats(eq_stats, model_res["proba"])
    return eq_stats_full, model_res


def main() -> None:
    inject_css()

    # ── Chargement des données brutes (cached) ────────────
    df_equip_raw, df_maint_raw = load_data()

    # ── Sidebar épurée ────────────────────────────────────
    build_sidebar(len(df_equip_raw))

    # ── Pipeline ML sur la totalité du parc (cached) ─────
    # Les filtres sont gérés dans chaque page ; le modèle
    # est entraîné une seule fois sur les 80 équipements.
    eq_stats, model_res = run_pipeline(df_equip_raw, df_maint_raw)

    # ── Hero banner ───────────────────────────────────────
    hero_banner(
        title    = "Predictive Maintenance & Planning",
        subtitle = "Tableau de bord intelligent · Analyse industrielle 2020 – 2024",
    )

    # ── Onglets ───────────────────────────────────────────
    tabs = st.tabs([
        "📊 Vue d'ensemble",
        "🔍 Santé des équipements",
        "🤖 Modèle prédictif",
        "📅 Planification",
        "📈 Tendances & Rapports",
    ])

    with tabs[0]:
        overview.render(df_equip_raw, df_maint_raw, eq_stats)

    with tabs[1]:
        equipment_health.render(eq_stats)

    with tabs[2]:
        predictive_model.render(eq_stats, model_res)

    with tabs[3]:
        planning.render(eq_stats)

    with tabs[4]:
        reports.render(df_equip_raw, df_maint_raw, eq_stats)


if __name__ == "__main__":
    main()
