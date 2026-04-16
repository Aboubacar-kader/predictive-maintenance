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
from datetime import datetime

import pandas as pd
import streamlit as st

# ── Config en tout premier (avant tout autre appel Streamlit) ─
st.set_page_config(
    page_title="Predictive Maintenance Pro",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports internes ─────────────────────────────────────────
from config.settings import REF_DATE
from modules.data_loader import apply_filters, load_data
from modules.feature_engineering import compute_equipment_stats, prepare_ml_features
from modules.ml_model import build_model, enrich_equipment_stats
from modules.styles import hero_banner, inject_css
from pages import equipment_health, overview, planning, predictive_model, reports

# Suppression ciblée uniquement des avertissements tiers connus et non actionnables.
# Ne pas utiliser filterwarnings("ignore") global qui masquerait des problèmes réels.
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")


def build_sidebar(df_equip: pd.DataFrame) -> tuple:
    """Affiche la sidebar et retourne les paramètres de filtrage."""
    with st.sidebar:
        st.markdown("## ⚙️ Maintenance Pro")
        st.markdown("---")
        st.markdown("### 🎚 Filtres")

        sel_types = st.multiselect(
            "Type d'équipement",
            options=sorted(df_equip["equipment_type"].unique()),
            default=sorted(df_equip["equipment_type"].unique()),
        )
        sel_tiers = st.multiselect(
            "Classe de coût",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
        dr = st.date_input(
            "Période d'analyse",
            value=(datetime(2020, 1, 1), datetime(2024, 12, 31)),
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2024, 12, 31),
        )

        st.markdown("---")
        st.markdown("**📦 Jeu de données**")
        st.metric("Équipements", len(df_equip))
        st.markdown("---")
        st.caption("Référence : Jan 2020 – Déc 2024")

    return sel_types, sel_tiers, dr


def parse_date_range(dr) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convertit la valeur du widget date_input en deux Timestamps."""
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        return pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
    return pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-31")


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

    # ── Sidebar → filtres ─────────────────────────────────
    sel_types, sel_tiers, dr = build_sidebar(df_equip_raw)
    d0, d1 = parse_date_range(dr)

    # ── Application des filtres ───────────────────────────
    df_equip, df_maint = apply_filters(
        df_equip_raw, df_maint_raw, sel_types, sel_tiers, d0, d1
    )

    if df_equip.empty:
        st.warning("Aucun équipement ne correspond aux filtres sélectionnés.")
        return

    # ── Pipeline ML (cached via les décorateurs des modules) ─
    eq_stats, model_res = run_pipeline(df_equip, df_maint)

    # ── Hero banner ───────────────────────────────────────
    hero_banner(
        title    = "Predictive Maintenance & Planning",
        subtitle = "Tableau de bord intelligent · Analyse industrielle 2020 – 2024",
    )

    # ── Plan de maintenance (utilisé par reports aussi) ───
    plan = eq_stats[[
        "equipment_id", "equipment_type",
        "risk_level", "risk_score",
        "days_to_fail", "next_failure", "rec_pm",
        "last_cm_date", "last_pm_date",
        "mtbf_days", "cm_per_year",
    ]].copy()
    plan["priority"] = plan["risk_level"].map({
        "Critique": "🔴 P1 – Urgent",
        "Modéré":   "🟠 P2 – Planifier",
        "Faible":   "🟢 P3 – Surveiller",
    })
    plan = plan.sort_values("risk_score", ascending=False)

    # ── Onglets ───────────────────────────────────────────
    tabs = st.tabs([
        "📊 Vue d'ensemble",
        "🔍 Santé des équipements",
        "🤖 Modèle prédictif",
        "📅 Planification",
        "📈 Tendances & Rapports",
    ])

    with tabs[0]:
        overview.render(df_equip, df_maint, eq_stats)

    with tabs[1]:
        equipment_health.render(eq_stats)

    with tabs[2]:
        predictive_model.render(eq_stats, model_res)

    with tabs[3]:
        planning.render(eq_stats)

    with tabs[4]:
        reports.render(df_equip, df_maint, eq_stats, plan)


if __name__ == "__main__":
    main()
