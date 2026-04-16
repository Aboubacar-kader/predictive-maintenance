"""
pages/reports.py
Tab 5 — Tendances & Rapports : analyses temporelles, comparaisons par type
         et export CSV des données.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from modules.charts import (
    chart_annual_downtime,
    chart_annual_operations,
    chart_downtime_top_equipment,
    chart_mtbf_by_type,
    chart_pm_ratio_trend,
    chart_top_failure_causes,
)
from modules.styles import fmt_date, section_header


def render(
    df_equip: pd.DataFrame,
    df_maint: pd.DataFrame,
    eq_stats: pd.DataFrame,
    plan: pd.DataFrame,
) -> None:
    """
    Affiche l'onglet Tendances & Rapports.

    Paramètres
    ----------
    df_equip  : équipements filtrés
    df_maint  : opérations de maintenance filtrées
    eq_stats  : statistiques enrichies par équipement
    plan      : DataFrame de planification (pour l'export)
    """
    section_header("📈 Tendances historiques & Rapports")

    # ── Opérations annuelles + Downtime annuel ────────────
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_annual_operations(df_maint), width="stretch")
    with c2:
        st.plotly_chart(chart_annual_downtime(df_maint), width="stretch")

    # ── MTBF par type + Top pannes ─────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(chart_mtbf_by_type(eq_stats), width="stretch")
    with c4:
        st.plotly_chart(chart_top_failure_causes(df_maint), width="stretch")

    # ── Ratio PM mensuel ──────────────────────────────────
    st.markdown("##### Évolution du ratio préventif mensuel")
    st.plotly_chart(chart_pm_ratio_trend(df_maint), width="stretch")

    # ── Top 15 équipements downtime ───────────────────────
    st.markdown("##### Top 15 équipements — Downtime cumulé")
    st.plotly_chart(
        chart_downtime_top_equipment(df_maint, df_equip),
        width="stretch",
    )

    # ── Export CSV ────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📥 Export des données")

    today = datetime.now().strftime("%Y%m%d")

    # Prépare l'export équipements
    exp_eq = eq_stats[[
        "equipment_id", "equipment_type", "age_years",
        "risk_score", "risk_level", "mtbf_days",
        "cm_per_year", "total_downtime_h", "days_to_fail",
    ]].rename(columns={"total_downtime_h": "total_downtime_hours"})

    # Prépare l'export plan (dates formatées)
    exp_plan = plan.copy()
    for c in ["next_failure", "rec_pm", "last_cm_date", "last_pm_date"]:
        if c in exp_plan.columns:
            exp_plan[c] = exp_plan[c].apply(fmt_date)

    # Prépare l'export historique (supprime Period non sérialisable)
    exp_hist = df_maint.drop(columns=["yearmonth"], errors="ignore")

    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button(
            label="📊 Analyse équipements (CSV)",
            data=exp_eq.to_csv(index=False),
            file_name=f"analyse_equipements_{today}.csv",
            mime="text/csv",
            width="stretch",
        )
    with e2:
        st.download_button(
            label="📅 Plan de maintenance (CSV)",
            data=exp_plan.to_csv(index=False),
            file_name=f"plan_maintenance_{today}.csv",
            mime="text/csv",
            width="stretch",
        )
    with e3:
        st.download_button(
            label="📋 Historique complet (CSV)",
            data=exp_hist.to_csv(index=False),
            file_name=f"historique_maintenance_{today}.csv",
            mime="text/csv",
            width="stretch",
        )

    # ── Documentation technique ───────────────────────────
    st.markdown("---")
    st.markdown("#### 📖 Documentation technique")
    try:
        with open("documentation_technique.html", "r", encoding="utf-8") as f:
            doc_html = f.read()
        st.download_button(
            label="📄 Télécharger la documentation technique (HTML)",
            data=doc_html,
            file_name="documentation_technique_predictive_maintenance.html",
            mime="text/html",
            width="stretch",
        )
        st.caption(
            "Ouvrez le fichier dans votre navigateur, puis utilisez "
            "Ctrl+P → Enregistrer en PDF pour obtenir un PDF imprimable."
        )
    except FileNotFoundError:
        st.info("Fichier documentation_technique.html introuvable.")
