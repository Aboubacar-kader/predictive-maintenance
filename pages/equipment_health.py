"""
pages/equipment_health.py
Tab 2 — Santé des équipements : scores de risque, matrice et tableau détaillé.
"""

import numpy as np
import pandas as pd
import streamlit as st

from modules.charts import chart_risk_bar, chart_risk_matrix
from modules.filters import render_filters
from modules.styles import fmt_days, fmt_mtbf, kpi_card, section_header


def render(eq_stats: pd.DataFrame) -> None:
    section_header("🔍 Analyse de la santé du parc machine")

    f = render_filters(eq_stats, key="health")
    eq_stats = f["eq_stats"]

    if eq_stats.empty:
        st.warning("Aucun équipement ne correspond aux filtres sélectionnés.")
        return

    st.markdown("---")
    # ── Compteurs par niveau de risque ────────────────────
    tot = len(eq_stats)
    n_c = (eq_stats["risk_level"] == "Critique").sum()
    n_m = (eq_stats["risk_level"] == "Modéré").sum()
    n_l = (eq_stats["risk_level"] == "Faible").sum()

    r1, r2, r3 = st.columns(3)
    kpi_card(r1, f"{n_c} ({n_c/tot:.0%})", "🔴 Risque Critique", "danger")
    kpi_card(r2, f"{n_m} ({n_m/tot:.0%})", "🟠 Risque Modéré",   "warning")
    kpi_card(r3, f"{n_l} ({n_l/tot:.0%})", "🟢 Faible Risque",   "success")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphiques ────────────────────────────────────────
    bl, br = st.columns([2, 3])
    with bl:
        st.plotly_chart(chart_risk_bar(eq_stats), width="stretch")
    with br:
        st.plotly_chart(chart_risk_matrix(eq_stats), width="stretch")

    # ── Tableau détaillé ──────────────────────────────────
    st.markdown("##### Tableau détaillé des équipements")

    disp = eq_stats[[
        "equipment_id", "equipment_type", "age_years",
        "risk_score", "risk_level",
        "cm_per_year", "mtbf_days", "total_downtime_h",
        "pm_compliance_score", "days_since_last_cm",
    ]].copy()

    disp.columns = [
        "Équipement", "Type", "Âge (ans)",
        "Score risque", "Niveau",
        "Pannes/an", "MTBF (j)", "Downtime (h)",
        "Compliance PM", "Jours dep. panne",
    ]

    disp["Âge (ans)"]        = disp["Âge (ans)"].round(1)
    disp["Score risque"]     = disp["Score risque"].round(1)
    disp["Pannes/an"]        = disp["Pannes/an"].round(1)
    disp["MTBF (j)"]         = disp["MTBF (j)"].apply(fmt_mtbf)
    disp["Downtime (h)"]     = disp["Downtime (h)"].round(1)
    disp["Compliance PM"]    = disp["Compliance PM"].round(2)
    disp["Jours dep. panne"] = disp["Jours dep. panne"].apply(fmt_days)

    st.dataframe(
        disp.sort_values("Score risque", ascending=False),
        width="stretch",
        hide_index=True,
    )
