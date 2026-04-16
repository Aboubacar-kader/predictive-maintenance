"""
pages/overview.py
Tab 1 — Vue d'ensemble : KPIs globaux, tendances et distributions.
"""

import pandas as pd
import streamlit as st

from modules.charts import (
    chart_cm_by_type,
    chart_cm_pm_donut,
    chart_monthly_cm,
    chart_operation_heatmap,
)
from modules.styles import kpi_card, section_header


def render(
    df_equip: pd.DataFrame,
    df_maint: pd.DataFrame,
    eq_stats: pd.DataFrame,
) -> None:
    """
    Affiche le tableau de bord Vue d'ensemble.

    Paramètres
    ----------
    df_equip  : équipements filtrés
    df_maint  : opérations de maintenance filtrées
    eq_stats  : statistiques agrégées par équipement (avec risk_score, risk_level…)
    """
    cm_df = df_maint[~df_maint["is_planned"]]
    pm_df = df_maint[df_maint["is_planned"]]

    n_critical   = (eq_stats["risk_level"] == "Critique").sum()
    total_cm     = len(cm_df)
    total_pm     = len(pm_df)
    total_down   = cm_df["repair_duration_hours"].sum()
    avg_mtbf     = eq_stats["mtbf_days"].replace(9_999, None).mean()
    avg_repair   = cm_df["repair_duration_hours"].mean() if len(cm_df) > 0 else 0
    pm_rate      = total_pm / len(df_maint) * 100 if len(df_maint) > 0 else 0
    hi_crit_pct  = (cm_df["operation_criticality"] == "High").mean() * 100 if len(cm_df) > 0 else 0

    # ── KPIs principaux ───────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, len(df_equip),        "Équipements analysés")
    kpi_card(c2, f"{total_cm:,}",      "Pannes correctives",        "danger")
    kpi_card(c3, f"{total_pm:,}",      "Maintenances préventives",  "success")
    kpi_card(c4, f"{total_down:,.0f}h","Downtime total",            "warning")
    kpi_card(c5, n_critical,           "Équipements critiques",     "danger")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tendance mensuelle + donut ─────────────────────────
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(chart_monthly_cm(df_maint), width="stretch")
    with right:
        st.plotly_chart(chart_cm_pm_donut(total_cm, total_pm), width="stretch")

    # ── Distribution par type + heatmap ───────────────────
    left2, right2 = st.columns([2, 3])
    with left2:
        st.plotly_chart(chart_cm_by_type(df_maint, df_equip), width="stretch")
    with right2:
        st.plotly_chart(chart_operation_heatmap(df_maint), width="stretch")

    # ── KPIs secondaires ──────────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, f"{avg_mtbf:.1f} j" if avg_mtbf else "—", "MTBF moyen")
    kpi_card(k2, f"{avg_repair:.1f} h",  "Durée moy. réparation",    "warning")
    kpi_card(k3, f"{pm_rate:.1f}%",      "Taux préventif",           "success")
    kpi_card(k4, f"{hi_crit_pct:.1f}%",  "Part haute criticité",     "danger")
