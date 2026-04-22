"""
pages/planning.py
Tab 4 — Planification : alertes, tableau de priorités, calendrier Gantt
         et recommandations opérationnelles.
"""

import pandas as pd
import streamlit as st

from config.settings import COLORS, PRIORITY_LABELS, REF_DATE
from modules.charts import chart_gantt
from modules.filters import render_filters
from modules.styles import fmt_date, fmt_mtbf, reco_card, section_header


def _build_plan(eq_stats: pd.DataFrame) -> pd.DataFrame:
    """Construit le DataFrame de planification trié par score de risque."""
    plan = eq_stats[[
        "equipment_id", "equipment_type",
        "risk_level", "risk_score",
        "days_to_fail", "next_failure", "rec_pm",
        "last_cm_date", "last_pm_date",
        "mtbf_days", "cm_per_year",
    ]].copy()
    plan["priority"] = plan["risk_level"].map(PRIORITY_LABELS)
    return plan.sort_values("risk_score", ascending=False)


def render(eq_stats: pd.DataFrame) -> None:
    section_header("📅 Planification intelligente de la maintenance")

    f = render_filters(eq_stats, key="planning")
    eq_stats = f["eq_stats"]

    if eq_stats.empty:
        st.warning("Aucun équipement ne correspond aux filtres sélectionnés.")
        return

    st.markdown("---")
    plan = _build_plan(eq_stats)

    # ── Alertes ───────────────────────────────────────────
    crit_eq = plan[plan["risk_level"] == "Critique"]
    soon_eq = plan[(plan["days_to_fail"] <= 30) & (plan["risk_level"] != "Critique")]

    if not crit_eq.empty:
        ids = ", ".join(crit_eq["equipment_id"].tolist())
        st.error(
            f"🚨 **{len(crit_eq)} équipement(s) en risque CRITIQUE** "
            f"— intervention immédiate requise : {ids}"
        )
    if not soon_eq.empty:
        ids = ", ".join(soon_eq["equipment_id"].tolist())
        st.warning(
            f"⚠️ **{len(soon_eq)} équipement(s)** avec panne estimée "
            f"dans les 30 jours : {ids}"
        )

    # ── Tableau de planification ──────────────────────────
    disp = plan.rename(columns={
        "equipment_id":  "Équipement",
        "equipment_type":"Type",
        "risk_level":    "Risque",
        "risk_score":    "Score",
        "days_to_fail":  "Jours avant panne",
        "next_failure":  "Prochaine panne est.",
        "rec_pm":        "PM recommandée",
        "last_cm_date":  "Dernière panne",
        "last_pm_date":  "Dernière PM",
        "mtbf_days":     "MTBF (j)",
        "cm_per_year":   "Pannes/an",
    })

    for col in ["Prochaine panne est.", "PM recommandée", "Dernière panne", "Dernière PM"]:
        disp[col] = disp[col].apply(fmt_date)

    disp["Score"]     = disp["Score"].round(1)
    disp["Pannes/an"] = disp["Pannes/an"].round(1)
    disp["MTBF (j)"]  = disp["MTBF (j)"].apply(fmt_mtbf)

    st.dataframe(
        disp[[
            "priority", "Équipement", "Type", "Risque", "Score",
            "Jours avant panne", "Prochaine panne est.", "PM recommandée",
            "MTBF (j)", "Pannes/an",
        ]],
        width="stretch",
        hide_index=True,
        height=380,
    )

    # ── Calendrier Gantt ─────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📆 Calendrier — 90 prochains jours")

    fig_gantt = chart_gantt(eq_stats)
    if fig_gantt:
        st.plotly_chart(fig_gantt, width="stretch")
    else:
        st.info("Aucune opération prévue dans les 90 prochains jours pour la sélection actuelle.")

    # ── Recommandations ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💡 Recommandations opérationnelles")

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("**🔴 Actions immédiates (Risque Critique)**")
        for _, row in eq_stats[eq_stats["risk_level"] == "Critique"].head(6).iterrows():
            reco_card(
                equipment_id  = row["equipment_id"],
                equip_type    = row["equipment_type"],
                risk_score    = row["risk_score"],
                body_html     = (
                    f"Pannes/an : <strong>{row['cm_per_year']:.1f}</strong> &nbsp;|&nbsp; "
                    f"MTBF : <strong>{fmt_mtbf(row['mtbf_days'])}</strong> &nbsp;|&nbsp; "
                    f"Downtime : <strong>{row['total_downtime_h']:.0f} h</strong><br>"
                    "➡️ Révision complète + inspection immédiate recommandée"
                ),
                border_color  = COLORS["danger"],
            )

    with c_right:
        st.markdown("**🟠 Actions préventives (Risque Modéré)**")
        for _, row in eq_stats[eq_stats["risk_level"] == "Modéré"].head(6).iterrows():
            reco_card(
                equipment_id  = row["equipment_id"],
                equip_type    = row["equipment_type"],
                risk_score    = row["risk_score"],
                body_html     = (
                    f"PM recommandée : <strong>{fmt_date(row['rec_pm'])}</strong><br>"
                    f"Pannes/an : <strong>{row['cm_per_year']:.1f}</strong> &nbsp;|&nbsp; "
                    f"MTBF : <strong>{fmt_mtbf(row['mtbf_days'])}</strong><br>"
                    "➡️ Planifier maintenance préventive dans les 30 jours"
                ),
                border_color  = COLORS["warning"],
            )
