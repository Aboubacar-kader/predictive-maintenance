"""
modules/charts.py
Bibliothèque de fonctions de visualisation.
Chaque fonction reçoit des données pré-traitées et retourne une figure Plotly.
Aucune logique métier ici : uniquement du rendu graphique.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.settings import COLORS, MAINT_TYPE_COLORS, REF_DATE, RISK_COLOR

# Apparence commune aux graphiques
_LAYOUT = dict(
    paper_bgcolor=COLORS["card"],
    plot_bgcolor="#fafafa",
    font=dict(family="Inter, Arial, sans-serif", color=COLORS["text"]),
    margin=dict(l=10, r=10, t=40, b=10),
)


def _apply(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(**_LAYOUT, **kwargs)
    return fig


# ─────────────────────────────────────────────────────────────
# TAB 1 · VUE D'ENSEMBLE
# ─────────────────────────────────────────────────────────────

def chart_monthly_cm(df_maint: pd.DataFrame) -> go.Figure:
    """Courbe de tendance mensuelle des pannes correctives."""
    cm = df_maint[~df_maint["is_planned"]].copy()
    cm["ym"] = cm["equipment_stop_time"].dt.to_period("M").astype(str)
    cnt = cm.groupby("ym").size().reset_index(name="n")

    fig = px.area(
        cnt, x="ym", y="n",
        title="Évolution mensuelle des pannes correctives",
        labels={"ym": "Mois", "n": "Nombre de pannes"},
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig.update_traces(
        line_color=COLORS["primary"],
        fillcolor=f"rgba(21,101,192,.18)",
    )
    return _apply(fig, showlegend=False, height=340, xaxis_tickangle=-45)


def chart_cm_pm_donut(total_cm: int, total_pm: int) -> go.Figure:
    """Donut chart répartition corrective / préventive."""
    df = pd.DataFrame({
        "Type":  ["Corrective (CM)", "Préventive (PM)"],
        "Count": [total_cm, total_pm],
    })
    fig = px.pie(
        df, values="Count", names="Type",
        title="Répartition CM / PM",
        color_discrete_sequence=[COLORS["danger"], COLORS["success"]],
        hole=0.52,
    )
    return _apply(fig, height=340)


def chart_cm_by_type(df_maint: pd.DataFrame, df_equip: pd.DataFrame) -> go.Figure:
    """Barres horizontales — pannes par type d'équipement."""
    by_type = (
        df_maint[~df_maint["is_planned"]]
        .merge(df_equip[["equipment_id", "equipment_type"]], on="equipment_id")
        .groupby("equipment_type").size()
        .reset_index(name="n")
        .sort_values("n")
    )
    fig = px.bar(
        by_type, x="n", y="equipment_type", orientation="h",
        title="Pannes par type d'équipement",
        color="n", color_continuous_scale="Blues",
        labels={"n": "Nb pannes", "equipment_type": "Type"},
        text="n",
    )
    fig.update_traces(textposition="outside")
    return _apply(fig, coloraxis_showscale=False, height=320)


def chart_operation_heatmap(df_maint: pd.DataFrame) -> go.Figure:
    """Heatmap type d'opération × criticité."""
    heat = (
        df_maint[~df_maint["is_planned"]]
        .groupby(["maintenance_operation", "operation_criticality"])
        .size().reset_index(name="n")
    )
    piv = heat.pivot(
        index="maintenance_operation",
        columns="operation_criticality",
        values="n",
    ).fillna(0)
    fig = px.imshow(
        piv, color_continuous_scale="YlOrRd", aspect="auto",
        title="Heatmap : Type de panne × Criticité",
        labels=dict(color="Nb. opérations"),
    )
    return _apply(fig, height=320)


# ─────────────────────────────────────────────────────────────
# TAB 2 · SANTÉ DES ÉQUIPEMENTS
# ─────────────────────────────────────────────────────────────

def chart_risk_bar(eq_stats: pd.DataFrame) -> go.Figure:
    """Barres horizontales — score de risque par équipement."""
    srt    = eq_stats.sort_values("risk_score", ascending=True)
    colors = srt["risk_level"].map(RISK_COLOR)

    fig = go.Figure(go.Bar(
        x=srt["risk_score"],
        y=srt["equipment_id"],
        orientation="h",
        marker_color=colors,
        text=srt["risk_score"].round(1),
        textposition="outside",
    ))
    fig.add_vline(x=65, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="Critique")
    fig.add_vline(x=38, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="Modéré")
    return _apply(
        fig,
        title="Score de risque par équipement (0–100)",
        xaxis_title="Score de risque",
        height=max(420, len(eq_stats) * 22),
        yaxis=dict(tickfont=dict(size=9)),
        showlegend=False,
    )


def chart_risk_matrix(eq_stats: pd.DataFrame) -> go.Figure:
    """Matrice de risque : MTBF vs fréquence des pannes, taille = downtime."""
    fig = px.scatter(
        eq_stats,
        x="mtbf_days", y="cm_per_year",
        size="total_downtime_h",
        color="risk_level",
        hover_data=["equipment_id", "equipment_type", "risk_score", "age_years"],
        title="Matrice de risque : MTBF vs Fréquence des pannes",
        color_discrete_map=RISK_COLOR,
        labels={
            "mtbf_days":   "MTBF (jours)",
            "cm_per_year": "Pannes / an",
            "risk_level":  "Niveau de risque",
        },
        size_max=55,
    )
    return _apply(fig, height=500)


# ─────────────────────────────────────────────────────────────
# TAB 3 · MODÈLE PRÉDICTIF
# ─────────────────────────────────────────────────────────────

def chart_feature_importance(fi: pd.Series) -> go.Figure:
    """Barres horizontales — importance des variables XGBoost."""
    top = fi.sort_values().tail(15)
    fig = px.bar(
        x=top.values, y=top.index, orientation="h",
        title="Importance des variables (XGBoost)",
        color=top.values, color_continuous_scale="Blues",
        labels={"x": "Importance", "y": "Variable"},
    )
    return _apply(fig, coloraxis_showscale=False, height=480)


def chart_failure_probability(eq_stats: pd.DataFrame) -> go.Figure:
    """Barres — probabilité de défaillance par équipement."""
    df = (
        eq_stats[["equipment_id", "ml_prob", "risk_level"]]
        .sort_values("ml_prob", ascending=False)
        .copy()
    )
    df["pct"] = (df["ml_prob"] * 100).round(1)

    fig = px.bar(
        df, x="equipment_id", y="pct",
        color="risk_level",
        color_discrete_map=RISK_COLOR,
        title="Probabilité de défaillance haute fréquence (%)",
        labels={
            "pct":         "Probabilité (%)",
            "equipment_id":"Équipement",
            "risk_level":  "Niveau de risque",
        },
        text="pct",
    )
    fig.add_hline(y=65, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="Seuil critique")
    fig.add_hline(y=38, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="Seuil modéré")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    return _apply(fig, height=420, xaxis_tickangle=-45)


def chart_prob_vs_age(eq_stats: pd.DataFrame) -> go.Figure:
    """Scatter — probabilité ML vs âge de l'équipement."""
    fig = px.scatter(
        eq_stats,
        x="age_years", y="ml_prob",
        color="risk_level", size="total_downtime_h",
        symbol="equipment_type",
        color_discrete_map=RISK_COLOR,
        title="Probabilité de défaillance vs Âge",
        labels={
            "age_years":  "Âge (ans)",
            "ml_prob":    "Probabilité ML",
            "risk_level": "Risque",
            "equipment_type": "Type",
        },
        hover_data=["equipment_id"],
    )
    return _apply(fig, height=380)


# ─────────────────────────────────────────────────────────────
# TAB 4 · PLANIFICATION
# ─────────────────────────────────────────────────────────────

def chart_gantt(eq_stats: pd.DataFrame) -> go.Figure | None:
    """
    Diagramme de Gantt des 90 prochains jours.
    Retourne None si aucune opération n'est dans la fenêtre.
    """
    horizon = REF_DATE + timedelta(days=90)
    rows: list[dict] = []

    for _, r in eq_stats.iterrows():
        # Panne prévue
        nf = r.get("next_failure")
        if pd.notna(nf) and REF_DATE <= pd.Timestamp(nf) <= horizon:
            dur = max(r["mtbf_days"] * 0.08, 1) if r["mtbf_days"] < 9_000 else 2
            rows.append({
                "Task":  f"{r['equipment_id']} ({r['equipment_type']})",
                "Start": pd.Timestamp(nf) - timedelta(days=1),
                "Finish":pd.Timestamp(nf) + timedelta(days=dur),
                "Type":  "Panne prévue",
            })
        # PM planifiée
        pm = r.get("rec_pm")
        if pd.notna(pm) and REF_DATE <= pd.Timestamp(pm) <= horizon:
            rows.append({
                "Task":  f"{r['equipment_id']} ({r['equipment_type']})",
                "Start": pd.Timestamp(pm),
                "Finish":pd.Timestamp(pm) + timedelta(days=1),
                "Type":  "PM planifiée",
            })

    if not rows:
        return None

    gdf = pd.DataFrame(rows)
    fig = px.timeline(
        gdf, x_start="Start", x_end="Finish", y="Task", color="Type",
        color_discrete_map=MAINT_TYPE_COLORS,
        title="Calendrier de maintenance prévisionnelle — 90 jours",
        labels={"Task": "Équipement", "Type": "Opération"},
    )
    # Plotly ne peut pas calculer _mean(str, str) pour positionner l'annotation
    # → on sépare la ligne verticale et l'annotation
    ref_str = REF_DATE.strftime("%Y-%m-%d")
    fig.add_vline(x=ref_str, line_dash="dot", line_color="black")
    fig.add_annotation(
        x=ref_str, y=1.0, xref="x", yref="paper",
        text="Référence (30/12/2024)", showarrow=False,
        xanchor="left", font=dict(size=10, color="black"),
    )
    fig.update_yaxes(autorange="reversed")
    return _apply(fig, height=max(380, len(gdf) * 28))


# ─────────────────────────────────────────────────────────────
# TAB 5 · TENDANCES & RAPPORTS
# ─────────────────────────────────────────────────────────────

def chart_annual_operations(df_maint: pd.DataFrame) -> go.Figure:
    """Barres groupées CM/PM par année."""
    ann = (
        df_maint
        .groupby(["year", "is_planned"])
        .agg(count=("equipment_id", "count"))
        .reset_index()
    )
    ann["Type"] = ann["is_planned"].map(
        {True: "Préventive (PM)", False: "Corrective (CM)"}
    )
    fig = px.bar(
        ann, x="year", y="count", color="Type", barmode="group",
        title="Opérations de maintenance par année",
        color_discrete_map={
            "Corrective (CM)": COLORS["danger"],
            "Préventive (PM)": COLORS["success"],
        },
        labels={"year": "Année", "count": "Nb opérations", "Type": "Type"},
    )
    return _apply(fig, height=340)


def chart_annual_downtime(df_maint: pd.DataFrame) -> go.Figure:
    """Barres — downtime annuel par année."""
    cm_ann = (
        df_maint[~df_maint["is_planned"]]
        .groupby("year")["repair_duration_hours"].sum()
        .reset_index(name="downtime")
    )
    fig = px.bar(
        cm_ann, x="year", y="downtime",
        title="Downtime annuel (heures)",
        color="downtime", color_continuous_scale="Reds",
        labels={"year": "Année", "downtime": "Downtime (h)"},
    )
    return _apply(fig, height=340, coloraxis_showscale=False)


def chart_mtbf_by_type(eq_stats: pd.DataFrame) -> go.Figure:
    """MTBF moyen par type d'équipement."""
    mtbf_t = (
        eq_stats.groupby("equipment_type")["mtbf_days"]
        .apply(lambda x: x.replace(9_999, np.nan).mean())
        .reset_index(name="avg_mtbf")
        .sort_values("avg_mtbf")
    )
    fig = px.bar(
        mtbf_t, x="avg_mtbf", y="equipment_type", orientation="h",
        title="MTBF moyen par type d'équipement (jours)",
        color="avg_mtbf", color_continuous_scale="Greens",
        labels={"avg_mtbf": "MTBF moyen (j)", "equipment_type": "Type"},
    )
    return _apply(fig, height=320, coloraxis_showscale=False)


def chart_top_failure_causes(df_maint: pd.DataFrame) -> go.Figure:
    """Top 10 causes de pannes correctives."""
    top = (
        df_maint[~df_maint["is_planned"]]
        .groupby("maintenance_operation")
        .agg(count=("equipment_id", "count"),
             avg_dur=("repair_duration_hours", "mean"))
        .reset_index()
        .sort_values("count", ascending=False).head(10)
        .sort_values("count", ascending=True)
    )
    fig = px.bar(
        top, x="count", y="maintenance_operation", orientation="h",
        title="Top 10 causes de pannes",
        color="avg_dur", color_continuous_scale="OrRd",
        labels={
            "count":                "Occurrences",
            "maintenance_operation":"Cause",
            "avg_dur":              "Durée moy. (h)",
        },
        text="count",
    )
    return _apply(fig, height=320)


def chart_pm_ratio_trend(df_maint: pd.DataFrame) -> go.Figure:
    """Courbe du ratio PM mensuel avec ligne de moyenne."""
    mr = (
        df_maint
        .assign(ym=df_maint["equipment_stop_time"].dt.to_period("M").astype(str))
        .groupby(["ym", "is_planned"]).size().reset_index(name="n")
    )
    piv = mr.pivot(index="ym", columns="is_planned", values="n").fillna(0)
    piv.columns = ["CM", "PM"]
    piv["pm_pct"] = piv["PM"] / (piv["CM"] + piv["PM"]) * 100
    piv = piv.reset_index()
    avg = piv["pm_pct"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=piv["ym"], y=piv["pm_pct"],
        mode="lines+markers", fill="tozeroy",
        line=dict(color=COLORS["success"], width=2),
        fillcolor=f"rgba(46,125,50,.12)",
        name="Ratio PM (%)",
    ))
    fig.add_hline(
        y=avg, line_dash="dash", line_color=COLORS["primary"],
        annotation_text=f"Moyenne : {avg:.1f}%",
    )
    return _apply(
        fig,
        title="Ratio maintenance préventive / total (%)",
        xaxis_title="Mois", yaxis_title="% préventive",
        height=320, showlegend=False,
        xaxis_tickangle=-45,
    )


def chart_downtime_top_equipment(
    df_maint: pd.DataFrame, df_equip: pd.DataFrame
) -> go.Figure:
    """Top 15 équipements par downtime cumulé."""
    dt_eq = (
        df_maint[~df_maint["is_planned"]]
        .groupby("equipment_id")["repair_duration_hours"].sum()
        .reset_index(name="downtime")
        .sort_values("downtime", ascending=False).head(15)
        .merge(df_equip[["equipment_id", "equipment_type"]], on="equipment_id")
        .sort_values("downtime")
    )
    fig = px.bar(
        dt_eq, x="downtime", y="equipment_id", orientation="h",
        color="equipment_type",
        title="Top 15 équipements — Downtime cumulé (heures)",
        labels={
            "downtime":      "Downtime (h)",
            "equipment_id":  "Équipement",
            "equipment_type":"Type",
        },
        text="downtime",
    )
    fig.update_traces(texttemplate="%{text:.0f} h", textposition="outside")
    return _apply(fig, height=420)
