"""
modules/filters.py
Composant de filtrage réutilisable affiché en haut de chaque page.
Le pipeline ML tourne sur la totalité du parc ; les filtres n'affectent que l'affichage.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

_ALL = "— Tous —"
_DATE_MIN = datetime(2020, 1, 1)
_DATE_MAX = datetime(2024, 12, 31)

_FILTER_CSS = """
<style>
/* Bordure individuelle sur chaque widget selectbox */
div[data-testid="stSelectbox"] > div:first-child {
    border: 1.5px solid #c5ceff;
    border-radius: 8px;
    background: #ffffff;
    box-shadow: 0 1px 4px rgba(67, 97, 238, 0.08);
    transition: border-color .2s;
}
div[data-testid="stSelectbox"] > div:first-child:hover {
    border-color: #4361ee;
}
/* Bordure individuelle sur le date_input */
div[data-testid="stDateInput"] > div:first-child {
    border: 1.5px solid #c5ceff;
    border-radius: 8px;
    background: #ffffff;
    box-shadow: 0 1px 4px rgba(67, 97, 238, 0.08);
}
div[data-testid="stDateInput"] > div:first-child:hover {
    border-color: #4361ee;
}
/* Label des filtres */
div[data-testid="stSelectbox"] label,
div[data-testid="stDateInput"] label {
    font-weight: 600;
    font-size: 9.5pt;
    color: #3a3a5c;
}
</style>
"""


def render_filters(
    eq_stats: pd.DataFrame,
    key: str,
    df_maint: pd.DataFrame | None = None,
    df_equip: pd.DataFrame | None = None,
    show_date: bool = False,
) -> dict:
    """
    Affiche des menus déroulants de filtrage stylisés en haut d'une page
    et retourne les DataFrames filtrés dans un dictionnaire.

    Paramètres
    ----------
    eq_stats  : DataFrame enrichi complet (80 équipements)
    key       : préfixe unique par page pour éviter les conflits de clé Streamlit
    df_maint  : historique de maintenance (optionnel)
    df_equip  : informations équipements (optionnel)
    show_date : affiche le filtre de période si True

    Retour
    ------
    dict avec les clés : "eq_stats", "df_maint" (si fourni), "df_equip" (si fourni)
    """
    st.markdown(_FILTER_CSS, unsafe_allow_html=True)

    n_cols = 4 if show_date else 3
    cols = st.columns(n_cols, gap="medium")

    # ── Filtre type ───────────────────────────────────────
    type_opts = [_ALL] + sorted(eq_stats["equipment_type"].unique())
    sel_type = cols[0].selectbox(
        "🔧 Type d'équipement",
        options=type_opts,
        key=f"{key}_type",
    )

    # ── Filtre classe de coût ─────────────────────────────
    sel_tier = cols[1].selectbox(
        "💰 Classe de coût",
        options=[_ALL, "High", "Medium", "Low"],
        key=f"{key}_tier",
    )

    # ── Filtre équipement (dynamique) ─────────────────────
    eq_pre = eq_stats.copy()
    if sel_type != _ALL:
        eq_pre = eq_pre[eq_pre["equipment_type"] == sel_type]
    if sel_tier != _ALL:
        eq_pre = eq_pre[eq_pre["cost_tier"] == sel_tier]

    equip_opts = [_ALL] + sorted(eq_pre["equipment_id"].unique())
    sel_equip = cols[2].selectbox(
        "⚙️ Équipement",
        options=equip_opts,
        key=f"{key}_equip",
    )

    # ── Filtre période (optionnel) ────────────────────────
    d0 = pd.Timestamp(_DATE_MIN)
    d1 = pd.Timestamp(_DATE_MAX)
    if show_date:
        dr = cols[3].date_input(
            "📅 Période",
            value=(_DATE_MIN, _DATE_MAX),
            min_value=_DATE_MIN,
            max_value=_DATE_MAX,
            key=f"{key}_date",
        )
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            d0, d1 = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])

    # ── Compteur de résultat ──────────────────────────────
    eq_f = eq_pre.copy()
    if sel_equip != _ALL:
        eq_f = eq_f[eq_f["equipment_id"] == sel_equip]

    n_total   = len(eq_stats)
    n_filtred = len(eq_f)
    label = (
        f"✅ {n_filtred} / {n_total} équipements sélectionnés"
        if n_filtred < n_total
        else f"✅ Tous les {n_total} équipements"
    )
    st.caption(label)

    # ── Application des filtres ───────────────────────────
    valid_ids = eq_f["equipment_id"]
    result: dict = {"eq_stats": eq_f}

    if df_maint is not None:
        mf = df_maint[df_maint["equipment_id"].isin(valid_ids)].copy()
        if show_date:
            mf = mf[
                (mf["equipment_stop_time"] >= d0) &
                (mf["equipment_stop_time"] <= d1)
            ]
        result["df_maint"] = mf

    if df_equip is not None:
        result["df_equip"] = df_equip[df_equip["equipment_id"].isin(valid_ids)].copy()

    return result
