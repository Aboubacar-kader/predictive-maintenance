"""
modules/data_loader.py
Chargement des fichiers Excel et enrichissement minimal des colonnes.
Le résultat est mis en cache par Streamlit pour ne lire les fichiers qu'une seule fois.
"""

import pandas as pd
import streamlit as st

DATA_PATH = "datasets"


@st.cache_data(show_spinner="Chargement des données…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les deux fichiers Excel et retourne :
      - df_equip : informations sur les équipements (80 lignes)
      - df_maint : historique des opérations de maintenance (6 297 lignes)

    Les colonnes de dates sont converties en datetime.
    Des colonnes dérivées pratiques sont ajoutées à df_maint.
    """
    df_equip = pd.read_excel(f"{DATA_PATH}/equipment_information.xlsx")
    df_maint = pd.read_excel(f"{DATA_PATH}/maintenance_operations.xlsx")

    # ── Conversion des dates ──────────────────────────────
    df_maint["equipment_stop_time"]    = pd.to_datetime(df_maint["equipment_stop_time"])
    df_maint["equipment_restart_time"] = pd.to_datetime(df_maint["equipment_restart_time"])

    # ── Colonnes dérivées ─────────────────────────────────
    df_maint["is_corrective"] = ~df_maint["is_planned"]
    df_maint["year"]          = df_maint["equipment_stop_time"].dt.year
    df_maint["month"]         = df_maint["equipment_stop_time"].dt.month
    df_maint["yearmonth"]     = df_maint["equipment_stop_time"].dt.to_period("M")

    return df_equip, df_maint


def apply_filters(
    df_equip: pd.DataFrame,
    df_maint: pd.DataFrame,
    selected_types: list[str],
    selected_tiers: list[str],
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applique les filtres de la sidebar et retourne les sous-ensembles filtrés.
    Les deux DataFrames restent cohérents (df_maint ne contient que les
    équipements présents dans df_equip filtré).
    """
    df_eq_f = df_equip[
        df_equip["equipment_type"].isin(selected_types) &
        df_equip["cost_tier"].isin(selected_tiers)
    ].copy()

    df_m_f = df_maint[
        df_maint["equipment_id"].isin(df_eq_f["equipment_id"]) &
        (df_maint["equipment_stop_time"] >= date_start) &
        (df_maint["equipment_stop_time"] <= date_end)
    ].copy()

    return df_eq_f, df_m_f
