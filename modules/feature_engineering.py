"""
modules/feature_engineering.py
Calcul des statistiques agrégées par équipement et préparation
des features pour le modèle XGBoost.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from config.settings import CAT_ORDERS, FEATURE_COLS, REF_DATE


# ─────────────────────────────────────────────────────────────
# STATISTIQUES PAR ÉQUIPEMENT
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Calcul des statistiques par équipement…")
def compute_equipment_stats(
    df_maint: pd.DataFrame,
    df_equip: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pour chaque équipement, calcule :
      - Compteurs CM / PM (total, par an, 6 derniers mois)
      - Durée moyenne et cumul de réparation
      - MTBF (Mean Time Between Failures, en jours)
      - Jours depuis la dernière panne / PM
      - Ratio haute criticité
      - Dates de la dernière panne et de la dernière PM

    Retourne un DataFrame fusionné avec les informations équipement.
    """
    ref    = REF_DATE
    cut6m  = ref - timedelta(days=180)
    span_y = max((ref - df_maint["equipment_stop_time"].min()).days / 365.25, 1)

    rows: list[dict] = []

    for eid in df_equip["equipment_id"]:
        sub = df_maint[df_maint["equipment_id"] == eid]
        cm  = sub[~sub["is_planned"]]
        pm  = sub[sub["is_planned"]]
        cm6 = cm[cm["equipment_stop_time"] >= cut6m]

        # ── MTBF ──────────────────────────────────────────
        if len(cm) > 1:
            diffs = (
                cm.sort_values("equipment_stop_time")["equipment_stop_time"]
                .diff().dropna()
            )
            mtbf = diffs.dt.total_seconds().mean() / 86_400
        elif len(cm) == 1:
            mtbf = span_y * 365.25          # un seul événement → durée totale
        else:
            mtbf = 9_999.0                  # aucune panne connue

        # ── Dates & délais ────────────────────────────────
        last_cm = cm["equipment_stop_time"].max() if len(cm) > 0 else None
        last_pm = pm["equipment_stop_time"].max() if len(pm) > 0 else None

        days_since_cm = int((ref - last_cm).days) if last_cm is not None else 9_999
        days_since_pm = int((ref - last_pm).days) if last_pm is not None else 9_999

        # ── Criticité ────────────────────────────────────
        high_crit = (
            (cm["operation_criticality"] == "High").mean()
            if len(cm) > 0 else 0.0
        )

        rows.append(
            dict(
                equipment_id       = eid,
                total_cm           = len(cm),
                total_pm           = len(pm),
                cm_per_year        = len(cm) / span_y,
                pm_per_year        = len(pm) / span_y,
                avg_repair_hours   = float(cm["repair_duration_hours"].mean()) if len(cm) > 0 else 0.0,
                total_downtime_h   = float(cm["repair_duration_hours"].sum()),
                high_crit_ratio    = high_crit,
                cm_last_6m         = len(cm6),
                mtbf_days          = mtbf,
                days_since_last_cm = days_since_cm,
                days_since_last_pm = days_since_pm,
                last_cm_date       = last_cm,
                last_pm_date       = last_pm,
            )
        )

    stats_df = pd.DataFrame(rows)
    return df_equip.merge(stats_df, on="equipment_id")


# ─────────────────────────────────────────────────────────────
# ENCODAGE DES VARIABLES CATÉGORIELLES
# ─────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode ordinalement les variables catégorielles définies dans CAT_ORDERS
    et encode le type d'équipement avec LabelEncoder.
    Retourne une copie du DataFrame avec les nouvelles colonnes *_e.
    """
    df = df.copy()

    # Encodage ordinal (l'ordre a un sens métier)
    for col, order in CAT_ORDERS.items():
        mapping = {v: i for i, v in enumerate(order)}
        df[col + "_e"] = df[col].map(mapping).fillna(0).astype(int)

    # Encodage nominal (pas d'ordre particulier)
    le = LabelEncoder()
    df["equipment_type_e"] = le.fit_transform(df["equipment_type"].astype(str))

    return df


# ─────────────────────────────────────────────────────────────
# PRÉPARATION DES FEATURES POUR LE MODÈLE ML
# ─────────────────────────────────────────────────────────────

def prepare_ml_features(
    eq_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Encode les catégorielles, sélectionne FEATURE_COLS et construit la cible binaire :
      target = 1  si cm_per_year > médiane du parc  (haute fréquence de pannes)
               0  sinon

    Retourne (X, y) prêts pour sklearn/XGBoost.
    """
    df = encode_categoricals(eq_stats)

    median_cm = df["cm_per_year"].median()
    y = (df["cm_per_year"] > median_cm).astype(int)

    X = (
        df[FEATURE_COLS]
        .replace([np.inf, -np.inf], 9_999)
        .fillna(0)
    )

    return X, y
