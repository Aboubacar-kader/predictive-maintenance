"""
modules/ml_model.py
Entraînement du modèle XGBoost, calcul du score de risque composite (0-100)
et estimation du calendrier de maintenance prévisionnelle.
"""

from datetime import timedelta
from typing import TypedDict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

from config.settings import FEAT_LABELS, FEATURE_COLS, REF_DATE, RISK_THRESHOLDS


# ─────────────────────────────────────────────────────────────
# TYPE DE RETOUR DU MODÈLE
# ─────────────────────────────────────────────────────────────

class ModelResult(TypedDict):
    model:    XGBClassifier
    proba:    np.ndarray        # probabilités sur tout l'ensemble
    fi:       pd.Series         # feature importances (index = labels français)
    cv_mean:  float
    cv_std:   float
    test_auc: float
    test_acc: float


# ─────────────────────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Entraînement du modèle XGBoost…")
def build_model(X: pd.DataFrame, y: pd.Series) -> ModelResult:
    """
    Entraîne un XGBClassifier sur (X, y), évalue ses performances
    par train/test split et cross-validation 5 folds, puis retourne
    les probabilités pour l'ensemble complet.

    La mise en cache évite de ré-entraîner à chaque interaction Streamlit.
    """
    X_clean = X.replace([np.inf, -np.inf], 9_999).fillna(0)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_clean, y, test_size=0.25, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)

    # ── Évaluation ────────────────────────────────────────
    cv_auc   = cross_val_score(model, X_clean, y, cv=5, scoring="roc_auc")
    y_pr     = model.predict_proba(X_te)[:, 1]
    test_auc = roc_auc_score(y_te, y_pr) if y_te.nunique() > 1 else 0.5
    test_acc = accuracy_score(y_te, model.predict(X_te))

    # ── Probabilités sur tout le dataset ─────────────────
    all_proba = model.predict_proba(X_clean)[:, 1]

    # ── Feature importance avec labels français ───────────
    fi = (
        pd.Series(model.feature_importances_, index=FEATURE_COLS)
        .rename(index=FEAT_LABELS)
        .sort_values()
    )

    return ModelResult(
        model    = model,
        proba    = all_proba,
        fi       = fi,
        cv_mean  = float(cv_auc.mean()),
        cv_std   = float(cv_auc.std()),
        test_auc = float(test_auc),
        test_acc = float(test_acc),
    )


# ─────────────────────────────────────────────────────────────
# SCORE DE RISQUE COMPOSITE
# ─────────────────────────────────────────────────────────────

def _normalize(s: pd.Series, higher_worse: bool = True) -> pd.Series:
    """Min-max normalisation ; higher_worse=False inverse la direction."""
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    n = (s - s.min()) / rng
    return n if higher_worse else 1 - n


def compute_risk_scores(
    eq_stats: pd.DataFrame,
    ml_proba: np.ndarray,
) -> pd.Series:
    """
    Calcule un score de risque composite (0-100) combinant :
      30% probabilité ML
      20% fréquence CM annuelle
      15% pannes récentes (6 mois)
      10% facteur de dépassement MTBF
      10% ratio haute criticité
      10% downtime cumulé
       5% non-conformité PM
    """
    es = eq_stats.reset_index(drop=True)
    safe_mtbf = es["mtbf_days"].replace(9_999, np.nan)
    safe_mtbf = safe_mtbf.fillna(safe_mtbf.max() * 2 if safe_mtbf.notna().any() else 9_999)

    # Facteur de dépassement : jours depuis dernière panne / MTBF
    overdue = (
        es["days_since_last_cm"].clip(0, 9_999) / safe_mtbf.clip(1)
    ).clip(0, 2) / 2

    score = (
        0.30 * pd.Series(ml_proba, index=es.index)
      + 0.20 * _normalize(es["cm_per_year"])
      + 0.15 * _normalize(es["cm_last_6m"])
      + 0.10 * overdue
      + 0.10 * _normalize(es["high_crit_ratio"])
      + 0.10 * _normalize(es["total_downtime_h"])
      + 0.05 * _normalize(1 - es["pm_compliance_score"])
    )
    return (score * 100).clip(0, 100).round(1)


def assign_risk_level(risk_score: pd.Series) -> pd.Series:
    """Convertit un score numérique en catégorie Critique / Modéré / Faible."""
    return risk_score.apply(
        lambda x: "Critique"
        if x >= RISK_THRESHOLDS["critique"]
        else ("Modéré" if x >= RISK_THRESHOLDS["modere"] else "Faible")
    )


# ─────────────────────────────────────────────────────────────
# PLANIFICATION PRÉVISIONNELLE
# ─────────────────────────────────────────────────────────────

def predict_maintenance_schedule(eq_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Estime, pour chaque équipement :
      - next_failure  : prochaine panne attendue (MTBF à partir de la dernière panne)
      - days_to_fail  : jours jusqu'à cette panne
      - rec_pm        : date recommandée pour la prochaine PM (70 % du MTBF)

    Logique de reprise : si la date estimée est déjà dépassée, ajoute
    des cycles MTBF jusqu'à retomber dans le futur.
    """
    ref  = REF_DATE
    rows: list[dict] = []

    for _, r in eq_stats.iterrows():
        raw_mtbf = r["mtbf_days"]
        # Plancher à 1 jour pour éviter toute boucle infinie
        mtbf    = max(float(raw_mtbf), 1.0) if raw_mtbf < 9_000 else None
        last_cm = r["last_cm_date"]
        last_pm = r["last_pm_date"]

        if last_cm is not None and mtbf:
            # Prochaine panne : avance par cycles entiers
            elapsed = max((ref - pd.Timestamp(last_cm)).days, 0)
            cycles  = int(elapsed / mtbf) + 1
            nxt     = pd.Timestamp(last_cm) + timedelta(days=mtbf * cycles)
            days_to = int((nxt - ref).days)

            # PM à 70 % du MTBF : même approche vectorielle
            pm_step   = mtbf * 0.70
            elapsed_pm = max((ref - pd.Timestamp(last_cm)).days, 0)
            cycles_pm  = int(elapsed_pm / pm_step) + 1
            pm_date    = pd.Timestamp(last_cm) + timedelta(days=pm_step * cycles_pm)

        else:
            # Pas assez de données : valeurs par défaut prudentes
            nxt     = ref + timedelta(days=365)
            days_to = 365
            pm_date = (
                (pd.Timestamp(last_pm) + timedelta(days=90))
                if last_pm is not None
                else ref + timedelta(days=30)
            )
            if pm_date < ref:
                pm_date = ref + timedelta(days=30)

        rows.append(
            dict(next_failure=nxt, days_to_fail=days_to, rec_pm=pm_date)
        )

    return pd.DataFrame(rows, index=eq_stats.index)


# ─────────────────────────────────────────────────────────────
# PIPELINE COMPLET
# ─────────────────────────────────────────────────────────────

def enrich_equipment_stats(
    eq_stats: pd.DataFrame,
    ml_proba: np.ndarray,
) -> pd.DataFrame:
    """
    Ajoute à eq_stats les colonnes issues du moteur de risque et du
    planificateur prévisionnels :
      risk_score, risk_level, ml_prob,
      next_failure, days_to_fail, rec_pm
    """
    df = eq_stats.copy().reset_index(drop=True)

    risk           = compute_risk_scores(df, ml_proba)
    df["risk_score"]  = risk.values
    df["risk_level"]  = assign_risk_level(risk).values
    df["ml_prob"]     = ml_proba

    sched = predict_maintenance_schedule(df)
    df["next_failure"] = sched["next_failure"].values
    df["days_to_fail"] = sched["days_to_fail"].values
    df["rec_pm"]       = sched["rec_pm"].values

    return df
