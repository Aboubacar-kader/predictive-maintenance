"""
pages/predictive_model.py
Tab 3 — Modèle prédictif : performance XGBoost, importance des variables
         et probabilité de défaillance par équipement.
"""

import pandas as pd
import streamlit as st

from modules.charts import chart_failure_probability, chart_feature_importance, chart_prob_vs_age
from modules.ml_model import ModelResult
from modules.styles import section_header


def render(eq_stats: pd.DataFrame, model_res: ModelResult) -> None:
    """
    Affiche l'onglet Modèle prédictif.

    Paramètres
    ----------
    eq_stats   : DataFrame enrichi (ml_prob, risk_level, age_years…)
    model_res  : résultat du pipeline d'entraînement XGBoost
    """
    section_header("🤖 Modèle prédictif XGBoost — Probabilité de défaillance")

    info_col, perf_col = st.columns([1, 2])

    # ── Description & métriques ───────────────────────────
    with info_col:
        st.info(
            """
            **Algorithme :** XGBoost Classifier

            **Objectif :** Identifier les équipements à **haute fréquence
            de pannes** (au-dessus de la médiane du parc).

            **22 variables d'entrée :**
            - Caractéristiques physiques & conditions opérationnelles
            - Historique de maintenance (MTBF, CM/an, ratio criticité…)
            - Compliance PM & compétence opérateur

            **Validation :** Cross-validation 5 folds stratifiée
            """
        )
        st.markdown("##### 📊 Performances du modèle")
        m1, m2 = st.columns(2)
        m1.metric("Précision (test)",   f"{model_res['test_acc']:.1%}")
        m2.metric("AUC-ROC (test)",     f"{model_res['test_auc']:.3f}")
        m1.metric("AUC-ROC (CV moy.)", f"{model_res['cv_mean']:.3f}")
        m2.metric("CV Écart-type",      f"±{model_res['cv_std']:.3f}")

    # ── Importance des variables ──────────────────────────
    with perf_col:
        st.plotly_chart(
            chart_feature_importance(model_res["fi"]),
            width="stretch",
        )

    # ── Probabilités par équipement ───────────────────────
    st.markdown("##### Probabilité de défaillance par équipement")
    st.plotly_chart(chart_failure_probability(eq_stats), width="stretch")

    # ── Probabilité vs Âge ────────────────────────────────
    st.markdown("##### Probabilité de défaillance vs Âge de l'équipement")
    st.plotly_chart(chart_prob_vs_age(eq_stats), width="stretch")
