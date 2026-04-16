"""
config/settings.py
Constantes globales, palettes de couleurs, mappings catégoriels et listes de features.
Importé par tous les autres modules pour éviter la duplication.
"""

import pandas as pd

# ─────────────────────────────────────────────────────────────
# DATE DE RÉFÉRENCE
# ─────────────────────────────────────────────────────────────
REF_DATE = pd.Timestamp("2024-12-30")

# ─────────────────────────────────────────────────────────────
# PALETTES & COULEURS
# ─────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#1565c0",
    "primary_dk":"#0d47a1",
    "accent":    "#0288d1",
    "success":   "#2e7d32",
    "warning":   "#e65100",
    "danger":    "#c62828",
    "bg":        "#f0f4f8",
    "card":      "#ffffff",
    "text":      "#263238",
    "muted":     "#607d8b",
}

# Couleur par niveau de risque
RISK_COLOR = {
    "Critique": COLORS["danger"],
    "Modéré":   COLORS["warning"],
    "Faible":   COLORS["success"],
}

# Seuils des niveaux de risque (score 0-100)
RISK_THRESHOLDS = {"critique": 65, "modere": 38}

# ─────────────────────────────────────────────────────────────
# ENCODAGE ORDINAL DES VARIABLES CATÉGORIELLES
# ─────────────────────────────────────────────────────────────
CAT_ORDERS: dict[str, list[str]] = {
    "temperature_condition": ["Low", "Normal", "High"],
    "humidity_condition":    ["Low", "Normal", "High"],
    "dust_level":            ["Clean", "Moderate", "Dusty"],
    "vibration_level":       ["Low", "Normal", "High"],
    "technology_generation": ["Legacy", "Standard", "Advanced"],
    "cost_tier":             ["Low", "Medium", "High"],
}

# ─────────────────────────────────────────────────────────────
# FEATURES DU MODÈLE ML
# ─────────────────────────────────────────────────────────────
FEATURE_COLS: list[str] = [
    "age_years",
    "reliability_factor",
    "utilization_rate",
    "pm_compliance_score",
    "operator_skill_level",
    "temperature_condition_e",
    "humidity_condition_e",
    "dust_level_e",
    "vibration_level_e",
    "cost_tier_e",
    "technology_generation_e",
    "equipment_type_e",
    "total_cm",
    "cm_per_year",
    "total_pm",
    "pm_per_year",
    "avg_repair_hours",
    "total_downtime_h",
    "high_crit_ratio",
    "cm_last_6m",
    "mtbf_days",
    "days_since_last_cm",
]

# Labels français des features (pour les graphiques)
FEAT_LABELS: dict[str, str] = {
    "age_years":               "Âge équipement",
    "reliability_factor":      "Facteur fiabilité",
    "utilization_rate":        "Taux utilisation",
    "pm_compliance_score":     "Compliance PM",
    "operator_skill_level":    "Compétence opérateur",
    "temperature_condition_e": "Condition thermique",
    "humidity_condition_e":    "Condition humidité",
    "dust_level_e":            "Niveau poussière",
    "vibration_level_e":       "Niveau vibration",
    "cost_tier_e":             "Classe de coût",
    "technology_generation_e": "Génération techno",
    "equipment_type_e":        "Type équipement",
    "total_cm":                "Total pannes (CM)",
    "cm_per_year":             "Pannes / an",
    "total_pm":                "Total PM",
    "pm_per_year":             "PM / an",
    "avg_repair_hours":        "Durée réparation moy.",
    "total_downtime_h":        "Downtime total (h)",
    "high_crit_ratio":         "Ratio haute criticité",
    "cm_last_6m":              "Pannes (6 derniers mois)",
    "mtbf_days":               "MTBF (jours)",
    "days_since_last_cm":      "Jours depuis dernière panne",
}

# ─────────────────────────────────────────────────────────────
# LABELS D'AFFICHAGE
# ─────────────────────────────────────────────────────────────
PRIORITY_LABELS: dict[str, str] = {
    "Critique": "🔴 P1 – Urgent",
    "Modéré":   "🟠 P2 – Planifier",
    "Faible":   "🟢 P3 – Surveiller",
}

MAINT_TYPE_COLORS: dict[str, str] = {
    "Corrective (CM)":   COLORS["danger"],
    "Préventive (PM)":   COLORS["success"],
    "Panne prévue":      COLORS["danger"],
    "PM planifiée":      COLORS["primary"],
}
