# Predictive Maintenance Pro

Tableau de bord interactif de maintenance prédictive industrielle développé avec **Streamlit**. L'application analyse l'historique de maintenance d'un parc de 80 équipements (2020–2024) et utilise un modèle **XGBoost** pour scorer le risque de panne et planifier automatiquement les interventions.

---

## Fonctionnalités

| Onglet | Description |
|---|---|
| **Vue d'ensemble** | KPIs globaux du parc, répartition CM/PM, évolution temporelle |
| **Santé des équipements** | Score de risque composite (0–100) par équipement, niveaux Critique / Modéré / Faible |
| **Modèle prédictif** | Performances XGBoost (AUC, accuracy, CV 5-folds), importance des features |
| **Planification** | Calendrier prévisionnel des prochaines pannes et PM recommandées |
| **Tendances & Rapports** | Analyses temporelles, downtime, MTBF, export des données |

---

## Architecture

```
predictive-maintenance/
├── app.py                         # Point d'entrée Streamlit
├── requirements.txt
├── config/
│   └── settings.py                # Constantes, palettes, seuils de risque, features ML
├── modules/
│   ├── data_loader.py             # Chargement & mise en cache des fichiers Excel
│   ├── feature_engineering.py    # Statistiques par équipement, encodage, features ML
│   ├── ml_model.py                # Entraînement XGBoost, score de risque, planification
│   ├── charts.py                  # Graphiques Plotly réutilisables
│   ├── filters.py                 # Composants de filtrage Streamlit
│   └── styles.py                  # CSS injecté, hero banner
├── pages/
│   ├── overview.py
│   ├── equipment_health.py
│   ├── predictive_model.py
│   ├── planning.py
│   └── reports.py
└── datasets/
    ├── equipment_information.xlsx  # 80 équipements (type, âge, conditions, etc.)
    └── maintenance_operations.xlsx # ~6 300 opérations de maintenance (2020–2024)
```

---

## Modèle ML

Le modèle **XGBClassifier** prédit si un équipement appartient à la moitié haute du parc en fréquence de pannes (cible binaire : `cm_per_year > médiane`).

**22 features** couvrant :
- Caractéristiques équipement : âge, génération technologique, classe de coût
- Conditions d'environnement : température, humidité, poussière, vibration
- Historique de maintenance : pannes totales, MTBF, downtime, compliance PM

**Score de risque composite (0–100)** pondérant 7 indicateurs :

| Indicateur | Poids |
|---|---|
| Probabilité ML | 30 % |
| Fréquence CM annuelle | 20 % |
| Pannes sur 6 derniers mois | 15 % |
| Dépassement MTBF | 10 % |
| Ratio haute criticité | 10 % |
| Downtime cumulé | 10 % |
| Non-conformité PM | 5 % |

Niveaux de risque : **Critique** ≥ 65 · **Modéré** ≥ 38 · **Faible** < 38

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/Aboubacar-kader/predictive-maintenance.git
cd predictive-maintenance

# 2. Créer et activer un environnement virtuel
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`.

---

## Dépendances principales

| Package | Version |
|---|---|
| streamlit | 1.56.0 |
| pandas | 3.0.2 |
| xgboost | 3.2.0 |
| scikit-learn | 1.8.0 |
| plotly | 6.7.0 |
| numpy | 2.4.4 |
| openpyxl | 3.1.5 |

---

## Données

Les deux fichiers Excel dans `datasets/` constituent le jeu de données de démonstration :

- **`equipment_information.xlsx`** — 80 équipements avec leurs caractéristiques (type, âge, conditions opérationnelles, facteur de fiabilité, compétence opérateur…)
- **`maintenance_operations.xlsx`** — ~6 300 opérations de maintenance couvrant janvier 2020 à décembre 2024, avec type d'opération (CM/PM), durée de réparation et criticité

---

## Auteur

**Aboubacar Kader** — [aboukader687@gmail.com](mailto:aboukader687@gmail.com)
