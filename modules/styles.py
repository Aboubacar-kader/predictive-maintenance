"""
modules/styles.py
Injection du CSS global, composants HTML réutilisables (cartes KPI, pills risque, etc.)
et fonctions utilitaires d'affichage.
"""

import html as _html

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import COLORS

# ─────────────────────────────────────────────────────────────
# FEUILLE DE STYLE GLOBALE
# ─────────────────────────────────────────────────────────────
_CSS = f"""
<style>
  /* ── Global ─────────────────────────────────────────── */
  [data-testid="stAppViewContainer"] {{ background: {COLORS['bg']}; }}

  /* ── Sidebar ─────────────────────────────────────────── */
  [data-testid="stSidebar"]   {{ background: {COLORS['primary_dk']}; }}
  [data-testid="stSidebar"] * {{ color: #e8eaf6 !important; }}
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2  {{ color: #ffffff !important; }}
  [data-testid="stSidebar"] hr  {{ border-color: rgba(255,255,255,.2); }}
  [data-testid="stSidebar"] label {{ color: #b0bec5 !important; font-size:.82rem; }}

  /* ── Hero banner ────────────────────────────────────── */
  .hero {{
    background: linear-gradient(135deg,
      {COLORS['primary_dk']} 0%,
      {COLORS['primary']} 50%,
      {COLORS['accent']} 100%);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    color: #fff;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 24px rgba(13,71,161,.35);
    display: flex; align-items: center; gap: 1.5rem;
  }}
  .hero h1 {{ margin:0; font-size:1.9rem; font-weight:700; line-height:1.2; }}
  .hero p  {{ margin:.4rem 0 0; font-size:1rem; opacity:.88; }}
  .hero-icon {{ font-size:3.5rem; }}

  /* ── KPI card ───────────────────────────────────────── */
  .kpi-card {{
    background: {COLORS['card']};
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 10px rgba(0,0,0,.08);
    border-top: 4px solid {COLORS['primary']};
    text-align: center;
  }}
  .kpi-value {{ font-size:2rem; font-weight:700; color:{COLORS['primary']}; }}
  .kpi-label {{ font-size:.82rem; color:{COLORS['muted']}; margin-top:.25rem; }}
  .kpi-card.danger  {{ border-top-color:{COLORS['danger']};  }}
  .kpi-card.warning {{ border-top-color:{COLORS['warning']}; }}
  .kpi-card.success {{ border-top-color:{COLORS['success']}; }}
  .kpi-card.danger  .kpi-value {{ color:{COLORS['danger']};  }}
  .kpi-card.warning .kpi-value {{ color:{COLORS['warning']}; }}
  .kpi-card.success .kpi-value {{ color:{COLORS['success']}; }}

  /* ── Section header ─────────────────────────────────── */
  .sec-head {{
    font-size:1.25rem; font-weight:600; color:{COLORS['primary_dk']};
    padding-bottom:.5rem;
    border-bottom: 2px solid #bbdefb;
    margin-bottom:1.2rem;
  }}

  /* ── Risk pills ─────────────────────────────────────── */
  .pill-crit {{ background:#ffebee; color:#b71c1c;
                padding:3px 12px; border-radius:20px;
                font-weight:600; font-size:.8rem; }}
  .pill-mod  {{ background:#fff3e0; color:#e65100;
                padding:3px 12px; border-radius:20px;
                font-weight:600; font-size:.8rem; }}
  .pill-low  {{ background:#e8f5e9; color:#1b5e20;
                padding:3px 12px; border-radius:20px;
                font-weight:600; font-size:.8rem; }}

  /* ── Recommendation card ────────────────────────────── */
  .reco-card {{
    background: #fafafa;
    border-radius: 8px;
    padding: .9rem 1.1rem;
    margin: .45rem 0;
    border-left: 5px solid {COLORS['primary']};
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }}

  /* ── Tabs ───────────────────────────────────────────── */
  [data-baseweb="tab-list"] {{
    gap:6px; background:#e3f2fd;
    border-radius:10px; padding:4px;
  }}
  [data-baseweb="tab"] {{
    border-radius:7px; padding:8px 18px;
    font-weight:500; color:#546e7a;
  }}
  [aria-selected="true"] {{
    background:{COLORS['primary']} !important;
    color:#fff !important;
  }}

  /* ── Misc ───────────────────────────────────────────── */
  #MainMenu, footer {{ visibility:hidden; }}
  div[data-testid="stMetricValue"] {{ font-size:1.5rem; }}
</style>
"""


def inject_css() -> None:
    """Injecte la feuille de style dans la page Streamlit."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# COMPOSANTS HTML
# ─────────────────────────────────────────────────────────────

def kpi_card(col: st.delta_generator.DeltaGenerator,
             value: str | int | float,
             label: str,
             style: str = "") -> None:
    """Affiche une carte KPI dans la colonne Streamlit donnée."""
    col.markdown(
        f'<div class="kpi-card {style}">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str) -> None:
    """Affiche un en-tête de section avec style."""
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)


def hero_banner(title: str, subtitle: str, icon: str = "⚙️") -> None:
    """Affiche le bandeau héro en haut de page."""
    st.markdown(
        f"""<div class="hero">
          <div class="hero-icon">{icon}</div>
          <div><h1>{title}</h1><p>{subtitle}</p></div>
        </div>""",
        unsafe_allow_html=True,
    )


def reco_card(equipment_id: str,
              equip_type: str,
              risk_score: float,
              body_html: str,
              border_color: str) -> None:
    """
    Affiche une carte de recommandation.
    Les champs issus de données externes (equipment_id, equip_type) sont échappés
    via html.escape() pour prévenir toute injection XSS si la source de données
    venait à changer (API, base SQL, etc.).
    Le paramètre body_html doit contenir uniquement des fragments HTML de confiance
    construits dans pages/planning.py avec des valeurs numériques/dates formatées.
    """
    pill_cls = (
        "pill-crit" if border_color == COLORS["danger"]
        else "pill-mod" if border_color == COLORS["warning"]
        else "pill-low"
    )
    safe_id   = _html.escape(str(equipment_id))
    safe_type = _html.escape(str(equip_type))
    st.markdown(
        f"""<div class="reco-card" style="border-left-color:{border_color}">
        <strong>{safe_id}</strong> — {safe_type} &nbsp;
        <span class="{pill_cls}">Score {risk_score:.0f}/100</span><br>
        {body_html}
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────

def fmt_date(d) -> str:
    """Formate une date en dd/mm/yyyy, retourne '—' si invalide."""
    if d is None:
        return "—"
    if isinstance(d, float) and np.isnan(d):
        return "—"
    try:
        return pd.Timestamp(d).strftime("%d/%m/%Y")
    except Exception:
        return "—"


def fmt_mtbf(v: float) -> str:
    """Affiche le MTBF ou '—' s'il n'est pas calculable."""
    return "—" if v >= 9000 else f"{v:.0f} j"


def fmt_days(v: float | int) -> str:
    """Affiche un nombre de jours ou '—' si non calculable."""
    return "—" if v >= 9000 else str(int(v))
