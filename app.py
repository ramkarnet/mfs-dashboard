# app.py
# RAMKAR MFS v2.6.1 - Mobile-Friendly Dashboard
# Responsive UI + Grafikler + Kompakt Layout

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# -----------------------------
# PAGE CONFIG - Mobile First
# -----------------------------
st.set_page_config(
    page_title="MFS v2.6",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"  # Mobilde kapalı başla
)

# Custom CSS for mobile
st.markdown("""
<style>
    /* Mobile-first responsive */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.5rem !important;
        }
        .stMetric {
            padding: 0.5rem !important;
        }
        h1 {
            font-size: 1.5rem !important;
        }
        h2, h3 {
            font-size: 1.2rem !important;
        }
        .stDataFrame {
            font-size: 0.8rem !important;
        }
    }
    
    /* Kompakt metrik kartları */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }
    
    /* Sidebar dar */
    section[data-testid="stSidebar"] {
        width: 280px !important;
    }
    
    /* Gauge chart container */
    .gauge-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    /* Hide hamburger on desktop */
    @media (min-width: 769px) {
        [data-testid="collapsedControl"] {
            display: none;
        }
    }
    
    /* Compact tables */
    .compact-table {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

APP_VERSION = "v2.6.1"

# -----------------------------
# CONFIG (Hesaplama - Dokunulmadı)
# -----------------------------
TH = {
    "K1_USDTRY_SHOCK": 0.05,
    "K2_CDS_SPIKE": 100.0,
    "K2_CDS_LEVEL": 700.0,
    "K3_VIX": 35.0,
    "K3_SP500": -0.03,
    "K4_XBANK_DROP": -0.05,
    "K4_XU100_STABLE": -0.01,
    "K5_VOLUME_RATIO": 0.5,
}

HYSTERESIS = {
    "ON_TO_NEUTRAL": 57,
    "NEUTRAL_TO_ON": 63,
    "NEUTRAL_TO_OFF": 37,
    "OFF_TO_NEUTRAL": 43,
    "CONFIRM_WEEKS": 2,
}

DATA_LIMITS = {
    "USDTRY_MAX_WEEKLY": 0.10,
    "USDTRY_WARN_WEEKLY": 0.05,
    "CDS_MAX_WEEKLY": 150,
    "CDS_WARN_WEEKLY": 75,
    "CDS_MIN": 50,
    "CDS_MAX": 1500,
    "VIX_MIN": 8,
    "VIX_MAX": 60,
}

BUDGET_REDUCTIONS = {"K4": 0.25, "K5": 0.15}
W = {"doviz": 0.30, "cds": 0.25, "global": 0.25, "faiz": 0.15, "likidite": 0.05}

BASE_BUDGETS = {
    "ON": (12, 2.5, "NORMAL"),
    "NEUTRAL": (7, 1.5, "SEÇİCİ"),
    "OFF": (4, 1.0, "SINIRLI"),
    "OFF-KILL": (2, 0.5, "YASAK"),
}

STATE_COLORS = {
    "ON": "#00c853",
    "NEUTRAL": "#ffc107", 
    "OFF": "#ff1744",
    "OFF-KILL": "#ad1457"
}

# -----------------------------
# SESSION STATE
# -----------------------------
defaults = {
    "previous_regime": "ON",
    "weeks_in_transition": 0,
    "last_score": 70,
    "show_emergency": False,
    "manual_kill": False,
    # Önceki hafta verileri
    "prev_usdtry": 35.30,
    "prev_cds": 204.0,
    "prev_vix": 17.5,
    "history": []  # Geçmiş kayıtları
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# -----------------------------
# HELPER FUNCTIONS (Hesaplama - Dokunulmadı)
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

@dataclass
class ValidationResult:
    is_valid: bool
    confidence: str
    errors: List[str]
    warnings: List[str]

def validate_data(usdtry_wchg, cds_level, cds_wdelta, vix_last, sp500_wchg):
    errors, warnings = [], []
    
    if abs(usdtry_wchg) > DATA_LIMITS["USDTRY_MAX_WEEKLY"]:
        errors.append(f"USDTRY %{usdtry_wchg*100:.1f} çok yüksek!")
    elif abs(usdtry_wchg) > DATA_LIMITS["USDTRY_WARN_WEEKLY"]:
        warnings.append(f"USDTRY %{usdtry_wchg*100:.1f} yüksek")
    
    if cds_level < DATA_LIMITS["CDS_MIN"] or cds_level > DATA_LIMITS["CDS_MAX"]:
        errors.append(f"CDS {cds_level:.0f} aralık dışı!")
    
    if abs(cds_wdelta) > DATA_LIMITS["CDS_MAX_WEEKLY"]:
        errors.append(f"CDS Δ{cds_wdelta:+.0f}bp çok yüksek!")
    elif abs(cds_wdelta) > DATA_LIMITS["CDS_WARN_WEEKLY"]:
        warnings.append(f"CDS Δ{cds_wdelta:+.0f}bp yüksek")
    
    if vix_last < DATA_LIMITS["VIX_MIN"] or vix_last > DATA_LIMITS["VIX_MAX"]:
        errors.append(f"VIX {vix_last:.1f} aralık dışı!")
    
    if cds_wdelta < -30 and usdtry_wchg > 0.03:
        warnings.append("Tutarsız: CDS↓ TL↓")
    
    confidence = "LOW" if errors else ("MEDIUM" if warnings else "HIGH")
    return ValidationResult(not errors, confidence, errors, warnings)


def get_regime_with_hysteresis(score, prev_regime, weeks_pending, hard_kill):
    if hard_kill:
        return "OFF-KILL", 0, "Kill aktif"
    
    if not prev_regime:
        if score >= 60: return "ON", 0, "İlk değerlendirme"
        elif score >= 40: return "NEUTRAL", 0, "İlk değerlendirme"
        else: return "OFF", 0, "İlk değerlendirme"
    
    target = None
    note = ""
    
    if prev_regime == "ON":
        if score < HYSTERESIS["ON_TO_NEUTRAL"]:
            target, note = "NEUTRAL", f"Skor<{HYSTERESIS['ON_TO_NEUTRAL']}"
        else:
            return "ON", 0, "ON devam"
    elif prev_regime == "NEUTRAL":
        if score > HYSTERESIS["NEUTRAL_TO_ON"]:
            target, note = "ON", f"Skor>{HYSTERESIS['NEUTRAL_TO_ON']}"
        elif score < HYSTERESIS["NEUTRAL_TO_OFF"]:
            target, note = "OFF", f"Skor<{HYSTERESIS['NEUTRAL_TO_OFF']}"
        else:
            return "NEUTRAL", 0, "NEUTRAL devam"
    elif prev_regime == "OFF":
        if score > HYSTERESIS["OFF_TO_NEUTRAL"]:
            target, note = "NEUTRAL", f"Skor>{HYSTERESIS['OFF_TO_NEUTRAL']}"
        else:
            return "OFF", 0, "OFF devam"
    elif prev_regime == "OFF-KILL":
        if score >= 60: return "ON", 0, "Kill kalktı"
        elif score >= 40: return "NEUTRAL", 0, "Kill kalktı"
        else: return "OFF", 0, "Kill kalktı"
    
    if target:
        new_weeks = weeks_pending + 1
        if new_weeks >= HYSTERESIS["CONFIRM_WEEKS"]:
            return target, 0, f"→ {target}"
        return prev_regime, new_weeks, f"⏳ {HYSTERESIS['CONFIRM_WEEKS']-new_weeks}h kaldı"
    
    return prev_regime, 0, ""


def score_doviz(wchg):
    c = abs(wchg)
    if c < 0.005: return 100
    if c < 0.015: return 70
    if c < 0.030: return 40
    if c < 0.050: return 10
    return 0

def score_cds(level, delta):
    if level < 300: base = 100
    elif level < 400: base = 70
    elif level < 500: base = 50
    elif level < 600: base = 30
    elif level < 700: base = 10
    else: base = 0
    if delta > 50: base = max(0, base - 20)
    return base

def score_global(vix, sp):
    if vix < 20: base = 100
    elif vix < 25: base = 80
    elif vix < 30: base = 60
    elif vix < 35: base = 40
    else: base = 20
    if sp < -0.02: base = max(0, base - 20)
    elif sp < -0.01: base = max(0, base - 10)
    return base

def score_likidite(vol):
    if vol >= 1.2: return 100
    if vol >= 0.8: return 70
    if vol >= 0.5: return 40
    return 10


# -----------------------------
# CHARTS
# -----------------------------
def create_gauge_chart(score, regime):
    """Skor gauge chart"""
    color = STATE_COLORS.get(regime, "#666")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 40, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#666"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "#1a1a2e",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(255,23,68,0.3)'},
                {'range': [40, 60], 'color': 'rgba(255,193,7,0.3)'},
                {'range': [60, 100], 'color': 'rgba(0,200,83,0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ccc'}
    )
    return fig


def create_factor_chart(scores_dict):
    """Faktör skorları radar chart"""
    categories = ['Döviz', 'CDS', 'Küresel', 'Faiz', 'Likidite']
    values = [scores_dict['doviz'], scores_dict['cds'], scores_dict['global'], 
              scores_dict['faiz'], scores_dict['likidite']]
    values.append(values[0])  # Close the radar
    categories.append(categories[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0,212,255,0.3)',
        line=dict(color='#00d4ff', width=2),
        name='Skorlar'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color='#888'),
                gridcolor='#333'
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='#ccc'),
                gridcolor='#333'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        height=250,
        margin=dict(l=60, r=60, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def create_killswitch_chart(checks):
    """Kill-switch durumu bar chart"""
    labels = ['K1\nDöviz', 'K2\nCDS', 'K3\nGlobal', 'K4\nBanka', 'K5\nLikid']
    colors = ['#00c853' if v else '#ff1744' for v in checks.values()]
    values = [1 if v else 0.3 for v in checks.values()]
    
    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=['✓' if v else '✗' for v in checks.values()],
        textposition='inside',
        textfont=dict(size=16, color='white')
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=10, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(visible=False, range=[0, 1.2]),
        xaxis=dict(tickfont=dict(size=10, color='#ccc')),
        bargap=0.3
    )
    return fig


def create_weight_pie(scores_dict, weights):
    """Ağırlık katkısı pie chart"""
    labels = ['Döviz', 'CDS', 'Küresel', 'Faiz', 'Likidite']
    contributions = [
        scores_dict['doviz'] * weights['doviz'],
        scores_dict['cds'] * weights['cds'],
        scores_dict['global'] * weights['global'],
        scores_dict['faiz'] * weights['faiz'],
        scores_dict['likidite'] * weights['likidite']
    ]
    
    fig = go.Figure(go.Pie(
        labels=labels,
        values=contributions,
        hole=0.6,
        marker=dict(colors=['#00d4ff', '#00c853', '#ffc107', '#ff6b6b', '#a855f7']),
        textinfo='label+percent',
        textfont=dict(size=10, color='white'),
        insidetextorientation='horizontal'
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        annotations=[dict(text=f'{sum(contributions):.0f}', x=0.5, y=0.5, 
                         font=dict(size=24, color='white'), showarrow=False)]
    )
    return fig


# -----------------------------
# MAIN UI
# -----------------------------
# Header - Kompakt
col_title, col_emergency = st.columns([3, 1])
with col_title:
    st.markdown(f"## 🎯 MFS {APP_VERSION}")
with col_emergency:
    if st.button("🚨 ACİL", type="primary", use_container_width=True):
        st.session_state.show_emergency = not st.session_state.show_emergency

# -----------------------------
# SIDEBAR - Veri Girişi
# -----------------------------
with st.sidebar:
    st.markdown("### 📊 Veri Girişi")
    
    # Önceki Hafta
    st.markdown("##### 📅 Önceki Hafta")
    prev_regime = st.selectbox(
        "Rejim",
        ["ON", "NEUTRAL", "OFF", "OFF-KILL"],
        index=["ON", "NEUTRAL", "OFF", "OFF-KILL"].index(st.session_state.previous_regime)
    )
    weeks_pending = st.number_input("Bekleyen Geçiş", 0, 5, st.session_state.weeks_in_transition)
    
    st.markdown("---")
    
    # Veriler - Kompakt 2 sütun
    st.markdown("##### 💵 Döviz")
    c1, c2 = st.columns(2)
    with c1:
        usdtry_price = st.number_input("USD/TRY", value=st.session_state.prev_usdtry, step=0.1, format="%.2f")
    with c2:
        usdtry_wchg_pct = st.number_input("Δ%", value=0.8, step=0.1, format="%.1f")
    usdtry_wchg = usdtry_wchg_pct / 100
    
    st.markdown("##### 📈 CDS")
    c1, c2 = st.columns(2)
    with c1:
        cds_level = st.number_input("Seviye", value=st.session_state.prev_cds, step=5.0, format="%.0f")
    with c2:
        cds_wdelta = st.number_input("Δbp", value=0.0, step=5.0, format="%.0f")
    
    st.markdown("##### 🌍 Küresel")
    c1, c2 = st.columns(2)
    with c1:
        vix_last = st.number_input("VIX", value=st.session_state.prev_vix, step=0.5, format="%.1f")
    with c2:
        sp500_wchg_pct = st.number_input("S&P%", value=1.0, step=0.5, format="%.1f")
    sp500_wchg = sp500_wchg_pct / 100
    
    st.markdown("##### 🏦 BIST")
    c1, c2 = st.columns(2)
    with c1:
        xu100_wchg_pct = st.number_input("XU100%", value=2.0, step=0.5, format="%.1f")
    with c2:
        xbank_wchg_pct = st.number_input("XBANK%", value=2.5, step=0.5, format="%.1f")
    xu100_wchg = xu100_wchg_pct / 100
    xbank_wchg = xbank_wchg_pct / 100
    
    st.markdown("##### 💧 Diğer")
    c1, c2 = st.columns(2)
    with c1:
        volume_ratio = st.number_input("Hacim", value=1.0, step=0.1, format="%.1f")
    with c2:
        faiz_score = st.number_input("Faiz", value=60, step=5, min_value=0, max_value=100)
    
    st.markdown("---")
    
    # Kaydet butonu
    if st.button("💾 Bu Haftayı Kaydet", use_container_width=True):
        st.session_state.prev_usdtry = usdtry_price
        st.session_state.prev_cds = cds_level
        st.session_state.prev_vix = vix_last
        st.toast("✅ Veriler kaydedildi!")


# -----------------------------
# CALCULATIONS (Dokunulmadı)
# -----------------------------
validation = validate_data(usdtry_wchg, cds_level, cds_wdelta, vix_last, sp500_wchg)

k1_ok = usdtry_wchg < TH["K1_USDTRY_SHOCK"]
k2_ok = (cds_level < TH["K2_CDS_LEVEL"]) and (cds_wdelta < TH["K2_CDS_SPIKE"])
k3_ok = not ((vix_last > TH["K3_VIX"]) and (sp500_wchg <= TH["K3_SP500"]))
k4_ok = not ((xbank_wchg <= TH["K4_XBANK_DROP"]) and (xu100_wchg > TH["K4_XU100_STABLE"]))
k5_ok = volume_ratio >= TH["K5_VOLUME_RATIO"]

checks = {"K1": k1_ok, "K2": k2_ok, "K3": k3_ok, "K4": k4_ok, "K5": k5_ok}
hard_kill = (not k1_ok) or (not k2_ok) or (not k3_ok) or st.session_state.manual_kill

soft_reduction = 0.0
if not k4_ok: soft_reduction += BUDGET_REDUCTIONS["K4"]
if not k5_ok: soft_reduction += BUDGET_REDUCTIONS["K5"]
soft_reduction = clamp(soft_reduction, 0.0, 0.5)

scores = {
    "doviz": score_doviz(usdtry_wchg),
    "cds": score_cds(cds_level, cds_wdelta),
    "global": score_global(vix_last, sp500_wchg),
    "faiz": faiz_score,
    "likidite": score_likidite(volume_ratio),
}

total = int(round(
    scores["doviz"] * W["doviz"] +
    scores["cds"] * W["cds"] +
    scores["global"] * W["global"] +
    scores["faiz"] * W["faiz"] +
    scores["likidite"] * W["likidite"]
))

regime, new_weeks, transition_note = get_regime_with_hysteresis(
    total, prev_regime, weeks_pending, hard_kill
)

# Session güncelle
st.session_state.previous_regime = regime
st.session_state.weeks_in_transition = new_weeks
st.session_state.last_score = total

base_pos, base_risk, base_entry = BASE_BUDGETS[regime]
if soft_reduction > 0:
    adj_pos = max(2, int(math.floor(base_pos * (1 - soft_reduction))))
    adj_risk = round(base_risk * (1 - soft_reduction), 1)
else:
    adj_pos, adj_risk = base_pos, base_risk


# -----------------------------
# ACİL DURUM PANELİ
# -----------------------------
if st.session_state.show_emergency:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff1744, #ad1457); 
                padding: 15px; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0;">🚨 ACİL DURUM PROTOKOLÜ</h3>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Olay Kategorisi**")
        cat = st.radio(
            "Seç:",
            ["A - Savaş/Darbe/Deprem", "B - Siyasi Kriz", "C - Faiz/Enflasyon"],
            label_visibility="collapsed"
        )
    
    with c2:
        st.markdown("**Durum Kontrolü**")
        tier1 = st.checkbox("Tier 1 teyit var")
        duygu = st.checkbox("Panik/titreme hissediyorum")
    
    with c3:
        st.markdown("**Aksiyon**")
        if duygu:
            st.error("💀 %50 KÜÇÜL HEMEN")
        elif "A -" in cat:
            st.error("🔴 %80 NAKİT")
        elif "B -" in cat:
            st.warning("🟡 48 SAAT BEKLE")
        else:
            st.success("🟢 MFS GÜNCELLE")
        
        if st.button("🔴 MANUEL KILL", type="secondary"):
            st.session_state.manual_kill = True
            st.rerun()
    
    if st.session_state.manual_kill:
        if st.button("🟢 Kill'i Kaldır"):
            st.session_state.manual_kill = False
            st.rerun()
    
    st.markdown("---")


# -----------------------------
# MAIN DASHBOARD
# -----------------------------
# Row 1: Ana Metrikler
st.markdown("### 📊 Durum")

c1, c2, c3, c4 = st.columns(4)

with c1:
    color = STATE_COLORS[regime]
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}22, {color}11);
                border: 2px solid {color}; border-radius: 10px; 
                padding: 15px; text-align: center;">
        <div style="font-size: 0.9rem; color: #888;">REJİM</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: {color};">{regime}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    score_color = "#00c853" if total >= 60 else "#ffc107" if total >= 40 else "#ff1744"
    st.markdown(f"""
    <div style="background: #1a1a2e; border: 2px solid {score_color}; 
                border-radius: 10px; padding: 15px; text-align: center;">
        <div style="font-size: 0.9rem; color: #888;">SKOR</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: {score_color};">{total}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div style="background: #1a1a2e; border: 2px solid #00d4ff; 
                border-radius: 10px; padding: 15px; text-align: center;">
        <div style="font-size: 0.9rem; color: #888;">POZİSYON</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: #00d4ff;">{adj_pos}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div style="background: #1a1a2e; border: 2px solid #00d4ff; 
                border-radius: 10px; padding: 15px; text-align: center;">
        <div style="font-size: 0.9rem; color: #888;">RİSK</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: #00d4ff;">{adj_risk}R</div>
    </div>
    """, unsafe_allow_html=True)

# Data Confidence + Transition Note
if validation.errors:
    st.error(f"⛔ VERİ HATASI: {', '.join(validation.errors)}")
elif validation.warnings:
    st.warning(f"⚠️ {', '.join(validation.warnings)}")

if transition_note and "devam" not in transition_note.lower():
    st.info(f"🔄 {transition_note}")


# Row 2: Grafikler
st.markdown("---")
st.markdown("### 📈 Analiz")

tab1, tab2, tab3 = st.tabs(["🎯 Skor", "📊 Faktörler", "🚨 Kill-Switch"])

with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(create_gauge_chart(total, regime), use_container_width=True)
    with c2:
        st.plotly_chart(create_weight_pie(scores, W), use_container_width=True)
        
with tab2:
    st.plotly_chart(create_factor_chart(scores), use_container_width=True)
    
    # Faktör detayları - kompakt tablo
    factor_data = {
        "Faktör": ["💵 Döviz", "📊 CDS", "🌍 Küresel", "🏛️ Faiz", "💧 Likidite"],
        "Skor": [scores['doviz'], scores['cds'], scores['global'], scores['faiz'], scores['likidite']],
        "Ağırlık": ["30%", "25%", "25%", "15%", "5%"],
        "Katkı": [f"{scores['doviz']*0.30:.1f}", f"{scores['cds']*0.25:.1f}", 
                 f"{scores['global']*0.25:.1f}", f"{scores['faiz']*0.15:.1f}", 
                 f"{scores['likidite']*0.05:.1f}"]
    }
    st.dataframe(pd.DataFrame(factor_data), hide_index=True, use_container_width=True)

with tab3:
    st.plotly_chart(create_killswitch_chart(checks), use_container_width=True)
    
    # Kill-switch durumu
    status_text = []
    if not k1_ok: status_text.append("❌ K1: Döviz şoku")
    if not k2_ok: status_text.append("❌ K2: CDS krizi")
    if not k3_ok: status_text.append("❌ K3: Küresel panik")
    if not k4_ok: status_text.append("⚠️ K4: Banka ayrışması")
    if not k5_ok: status_text.append("⚠️ K5: Düşük likidite")
    
    if status_text:
        for s in status_text:
            if s.startswith("❌"):
                st.error(s)
            else:
                st.warning(s)
    else:
        st.success("✅ Tüm kontroller OK")


# Row 3: Karar ve Özet
st.markdown("---")

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("### 🎯 Haftalık Karar")
    
    if regime == "ON":
        st.success(f"""
        **🟢 YEŞİL IŞIK**
        
        • Makro ortam uygun
        • Max **{adj_pos}** pozisyon
        • Max **{adj_risk}R** risk
        • RAMKAR sinyallerini değerlendir
        """)
    elif regime == "NEUTRAL":
        st.warning(f"""
        **🟡 DİKKATLİ OL**
        
        • Makro ortam karışık
        • Max **{adj_pos}** pozisyon
        • Max **{adj_risk}R** risk
        • Sadece A kalite sinyaller
        """)
    elif regime == "OFF":
        st.error(f"""
        **🔴 RİSK YÜKSEK**
        
        • Makro ortam olumsuz
        • Max **{adj_pos}** pozisyon
        • Max **{adj_risk}R** risk
        • Çok sınırlı işlem
        """)
    else:
        st.error(f"""
        **💀 SİSTEM KİLİTLİ**
        
        • YENİ İŞLEM YAPMA
        • Pozisyonları koru
        • Piyasa sakinleşene kadar bekle
        """)

with c2:
    st.markdown("### 📋 Gelecek Hafta")
    
    st.info(f"""
    **Kaydet:**
    - Rejim: **{regime}**
    - Skor: **{total}**
    - Bekleyen: **{new_weeks}** hafta
    
    Gelecek hafta sidebar'dan bu değerleri gir.
    """)
    
    st.markdown("**Veri Özeti:**")
    st.markdown(f"""
    | | Değer |
    |---|---|
    | USD/TRY | {usdtry_price:.2f} ({usdtry_wchg*100:+.1f}%) |
    | CDS | {cds_level:.0f} bp (Δ{cds_wdelta:+.0f}) |
    | VIX | {vix_last:.1f} |
    | S&P | {sp500_wchg*100:+.1f}% |
    """)


# Footer - Kompakt
st.markdown("---")
st.caption(f"MFS {APP_VERSION} • {datetime.now().strftime('%Y-%m-%d %H:%M')} • Yatırım tavsiyesi değildir")
