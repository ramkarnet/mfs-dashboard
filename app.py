# app.py
# RAMKAR MFS v2.3 - Streamlit Dashboard
# Manuel veri girişli versiyon

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="RAMKAR MFS v2.3 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_VERSION = "v2.3"

# Thresholds
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

BUDGET_REDUCTIONS = {"K4": 0.25, "K5": 0.15}

W = {"doviz": 0.30, "cds": 0.25, "global": 0.25, "faiz": 0.15, "likidite": 0.05}

BASE_BUDGETS = {
    "ON": (12, 2.5, "✅ NORMAL"),
    "NEUTRAL": (7, 1.5, "✅ SEÇİCİ"),
    "OFF": (4, 1.0, "⚠️ SINIRLI"),
    "OFF-KILL": (2, 0.5, "❌ YASAK"),
}

STATE_ICON = {"ON": "🟢", "NEUTRAL": "🟡", "OFF": "🔴", "OFF-KILL": "💀"}
STATE_COLOR = {"ON": "green", "NEUTRAL": "orange", "OFF": "red", "OFF-KILL": "purple"}


# -----------------------------
# HELPERS
# -----------------------------
def bar10(score: int) -> str:
    score = int(max(0, min(100, score)))
    filled = score // 10
    return "█" * filled + "░" * (10 - filled)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score_doviz(usdtry_wchg: float) -> Tuple[int, str]:
    c = abs(usdtry_wchg)
    if c < 0.005:
        return 100, "✅ Güvenli"
    if c < 0.015:
        return 70, "⚠️ Normal"
    if c < 0.030:
        return 40, "🟠 Alarm"
    if c < 0.050:
        return 10, "🔴 Tehlike"
    return 0, "💀 Şok"


def score_cds(cds_level: float, cds_wdelta: float) -> Tuple[int, str]:
    if cds_level < 300:
        base, status = 100, "✅ Güvenli"
    elif cds_level < 400:
        base, status = 70, "⚠️ Normal"
    elif cds_level < 500:
        base, status = 50, "🟠 Dikkat"
    elif cds_level < 600:
        base, status = 30, "🔴 Riskli"
    elif cds_level < 700:
        base, status = 10, "💀 Kriz"
    else:
        base, status = 0, "💀 Çöküş"
    
    if cds_wdelta > 50:
        base = max(0, base - 20)
    return base, status


def score_global(vix_last: float, sp500_wchg: float) -> Tuple[int, str]:
    if vix_last < 20:
        base, status = 100, "✅ Sakin"
    elif vix_last < 25:
        base, status = 80, "✅ Normal"
    elif vix_last < 30:
        base, status = 60, "⚠️ Gergin"
    elif vix_last < 35:
        base, status = 40, "🟠 Alarm"
    else:
        base, status = 20, "🔴 Panik"
    
    if sp500_wchg < -0.02:
        base = max(0, base - 20)
    elif sp500_wchg < -0.01:
        base = max(0, base - 10)
    return base, status


def score_likidite(volume_ratio: float) -> Tuple[int, str]:
    if volume_ratio >= 1.2:
        return 100, "✅ Yüksek"
    if volume_ratio >= 0.8:
        return 70, "✅ Normal"
    if volume_ratio >= 0.5:
        return 40, "⚠️ Düşük"
    return 10, "🔴 Kritik"


# -----------------------------
# UI - SIDEBAR
# -----------------------------
st.sidebar.title("📊 Veri Girişi")
st.sidebar.caption("Verileri gir, sonucu gör!")

st.sidebar.markdown("---")
st.sidebar.subheader("💵 Döviz")
usdtry_price = st.sidebar.number_input("USDTRY Fiyat", value=35.30, step=0.10, format="%.2f")
usdtry_wchg_pct = st.sidebar.number_input("USDTRY Haftalık % Değişim", value=0.8, step=0.1, format="%.2f")
usdtry_wchg = usdtry_wchg_pct / 100

st.sidebar.markdown("---")
st.sidebar.subheader("📈 CDS")
cds_level = st.sidebar.number_input("CDS Seviyesi (bp)", value=204.0, step=5.0, format="%.1f")
cds_wdelta = st.sidebar.number_input("CDS Haftalık Δ (bp)", value=0.0, step=5.0, format="%.1f")

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Küresel")
vix_last = st.sidebar.number_input("VIX", value=17.5, step=0.5, format="%.1f")
sp500_wchg_pct = st.sidebar.number_input("S&P500 Haftalık %", value=1.0, step=0.5, format="%.2f")
sp500_wchg = sp500_wchg_pct / 100

st.sidebar.markdown("---")
st.sidebar.subheader("🏦 BIST")
xu100_wchg_pct = st.sidebar.number_input("XU100 Haftalık %", value=2.0, step=0.5, format="%.2f")
xu100_wchg = xu100_wchg_pct / 100
xbank_wchg_pct = st.sidebar.number_input("XBANK Haftalık %", value=2.5, step=0.5, format="%.2f")
xbank_wchg = xbank_wchg_pct / 100

st.sidebar.markdown("---")
st.sidebar.subheader("💧 Likidite")
volume_ratio = st.sidebar.number_input("Hacim Oranı (1.0 = normal)", value=1.0, step=0.1, format="%.1f")

st.sidebar.markdown("---")
st.sidebar.subheader("🏛️ Faiz")
faiz_score = st.sidebar.slider("Faiz Skoru (proxy)", 0, 100, 60)

st.sidebar.markdown("---")
st.sidebar.caption("📅 Veri kaynakları:")
st.sidebar.caption("• investing.com/tr")
st.sidebar.caption("• worldgovernmentbonds.com")
st.sidebar.caption("• tradingview.com")


# -----------------------------
# CALCULATIONS
# -----------------------------
# Kill-Switch Checks
k1_ok = usdtry_wchg < TH["K1_USDTRY_SHOCK"]
k2_ok = (cds_level < TH["K2_CDS_LEVEL"]) and (cds_wdelta < TH["K2_CDS_SPIKE"])
k3_ok = not ((vix_last > TH["K3_VIX"]) and (sp500_wchg <= TH["K3_SP500"]))
k4_ok = not ((xbank_wchg <= TH["K4_XBANK_DROP"]) and (xu100_wchg > TH["K4_XU100_STABLE"]))
k5_ok = volume_ratio >= TH["K5_VOLUME_RATIO"]

checks = {"K1": k1_ok, "K2": k2_ok, "K3": k3_ok, "K4": k4_ok, "K5": k5_ok}
hard_kill = (not k1_ok) or (not k2_ok) or (not k3_ok)

# Soft Veto
soft_reduction = 0.0
soft_reasons = []
if not k4_ok:
    soft_reduction += BUDGET_REDUCTIONS["K4"]
    soft_reasons.append("K4: Banka ayrışması")
if not k5_ok:
    soft_reduction += BUDGET_REDUCTIONS["K5"]
    soft_reasons.append("K5: Düşük likidite")
soft_reduction = clamp(soft_reduction, 0.0, 0.5)

# Factor Scores
doviz_score, doviz_status = score_doviz(usdtry_wchg)
cds_score, cds_status = score_cds(cds_level, cds_wdelta)
glob_score, glob_status = score_global(vix_last, sp500_wchg)
lik_score, lik_status = score_likidite(volume_ratio)

scores = {
    "doviz": doviz_score,
    "cds": cds_score,
    "global": glob_score,
    "faiz": faiz_score,
    "likidite": lik_score,
}

total = int(round(
    scores["doviz"] * W["doviz"] +
    scores["cds"] * W["cds"] +
    scores["global"] * W["global"] +
    scores["faiz"] * W["faiz"] +
    scores["likidite"] * W["likidite"]
))

# Regime
if hard_kill:
    regime = "OFF-KILL"
elif total >= 60:
    regime = "ON"
elif total >= 40:
    regime = "NEUTRAL"
else:
    regime = "OFF"

# Budget
base_pos, base_risk, base_entry = BASE_BUDGETS[regime]
if soft_reduction > 0:
    adj_pos = max(2, int(math.floor(base_pos * (1 - soft_reduction))))
    adj_risk = round(base_risk * (1 - soft_reduction), 1)
    adj_entry = "⚠️ DİKKATLİ" if soft_reduction >= 0.3 else base_entry
else:
    adj_pos, adj_risk, adj_entry = base_pos, base_risk, base_entry


# -----------------------------
# UI - MAIN
# -----------------------------
st.title(f"🎯 RAMKAR MFS {APP_VERSION} — Makro Risk Dashboard")
st.caption("Makro ortam 'risk al' mı 'risk kıs' mı? Tek bakışta gör!")

# KPI Row
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 20px; border-radius: 15px; text-align: center;
                border: 2px solid {STATE_COLOR[regime]};">
        <div style="font-size: 14px; color: #888;">RiskState</div>
        <div style="font-size: 36px; font-weight: 800;">{STATE_ICON[regime]} {regime}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    score_color = "#00c853" if total >= 60 else "#ffc107" if total >= 40 else "#ff1744"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 20px; border-radius: 15px; text-align: center;">
        <div style="font-size: 14px; color: #888;">Toplam Skor</div>
        <div style="font-size: 36px; font-weight: 800; color: {score_color};">{total} / 100</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 20px; border-radius: 15px; text-align: center;">
        <div style="font-size: 14px; color: #888;">Max Pozisyon</div>
        <div style="font-size: 36px; font-weight: 800; color: #00d4ff;">{adj_pos}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 20px; border-radius: 15px; text-align: center;">
        <div style="font-size: 14px; color: #888;">Max Risk</div>
        <div style="font-size: 36px; font-weight: 800; color: #00d4ff;">{adj_risk}R</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Kill-Switch Status
st.subheader("🚨 Kill-Switch Durumu")

ks_cols = st.columns(5)
ks_labels = {"K1": "Döviz", "K2": "CDS", "K3": "Küresel", "K4": "Banka", "K5": "Likidite"}

for i, (k, ok) in enumerate(checks.items()):
    with ks_cols[i]:
        if ok:
            st.success(f"**{k}** {ks_labels[k]}\n\n✅ OK")
        elif k in ["K1", "K2", "K3"]:
            st.error(f"**{k}** {ks_labels[k]}\n\n❌ KILL")
        else:
            st.warning(f"**{k}** {ks_labels[k]}\n\n⚠️ VETO")

if hard_kill:
    st.error("⛔ **Hard Kill aktif!** Yeni giriş yasak. Rejim: OFF-KILL")
elif soft_reduction > 0:
    st.warning(f"⚡ **Soft Veto aktif:** Risk bütçesi -%{int(soft_reduction*100)} | {' | '.join(soft_reasons)}")
else:
    st.success("✅ Tüm kontroller normal. Hard kill yok, soft veto yok.")

st.markdown("---")

# Two Columns
left, right = st.columns([1.2, 0.8])

with left:
    st.subheader("📈 Faktör Skorları")
    
    df_scores = pd.DataFrame([
        {"Faktör": "💵 Döviz (USDTRY)", "Bar": bar10(doviz_score), "Skor": doviz_score, "Durum": doviz_status, "Detay": f"%{usdtry_wchg*100:+.2f} haftalık"},
        {"Faktör": "📊 CDS", "Bar": bar10(cds_score), "Skor": cds_score, "Durum": cds_status, "Detay": f"{cds_level:.0f}bp (Δ{cds_wdelta:+.0f})"},
        {"Faktör": "🌍 Küresel", "Bar": bar10(glob_score), "Skor": glob_score, "Durum": glob_status, "Detay": f"VIX={vix_last:.1f}, S&P={sp500_wchg*100:+.1f}%"},
        {"Faktör": "🏛️ Faiz", "Bar": bar10(faiz_score), "Skor": faiz_score, "Durum": "⚠️ Proxy", "Detay": "TCMB verisi yok"},
        {"Faktör": "💧 Likidite", "Bar": bar10(lik_score), "Skor": lik_score, "Durum": lik_status, "Detay": f"Hacim: {volume_ratio:.1f}x"},
    ])
    
    st.dataframe(df_scores, use_container_width=True, hide_index=True)
    
    # Score breakdown
    st.markdown("**Skor Dağılımı:**")
    breakdown = f"""
    | Faktör | Ağırlık | Skor | Katkı |
    |--------|---------|------|-------|
    | Döviz | %30 | {doviz_score} | {doviz_score * 0.30:.1f} |
    | CDS | %25 | {cds_score} | {cds_score * 0.25:.1f} |
    | Küresel | %25 | {glob_score} | {glob_score * 0.25:.1f} |
    | Faiz | %15 | {faiz_score} | {faiz_score * 0.15:.1f} |
    | Likidite | %5 | {lik_score} | {lik_score * 0.05:.1f} |
    | **TOPLAM** | **%100** | | **{total}** |
    """
    st.markdown(breakdown)

with right:
    st.subheader("🎯 Haftalık Karar")
    
    if regime == "ON":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0,200,83,0.2), rgba(0,230,118,0.1));
                    padding: 25px; border-radius: 15px; border: 2px solid #00c853;">
            <div style="font-size: 28px; font-weight: 800; color: #00c853;">🟢 YEŞİL IŞIK</div>
            <div style="margin-top: 15px; color: #ccc;">
                • Makro ortam <b>uygun</b><br>
                • Max <b>{adj_pos}</b> pozisyon açabilirsin<br>
                • Max <b>{adj_risk}R</b> toplam risk<br>
                • RAMKAR sinyallerini değerlendir
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif regime == "NEUTRAL":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255,193,7,0.2), rgba(255,202,40,0.1));
                    padding: 25px; border-radius: 15px; border: 2px solid #ffc107;">
            <div style="font-size: 28px; font-weight: 800; color: #ffc107;">🟡 DİKKATLİ OL</div>
            <div style="margin-top: 15px; color: #ccc;">
                • Makro ortam <b>karışık</b><br>
                • Max <b>{adj_pos}</b> pozisyon<br>
                • Max <b>{adj_risk}R</b> risk<br>
                • Sadece A kalite sinyaller
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif regime == "OFF":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(255,23,68,0.2), rgba(255,82,82,0.1));
                    padding: 25px; border-radius: 15px; border: 2px solid #ff1744;">
            <div style="font-size: 28px; font-weight: 800; color: #ff1744;">🔴 RİSK YÜKSEK</div>
            <div style="margin-top: 15px; color: #ccc;">
                • Makro ortam <b>olumsuz</b><br>
                • Max <b>{adj_pos}</b> pozisyon<br>
                • Max <b>{adj_risk}R</b> risk<br>
                • Çok sınırlı işlem
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(136,14,79,0.3), rgba(173,20,87,0.2));
                    padding: 25px; border-radius: 15px; border: 2px solid #ad1457;">
            <div style="font-size: 28px; font-weight: 800; color: #e91e63;">💀 SİSTEM KİLİTLİ</div>
            <div style="margin-top: 15px; color: #ccc;">
                • <b>YENİ İŞLEM YAPMA!</b><br>
                • Mevcut pozisyonları koru<br>
                • Max <b>{adj_pos}</b> poz, <b>{adj_risk}R</b> risk<br>
                • Piyasa sakinleşene kadar bekle
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📡 Veri Özeti")
    st.markdown(f"""
    | Veri | Değer |
    |------|-------|
    | USDTRY | {usdtry_price:.2f} ({usdtry_wchg*100:+.2f}%) |
    | CDS | {cds_level:.0f} bp |
    | VIX | {vix_last:.1f} |
    | S&P500 | {sp500_wchg*100:+.2f}% |
    | XU100 | {xu100_wchg*100:+.2f}% |
    | XBANK | {xbank_wchg*100:+.2f}% |
    | Hacim | {volume_ratio:.1f}x |
    """)
    
    st.caption(f"📅 Güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Footer
st.markdown("---")
st.caption("⚠️ **Uyarı:** Bu dashboard yatırım tavsiyesi değildir. MFS sadece makro risk filtresidir; işlem kararı RAMKAR sinyali + risk yönetimi ile verilir.")
st.caption("🎯 **RAMKAR MFS v2.3** | *Para kazandırmak değil, para kaybettirmemek için.*")
