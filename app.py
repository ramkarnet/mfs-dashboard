# app.py
# RAMKAR MFS v2.6 - Streamlit Dashboard
# Manuel Kill-Switch Protokolü Entegrasyonu

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

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
    page_title="RAMKAR MFS v2.6 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_VERSION = "v2.6"

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

# v2.4: Histerezis Eşikleri
HYSTERESIS = {
    "ON_TO_NEUTRAL": 57,
    "NEUTRAL_TO_ON": 63,
    "NEUTRAL_TO_OFF": 37,
    "OFF_TO_NEUTRAL": 43,
    "CONFIRM_WEEKS": 2,
}

# v2.4: Veri Doğrulama Limitleri (v2.6: VIX 60'a güncellendi)
DATA_LIMITS = {
    "USDTRY_MAX_WEEKLY": 0.10,
    "USDTRY_WARN_WEEKLY": 0.05,
    "CDS_MAX_WEEKLY": 150,
    "CDS_WARN_WEEKLY": 75,
    "CDS_MIN": 50,
    "CDS_MAX": 1500,
    "VIX_MIN": 8,
    "VIX_MAX": 60,  # v2.6: 80'den 60'a düşürüldü
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

# v2.6: Manuel Kill-Switch Olay Kategorileri
EVENT_CATEGORIES = {
    "A": {
        "name": "Kategori A - IŞIK HIZI",
        "color": "#ff1744",
        "events": ["Savaş ilanı", "Darbe/darbe girişimi", "Büyük deprem (7+)", "Nükleer/biyolojik tehdit"],
        "action": "MFS'e BAKMA. %80 NAKİT HEMEN.",
        "quarantine": "2 hafta OFF-KILL"
    },
    "B": {
        "name": "Kategori B - 48 SAAT İZLE",
        "color": "#ffc107",
        "events": ["Bakan istifası", "Erken seçim ilanı", "Ambargo haberi", "Büyük banka iflası (global)", "Siyasi kriz"],
        "action": "Yeni alımları DURDUR. Mevcutları hafiflet. 48 saat bekle.",
        "quarantine": "1 hafta gözlem"
    },
    "C": {
        "name": "Kategori C - YAVAŞ SİNDİRİLEN",
        "color": "#00c853",
        "events": ["Faiz kararı (beklenti dışı)", "Enflasyon verisi (şok)", "Küresel satış dalgası", "Kredi notu değişikliği"],
        "action": "MFS skorunu güncelle. Histerezis bekle. %25 küçül.",
        "quarantine": "Histerezis (2 hafta onay)"
    }
}


# -----------------------------
# HELPERS
# -----------------------------
def bar10(score: int) -> str:
    score = int(max(0, min(100, score)))
    filled = score // 10
    return "█" * filled + "░" * (10 - filled)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------
# VERİ DOĞRULAMA (v2.4)
# -----------------------------
@dataclass
class ValidationResult:
    is_valid: bool
    confidence: str
    errors: List[str]
    warnings: List[str]
    
def validate_data(
    usdtry_wchg: float,
    cds_level: float,
    cds_wdelta: float,
    vix_last: float,
    sp500_wchg: float
) -> ValidationResult:
    errors = []
    warnings = []
    
    if abs(usdtry_wchg) > DATA_LIMITS["USDTRY_MAX_WEEKLY"]:
        errors.append(f"⛔ USDTRY haftalık değişim (%{usdtry_wchg*100:.1f}) çok yüksek! Max ±%10 beklenir.")
    elif abs(usdtry_wchg) > DATA_LIMITS["USDTRY_WARN_WEEKLY"]:
        warnings.append(f"⚠️ USDTRY haftalık değişim (%{usdtry_wchg*100:.1f}) yüksek. Şok mu, hata mı kontrol et!")
    
    if cds_level < DATA_LIMITS["CDS_MIN"]:
        errors.append(f"⛔ CDS seviyesi ({cds_level:.0f}) çok düşük! Minimum {DATA_LIMITS['CDS_MIN']} beklenir.")
    elif cds_level > DATA_LIMITS["CDS_MAX"]:
        errors.append(f"⛔ CDS seviyesi ({cds_level:.0f}) çok yüksek! Maximum {DATA_LIMITS['CDS_MAX']} beklenir.")
    
    if abs(cds_wdelta) > DATA_LIMITS["CDS_MAX_WEEKLY"]:
        errors.append(f"⛔ CDS haftalık değişim ({cds_wdelta:+.0f} bp) çok yüksek! Max ±150 bp beklenir.")
    elif abs(cds_wdelta) > DATA_LIMITS["CDS_WARN_WEEKLY"]:
        warnings.append(f"⚠️ CDS haftalık değişim ({cds_wdelta:+.0f} bp) yüksek. Şok mu kontrol et!")
    
    if vix_last < DATA_LIMITS["VIX_MIN"]:
        errors.append(f"⛔ VIX ({vix_last:.1f}) çok düşük! Minimum {DATA_LIMITS['VIX_MIN']} beklenir.")
    elif vix_last > DATA_LIMITS["VIX_MAX"]:
        errors.append(f"⛔ VIX ({vix_last:.1f}) çok yüksek! Maximum {DATA_LIMITS['VIX_MAX']} beklenir. (Kıyamet senaryosu)")
    
    if cds_wdelta < -30 and usdtry_wchg > 0.03:
        warnings.append("🔍 Tutarsızlık: CDS düşerken TL değer kaybediyor. Veriyi kontrol et!")
    
    if cds_wdelta > 50 and usdtry_wchg < -0.02:
        warnings.append("🔍 Tutarsızlık: CDS yükselirken TL değer kazanıyor. Veriyi kontrol et!")
    
    if vix_last > 30 and sp500_wchg > 0.02:
        warnings.append("🔍 Dikkat: VIX yüksek ama S&P pozitif. Piyasa geçiş döneminde olabilir.")
    
    if errors:
        confidence = "LOW"
        is_valid = False
    elif len(warnings) >= 2:
        confidence = "MEDIUM"
        is_valid = True
    elif warnings:
        confidence = "MEDIUM"
        is_valid = True
    else:
        confidence = "HIGH"
        is_valid = True
    
    return ValidationResult(is_valid=is_valid, confidence=confidence, errors=errors, warnings=warnings)


# -----------------------------
# HİSTEREZİS (v2.4)
# -----------------------------
def get_regime_with_hysteresis(
    current_score: int,
    previous_regime: str,
    weeks_in_transition: int,
    hard_kill: bool
) -> Tuple[str, int, str]:
    
    if hard_kill:
        return "OFF-KILL", 0, ""
    
    if previous_regime is None or previous_regime == "":
        if current_score >= 60:
            return "ON", 0, "🆕 İlk değerlendirme"
        elif current_score >= 40:
            return "NEUTRAL", 0, "🆕 İlk değerlendirme"
        else:
            return "OFF", 0, "🆕 İlk değerlendirme"
    
    target_regime = None
    transition_note = ""
    
    if previous_regime == "ON":
        if current_score < HYSTERESIS["ON_TO_NEUTRAL"]:
            target_regime = "NEUTRAL"
            transition_note = f"📉 Skor {HYSTERESIS['ON_TO_NEUTRAL']} altına düştü"
        else:
            return "ON", 0, "✅ ON rejiminde kalınıyor"
    
    elif previous_regime == "NEUTRAL":
        if current_score > HYSTERESIS["NEUTRAL_TO_ON"]:
            target_regime = "ON"
            transition_note = f"📈 Skor {HYSTERESIS['NEUTRAL_TO_ON']} üstüne çıktı"
        elif current_score < HYSTERESIS["NEUTRAL_TO_OFF"]:
            target_regime = "OFF"
            transition_note = f"📉 Skor {HYSTERESIS['NEUTRAL_TO_OFF']} altına düştü"
        else:
            return "NEUTRAL", 0, "✅ NEUTRAL rejiminde kalınıyor"
    
    elif previous_regime == "OFF":
        if current_score > HYSTERESIS["OFF_TO_NEUTRAL"]:
            target_regime = "NEUTRAL"
            transition_note = f"📈 Skor {HYSTERESIS['OFF_TO_NEUTRAL']} üstüne çıktı"
        else:
            return "OFF", 0, "✅ OFF rejiminde kalınıyor"
    
    elif previous_regime == "OFF-KILL":
        if current_score >= 60:
            target_regime = "ON"
        elif current_score >= 40:
            target_regime = "NEUTRAL"
        else:
            target_regime = "OFF"
        transition_note = "🔓 Kill-switch kalktı"
        return target_regime, 0, transition_note
    
    if target_regime:
        new_weeks = weeks_in_transition + 1
        if new_weeks >= HYSTERESIS["CONFIRM_WEEKS"]:
            return target_regime, 0, f"✅ {HYSTERESIS['CONFIRM_WEEKS']} hafta onaylandı → {target_regime}"
        else:
            remaining = HYSTERESIS["CONFIRM_WEEKS"] - new_weeks
            return previous_regime, new_weeks, f"⏳ Geçiş beklemede: {remaining} hafta daha ({transition_note})"
    
    return previous_regime, 0, ""


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
# SESSION STATE
# -----------------------------
if "previous_regime" not in st.session_state:
    st.session_state.previous_regime = None
if "weeks_in_transition" not in st.session_state:
    st.session_state.weeks_in_transition = 0
if "last_score" not in st.session_state:
    st.session_state.last_score = None
if "show_emergency_panel" not in st.session_state:
    st.session_state.show_emergency_panel = False
if "manual_override_active" not in st.session_state:
    st.session_state.manual_override_active = False


# -----------------------------
# UI - SIDEBAR
# -----------------------------
st.sidebar.title("📊 Veri Girişi")
st.sidebar.caption("v2.6 - Manuel Kill-Switch Protokolü")

# v2.6: Acil Durum Butonu
st.sidebar.markdown("---")
if st.sidebar.button("🚨 ACİL DURUM PROTOKOLÜ", type="primary", use_container_width=True):
    st.session_state.show_emergency_panel = not st.session_state.show_emergency_panel

st.sidebar.markdown("---")

# Önceki Hafta Bilgisi
st.sidebar.subheader("📅 Önceki Hafta")
prev_regime_options = ["", "ON", "NEUTRAL", "OFF", "OFF-KILL"]
prev_regime_idx = prev_regime_options.index(st.session_state.previous_regime) if st.session_state.previous_regime in prev_regime_options else 0
previous_regime_input = st.sidebar.selectbox(
    "Önceki Rejim",
    prev_regime_options,
    index=prev_regime_idx,
    help="Geçen haftaki MFS rejimi"
)
weeks_pending = st.sidebar.number_input(
    "Bekleyen Geçiş Haftası",
    min_value=0,
    max_value=5,
    value=st.session_state.weeks_in_transition,
    help="Rejim değişimi için kaç haftadır bekleniyor?"
)

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
validation = validate_data(usdtry_wchg, cds_level, cds_wdelta, vix_last, sp500_wchg)

k1_ok = usdtry_wchg < TH["K1_USDTRY_SHOCK"]
k2_ok = (cds_level < TH["K2_CDS_LEVEL"]) and (cds_wdelta < TH["K2_CDS_SPIKE"])
k3_ok = not ((vix_last > TH["K3_VIX"]) and (sp500_wchg <= TH["K3_SP500"]))
k4_ok = not ((xbank_wchg <= TH["K4_XBANK_DROP"]) and (xu100_wchg > TH["K4_XU100_STABLE"]))
k5_ok = volume_ratio >= TH["K5_VOLUME_RATIO"]

checks = {"K1": k1_ok, "K2": k2_ok, "K3": k3_ok, "K4": k4_ok, "K5": k5_ok}
hard_kill = (not k1_ok) or (not k2_ok) or (not k3_ok)

# v2.6: Manuel Override kontrolü
if st.session_state.manual_override_active:
    hard_kill = True

soft_reduction = 0.0
soft_reasons = []
if not k4_ok:
    soft_reduction += BUDGET_REDUCTIONS["K4"]
    soft_reasons.append("K4: Banka ayrışması")
if not k5_ok:
    soft_reduction += BUDGET_REDUCTIONS["K5"]
    soft_reasons.append("K5: Düşük likidite")
soft_reduction = clamp(soft_reduction, 0.0, 0.5)

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

regime, new_weeks, transition_note = get_regime_with_hysteresis(
    current_score=total,
    previous_regime=previous_regime_input if previous_regime_input else None,
    weeks_in_transition=weeks_pending,
    hard_kill=hard_kill
)

st.session_state.previous_regime = regime
st.session_state.weeks_in_transition = new_weeks
st.session_state.last_score = total

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
st.caption("Manuel Kill-Switch Protokolü Entegrasyonu")

# v2.6: ACİL DURUM PANELİ
if st.session_state.show_emergency_panel:
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(255,23,68,0.3), rgba(173,20,87,0.2));
                padding: 20px; border-radius: 15px; border: 3px solid #ff1744;">
        <h2 style="color: #ff1744; margin: 0;">🚨 ACİL DURUM PROTOKOLÜ</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Hızlı Değerlendirme")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1️⃣ Haber Tier 1 kaynaktan mı?**")
        tier1_confirmed = st.checkbox("Evet, Tier 1 teyit aldım (KAP, TCMB, Bloomberg, Reuters)", key="tier1")
        
        st.markdown("**2️⃣ Olay kategorisi nedir?**")
        event_category = st.radio(
            "Kategori seç:",
            ["Seçilmedi", "A - Işık Hızı (Savaş, Darbe, Deprem)", "B - 48 Saat (Siyasi Kriz)", "C - Yavaş (Faiz, Enflasyon)"],
            key="event_cat"
        )
        
        st.markdown("**3️⃣ Duygu Kotası Kontrolü**")
        duygu_check = st.checkbox("⚠️ Ellerim titriyor / Kalp atışım hızlı / Panik hissediyorum", key="duygu")
    
    with col2:
        st.markdown("**📍 Mevcut Pozisyon Durumu**")
        position_status = st.radio(
            "Pozisyonlarım:",
            ["Pozisyonum yok", "Kârdayım", "Zarardayım", "Başabaştayım"],
            key="pos_status"
        )
        
        st.markdown("**📝 Override Sebebi (zorunlu)**")
        override_reason = st.text_area(
            "Neden manuel müdahale?",
            placeholder="Örn: Cumartesi gecesi savaş haberi geldi, piyasa kapalı...",
            key="override_reason"
        )
    
    # Karar Önerisi
    st.markdown("---")
    st.markdown("### 🎯 SİSTEM ÖNERİSİ")
    
    if duygu_check:
        st.error("""
        ### 💀 DUYGU KOTASI TETİKLENDİ
        
        **HEMEN UYGULA:**
        - Tüm pozisyonları **%50 AZALT**
        - MFS ne derse desin, bu kural öncelikli
        - Sakinleşene kadar yeni işlem **YASAK**
        """)
        if st.button("🔴 MANUEL KILL-SWITCH AKTİFLEŞTİR", type="primary"):
            st.session_state.manual_override_active = True
            st.rerun()
    
    elif "A -" in event_category:
        st.error("""
        ### 🔴 KATEGORİ A - IŞIK HIZI
        
        **HEMEN UYGULA:**
        - **%80 NAKİT** - Sorma, düşünme, yap!
        - MFS'e bakma, bu olay sistemin dışında
        - **2 hafta OFF-KILL** karantina
        - Tier 1 teyit beklemeden hareket et
        """)
        if st.button("🔴 MANUEL KILL-SWITCH AKTİFLEŞTİR", type="primary"):
            st.session_state.manual_override_active = True
            st.rerun()
    
    elif "B -" in event_category:
        st.warning("""
        ### 🟡 KATEGORİ B - 48 SAAT İZLE
        
        **UYGULA:**
        - Yeni alımları **DURDUR**
        - Mevcutları en yakın dirençte **hafiflet**
        - **48 saat** CDS/VIX verisinin oturmasını bekle
        - 1 hafta gözlem süresi
        """)
        
        if position_status == "Kârdayım":
            st.info("💡 **Kârda olduğun için:** Kârın yarısını realize et, kalan yarıyı sıkılaştır.")
        elif position_status == "Zarardayım":
            st.info("💡 **Zararda olduğun için:** Ekleme YAPMA. En yakın desteğe kadar tut, kırılırsa çık.")
        elif position_status == "Başabaştayım":
            st.info("💡 **Başabaş olduğun için:** Bedava çıkış hakkın var. Şimdi çık, sonra tekrar girersin.")
    
    elif "C -" in event_category:
        st.success("""
        ### 🟢 KATEGORİ C - YAVAŞ SİNDİRİLEN
        
        **UYGULA:**
        - MFS skorunu **güncelle** (sidebar'dan)
        - Histerezis onayını bekle (2 hafta)
        - Pozisyon büyüklüğünü **%25 azalt** (önden traşla)
        - Sistem normal çalışıyor, paniğe gerek yok
        """)
    
    elif not tier1_confirmed and event_category != "Seçilmedi":
        st.warning("""
        ### ⚠️ TIER 1 TEYİT BEKLENİYOR
        
        Henüz resmi kaynaklardan teyit almadın.
        - Twitter/WhatsApp haberleri **TEK BAŞINA** karar için yeterli değil
        - Tier 1 teyit gelene kadar **BEKLE**
        - Maksimum 15-30 dakika içinde netleşir
        """)
    
    # Manuel Override Log
    if override_reason and len(override_reason) > 10:
        st.markdown("---")
        st.markdown("### 📋 Override Kaydı")
        st.code(f"""
TARIH: {datetime.now().strftime('%Y-%m-%d %H:%M')}
KATEGORİ: {event_category}
POZİSYON: {position_status}
DUYGU KOTASI: {'EVET' if duygu_check else 'HAYIR'}
TIER1 TEYİT: {'EVET' if tier1_confirmed else 'HAYIR'}
SEBEP: {override_reason}
        """)
        st.caption("Bu kaydı kopyalayıp bir yere kaydet!")
    
    # Manuel Override İptal
    if st.session_state.manual_override_active:
        st.markdown("---")
        st.error("🔴 **MANUEL KILL-SWITCH AKTİF** - Sistem OFF-KILL modunda")
        if st.button("🟢 Manuel Kill-Switch'i Kaldır"):
            st.session_state.manual_override_active = False
            st.rerun()
    
    st.markdown("---")

# DATA CONFIDENCE BANNER
if validation.confidence == "HIGH":
    conf_color = "#00c853"
    conf_icon = "✅"
elif validation.confidence == "MEDIUM":
    conf_color = "#ffc107"
    conf_icon = "⚠️"
else:
    conf_color = "#ff1744"
    conf_icon = "⛔"

# Manuel Override uyarısı
if st.session_state.manual_override_active:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(173,20,87,0.9), rgba(136,14,79,0.9));
                padding: 15px; border-radius: 10px; text-align: center;
                border: 3px solid #ff1744; margin-bottom: 20px;">
        <span style="font-size: 20px; font-weight: 700; color: white;">
            🚨 MANUEL KILL-SWITCH AKTİF - TÜM İŞLEMLER DURDURULDU
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(26,26,46,0.9), rgba(22,33,62,0.9));
            padding: 15px; border-radius: 10px; text-align: center;
            border: 2px solid {conf_color}; margin-bottom: 20px;">
    <span style="font-size: 18px; font-weight: 700; color: {conf_color};">
        {conf_icon} DATA CONFIDENCE: {validation.confidence}
    </span>
</div>
""", unsafe_allow_html=True)

if validation.errors:
    for err in validation.errors:
        st.error(err)
    st.error("⛔ **VERİ HATASI!** Yukarıdaki sorunları düzeltmeden devam etme.")

if validation.warnings:
    for warn in validation.warnings:
        st.warning(warn)

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

# Histerezis Durumu
if transition_note:
    st.markdown("---")
    st.subheader("🔄 Rejim Geçiş Durumu")
    
    if "beklemede" in transition_note.lower() or "⏳" in transition_note:
        st.info(f"""
        **{transition_note}**
        
        Histerezis koruması aktif. Rejim değişimi için {HYSTERESIS['CONFIRM_WEEKS']} hafta üst üste aynı yönde sinyal gerekiyor.
        """)
    elif "onaylandı" in transition_note.lower() or "✅" in transition_note:
        st.success(f"**{transition_note}**")
    else:
        st.info(f"**{transition_note}**")

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

if st.session_state.manual_override_active:
    st.error("🚨 **MANUEL KILL-SWITCH AKTİF!** Acil durum protokolü devrede. Tüm işlemler durduruldu.")
elif hard_kill:
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
    
    st.markdown("---")
    st.markdown("**🔄 Histerezis Eşikleri:**")
    hyst_table = f"""
    | Geçiş | Eşik | Onay Süresi |
    |-------|------|-------------|
    | ON → NEUTRAL | Skor < {HYSTERESIS['ON_TO_NEUTRAL']} | {HYSTERESIS['CONFIRM_WEEKS']} hafta |
    | NEUTRAL → ON | Skor > {HYSTERESIS['NEUTRAL_TO_ON']} | {HYSTERESIS['CONFIRM_WEEKS']} hafta |
    | NEUTRAL → OFF | Skor < {HYSTERESIS['NEUTRAL_TO_OFF']} | {HYSTERESIS['CONFIRM_WEEKS']} hafta |
    | OFF → NEUTRAL | Skor > {HYSTERESIS['OFF_TO_NEUTRAL']} | {HYSTERESIS['CONFIRM_WEEKS']} hafta |
    """
    st.markdown(hyst_table)

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

# Gelecek Hafta
st.markdown("---")
st.subheader("📋 Gelecek Hafta İçin")
st.info(f"""
**Mevcut Durum Kaydet:**
- Rejim: **{regime}**
- Skor: **{total}**
- Bekleyen Geçiş: **{new_weeks}** hafta

Gelecek hafta sidebar'dan "Önceki Rejim" = **{regime}** ve "Bekleyen Geçiş Haftası" = **{new_weeks}** gir.
""")

# Footer
st.markdown("---")
st.caption("⚠️ **Uyarı:** Bu dashboard yatırım tavsiyesi değildir. MFS sadece makro risk filtresidir.")
st.caption(f"🎯 **RAMKAR MFS {APP_VERSION}** | *Manuel Kill-Switch Protokolü Entegrasyonu*")
st.caption("📊 **v2.6 Yenilikler:** Acil Durum Butonu, İnteraktif Checklist, Duygu Kotası, Override Kaydı, VIX eşiği 60'a güncellendi")
