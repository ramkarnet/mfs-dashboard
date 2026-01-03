# app.py
# RAMKAR MFS v2.7.2 - TAM GÜVENLİK
# 2 Adımlı Kill + Kilitli Tarama + CSV Kara Kutu

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import os
import csv
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="MFS + RAMKAR",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem !important; }
        h1, h2 { font-size: 1.3rem !important; }
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    section[data-testid="stSidebar"] { width: 280px !important; }
</style>
""", unsafe_allow_html=True)

APP_VERSION = "v2.7.2"
LOG_FILE = "mfs_kara_kutu.csv"

# ================================
# KATILIM ENDEKSİ HİSSELERİ (217)
# ================================
KATILIM_HISSELERI = [
    "ACSEL", "AHSGY", "AKFYE", "AKSA", "AKYHO", "ALBRK", "ALCTL", "ALKA", 
    "ALKIM", "ALKLC", "ALTNY", "ALVES", "ANGEN", "ARASE", "ARDYZ", "ASELS", 
    "ATAKP", "ATATP", "AVPGY", "AYEN", "BAHKM", "BAKAB", "BANVT", "BASGZ",
    "BEGYO", "BERA", "BIENY", "BIMAS", "BINBN", "BINHO", "BMSTL", "BNTAS", 
    "BORSK", "BOSSA", "BRISA", "BRKSN", "BRLSM", "BSOKE", "BURCE", "BURVA", 
    "CANTE", "CATES", "CELHA", "CEMTS", "CEMZY", "CIMSA", "CMBTN", "COSMO",
    "CVKMD", "CWENE", "DAPGM", "DARDL", "DCTTR", "DENGE", "DESPC", "DGATE", 
    "DGNMO", "DMSAS", "DOFER", "DOFRB", "DOGUB", "DYOBY", "EBEBK", "EDATA", 
    "EDIP", "EFOR", "EGEPO", "EGGUB", "EGPRO", "EKGYO", "EKSUN", "ELITE",
    "ENJSA", "EREGL", "ESCOM", "EUPWR", "EYGYO", "FADE", "FONET", "FORMT", 
    "FORTE", "FZLGY", "GEDZA", "GENIL", "GENTS", "GEREL", "GESAN", "GLRMK", 
    "GOKNR", "GOLTS", "GOODY", "GRSEL", "GRTHO", "GUBRF", "GUNDG", "HATSN",
    "HKTM", "HOROZ", "HRKET", "IDGYO", "IHEVA", "IHLAS", "IHLGM", "IHYAY", 
    "IMASM", "INTEM", "ISDMR", "ISSEN", "IZFAS", "IZINV", "JANTS", "KARSN", 
    "KATMR", "KBORU", "KCAER", "KIMMR", "KLSYN", "KNFRT", "KOCMT", "KONKA",
    "KONTR", "KONYA", "KOPOL", "KOTON", "KRDMA", "KRDMB", "KRDMD", "KRGYO", 
    "KRONT", "KRPLS", "KRSTL", "KRVGD", "KTLEV", "KUTPO", "KUYAS", "KZBGY", 
    "LKMNH", "LMKDC", "LOGO", "MAGEN", "MAKIM", "MARBL", "MAVI", "MEDTR",
    "MEKAG", "MERCN", "MNDRS", "MNDTR", "MOBTL", "MPARK", "NETAS", "NTGAZ", 
    "OBAMS", "OBASE", "OFSYM", "ONCSM", "ORGE", "OSTIM", "OZRDN", "OZYSR", 
    "PAGYO", "PARSN", "PASEU", "PENGD", "PENTA", "PETKM", "PETUN", "PKART",
    "PLTUR", "PNLSN", "POLHO", "QUAGR", "RGYAS", "RNPOL", "RODRG", "RUBNS", 
    "SAFKR", "SAMAT", "SANEL", "SANKO", "SARKY", "SAYAS", "SEKUR", "SELEC", 
    "SELVA", "SILVR", "SMART", "SMRTG", "SNGYO", "SNICA", "SOKE", "SRVGY",
    "SUNTK", "SURGY", "SUWEN", "TARKM", "TDGYO", "TEZOL", "TKNSA", "TMSN", 
    "TNZTP", "TUCLK", "TUKAS", "TUPRS", "TUREX", "ULAS", "ULUSE", "USAK", 
    "VAKKO", "VANGD", "VESBE", "VRGYO", "YATAS", "YEOTK", "YUNSA", "ZEDUR", "ZERGY"
]

# ================================
# MFS CONFIG
# ================================
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
    "ON": (12, 2.5),
    "NEUTRAL": (7, 1.5),
    "OFF": (4, 1.0),
    "OFF-KILL": (2, 0.5),
}

STATE_COLORS = {"ON": "#00c853", "NEUTRAL": "#ffc107", "OFF": "#ff1744", "OFF-KILL": "#ad1457"}


# ================================
# KARA KUTU (CSV LOG)
# ================================
def log_to_csv(event_type: str, details: str, mfs_score: int, regime: str):
    """Her önemli olayı CSV'ye kaydet"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Dosya yoksa başlık yaz
    file_exists = os.path.exists(LOG_FILE)
    
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Tarih', 'Olay', 'Detay', 'MFS_Skor', 'Rejim'])
        writer.writerow([timestamp, event_type, details, mfs_score, regime])


def get_log_history() -> pd.DataFrame:
    """Log geçmişini oku"""
    if os.path.exists(LOG_FILE):
        try:
            return pd.read_csv(LOG_FILE)
        except:
            return pd.DataFrame()
    return pd.DataFrame()


# ================================
# RAMKAR TEKNİK FONKSİYONLAR
# ================================
def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stoch_rsi(data, period=14, smooth_k=3, smooth_d=3):
    rsi = calculate_rsi(data, period)
    stoch = ((rsi - rsi.rolling(period).min()) / 
             (rsi.rolling(period).max() - rsi.rolling(period).min())) * 100
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d

def calculate_adx(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    return adx, plus_di, minus_di

def calculate_mfi(data, period=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
    
    return 100 - (100 / (1 + positive_flow / negative_flow))

def calculate_sar(data, acceleration=0.02, maximum=0.2):
    high = data['High'].values
    low = data['Low'].values
    length = len(high)
    
    sar = np.zeros(length)
    af = acceleration
    ep = low[0]
    uptrend = True
    sar[0] = high[0]
    
    for i in range(1, length):
        if uptrend:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
            
            if low[i] < sar[i]:
                uptrend = False
                sar[i] = ep
                ep = low[i]
                af = acceleration
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + acceleration, maximum)
        else:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
            
            if high[i] > sar[i]:
                uptrend = True
                sar[i] = ep
                ep = high[i]
                af = acceleration
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + acceleration, maximum)
    
    return pd.Series(sar, index=data.index)

def calculate_atr_percent(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return (atr / close) * 100


# ================================
# RAMKAR TARAMA
# ================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_xu100_data():
    if not YF_AVAILABLE:
        return None, None, None
    try:
        xu100 = yf.Ticker("XU100.IS")
        data = xu100.history(period="1y", interval="1wk")
        if len(data) < 50:
            return None, None, None
        
        xu100_close = data['Close'].iloc[-1]
        xu100_ema50 = calculate_ema(data, 50).iloc[-1]
        
        return data, xu100_close, xu100_ema50
    except:
        return None, None, None


def ramkar_scan_single(symbol: str, xu100_close, xu100_ema50) -> Tuple[Optional[Dict], bool]:
    """Tek hisse tara. Returns (result, success)"""
    try:
        stock = yf.Ticker(f"{symbol}.IS")
        data = stock.history(period="1y", interval="1wk")
        
        if len(data) < 50:
            return None, False
        
        ema20 = calculate_ema(data, 20)
        k, d = calculate_stoch_rsi(data, 14, 3, 3)
        adx, di_plus, di_minus = calculate_adx(data, 14)
        mfi = calculate_mfi(data, 14)
        vol_avg = data['Volume'].rolling(20).mean()
        sar = calculate_sar(data)
        atr_pct = calculate_atr_percent(data, 14)
        
        last_close = data['Close'].iloc[-1]
        last_ema20 = ema20.iloc[-1]
        last_k = k.iloc[-1] if not pd.isna(k.iloc[-1]) else 0
        last_d = d.iloc[-1] if not pd.isna(d.iloc[-1]) else 0
        last_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        last_di_plus = di_plus.iloc[-1] if not pd.isna(di_plus.iloc[-1]) else 0
        last_di_minus = di_minus.iloc[-1] if not pd.isna(di_minus.iloc[-1]) else 0
        last_volume = data['Volume'].iloc[-1]
        last_vol_avg = vol_avg.iloc[-1] if not pd.isna(vol_avg.iloc[-1]) else 1
        last_sar = sar.iloc[-1] if not pd.isna(sar.iloc[-1]) else last_close
        last_mfi = mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else 50
        last_atr_pct = atr_pct.iloc[-1] if not pd.isna(atr_pct.iloc[-1]) else 5
        
        ema_dist = ((last_close - last_ema20) / last_ema20) * 100
        bist_ok = xu100_close > xu100_ema50 if xu100_close and xu100_ema50 else False
        
        trend_ok = (last_close > last_ema20) and (last_k > last_d)
        power_ok = (last_adx >= 28) and (last_di_plus > last_di_minus)
        vol_ok = last_volume >= (last_vol_avg * 1.2) if last_vol_avg > 0 else False
        sar_ok = last_close > last_sar
        dist_ok = (ema_dist <= 30.0) and (ema_dist >= -2.0)
        
        score = sum([trend_ok, power_ok, vol_ok, sar_ok, dist_ok, bist_ok])
        score_pct = round((score / 6) * 100)
        
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else last_close
        prev_ema20 = ema20.iloc[-2] if len(ema20) > 1 else last_ema20
        stop_onay = (last_close < last_ema20) and (prev_close < prev_ema20)
        
        if stop_onay:
            status, status_color, status_icon = "STOP", "#FF1744", "🛑"
        elif ema_dist > 32:
            status, status_color, status_icon = "KAR AL", "#FFB300", "⚠️"
        elif last_atr_pct > 12:
            status, status_color, status_icon = "VOLATİL", "#5D4037", "🌀"
        elif score == 6:
            status, status_color, status_icon = "RADAR", "#00E676", "🚀"
        elif score >= 4:
            status, status_color, status_icon = "İZLEME", "#37474F", "⚡"
        else:
            status, status_color, status_icon = "PUSU", "#666666", "💤"
        
        return {
            'symbol': symbol,
            'price': round(last_close, 2),
            'score': score,
            'score_pct': score_pct,
            'status': status,
            'status_color': status_color,
            'status_icon': status_icon,
            'trend_ok': trend_ok,
            'power_ok': power_ok,
            'vol_ok': vol_ok,
            'sar_ok': sar_ok,
            'dist_ok': dist_ok,
            'bist_ok': bist_ok,
            'adx': round(last_adx, 1),
            'ema_dist': round(ema_dist, 1),
            'mfi': round(last_mfi, 0),
            'atr_pct': round(last_atr_pct, 1),
            'stop_onay': stop_onay
        }, True
        
    except Exception as e:
        return None, False


def run_full_scan(progress_callback=None):
    """Tüm hisseleri tara + hata sayacı"""
    if not YF_AVAILABLE:
        return [], None, None, 0, 0
    
    _, xu100_close, xu100_ema50 = get_xu100_data()
    
    results = []
    success_count = 0
    error_count = 0
    total = len(KATILIM_HISSELERI)
    
    for i, symbol in enumerate(KATILIM_HISSELERI):
        if progress_callback:
            progress_callback((i + 1) / total, symbol)
        
        result, success = ramkar_scan_single(symbol, xu100_close, xu100_ema50)
        
        if success and result:
            results.append(result)
            success_count += 1
        else:
            error_count += 1
    
    results.sort(key=lambda x: (x['score'], x['score_pct'], -x['atr_pct']), reverse=True)
    
    return results, xu100_close, xu100_ema50, success_count, error_count


# ================================
# MFS FONKSİYONLARI
# ================================
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
    
    if vix_last < DATA_LIMITS["VIX_MIN"] or vix_last > DATA_LIMITS["VIX_MAX"]:
        errors.append(f"VIX {vix_last:.1f} aralık dışı!")
    
    confidence = "LOW" if errors else ("MEDIUM" if warnings else "HIGH")
    return ValidationResult(not errors, confidence, errors, warnings)


def get_regime_with_hysteresis(score, prev_regime, weeks_pending, hard_kill):
    if hard_kill:
        return "OFF-KILL", 0, "Kill aktif"
    
    if not prev_regime:
        if score >= 60: return "ON", 0, ""
        elif score >= 40: return "NEUTRAL", 0, ""
        else: return "OFF", 0, ""
    
    target = None
    
    if prev_regime == "ON":
        if score < HYSTERESIS["ON_TO_NEUTRAL"]:
            target = "NEUTRAL"
        else:
            return "ON", 0, ""
    elif prev_regime == "NEUTRAL":
        if score > HYSTERESIS["NEUTRAL_TO_ON"]:
            target = "ON"
        elif score < HYSTERESIS["NEUTRAL_TO_OFF"]:
            target = "OFF"
        else:
            return "NEUTRAL", 0, ""
    elif prev_regime == "OFF":
        if score > HYSTERESIS["OFF_TO_NEUTRAL"]:
            target = "NEUTRAL"
        else:
            return "OFF", 0, ""
    elif prev_regime == "OFF-KILL":
        if score >= 60: return "ON", 0, ""
        elif score >= 40: return "NEUTRAL", 0, ""
        else: return "OFF", 0, ""
    
    if target:
        new_weeks = weeks_pending + 1
        if new_weeks >= HYSTERESIS["CONFIRM_WEEKS"]:
            return target, 0, f"→{target}"
        return prev_regime, new_weeks, f"⏳{HYSTERESIS['CONFIRM_WEEKS']-new_weeks}h"
    
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


# ================================
# CHARTS
# ================================
def create_gauge(score, regime):
    color = STATE_COLORS.get(regime, "#666")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'size': 32, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#666"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "#1a1a2e",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(255,23,68,0.2)'},
                {'range': [40, 60], 'color': 'rgba(255,193,7,0.2)'},
                {'range': [60, 100], 'color': 'rgba(0,200,83,0.2)'}
            ],
        }
    ))
    
    fig.update_layout(
        height=160, margin=dict(l=20, r=20, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def create_radar(scores_dict):
    cats = ['Döviz', 'CDS', 'Küresel', 'Faiz', 'Likidite']
    vals = [scores_dict['doviz'], scores_dict['cds'], scores_dict['global'], 
            scores_dict['faiz'], scores_dict['likidite']]
    vals.append(vals[0])
    cats.append(cats[0])
    
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill='toself',
        fillcolor='rgba(0,212,255,0.3)',
        line=dict(color='#00d4ff', width=2),
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color='#888')),
            angularaxis=dict(tickfont=dict(size=10, color='#ccc')),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False, height=180,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ================================
# SESSION STATE
# ================================
defaults = {
    "previous_regime": "ON",
    "weeks_in_transition": 0,
    "manual_kill": False,
    "kill_reason": "",
    "kill_confirmed": False,
    "prev_usdtry": 35.30,
    "prev_cds": 204.0,
    "prev_vix": 17.5,
    "scan_results": [],
    "xu100_close": None,
    "xu100_ema50": None,
    "last_scan": None,
    "scan_errors": 0,
    "show_kill_panel": False
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.markdown("### ⚙️ MFS Ayarları")
    
    st.markdown("##### 📅 Önceki Hafta")
    prev_regime = st.selectbox("Rejim", ["ON", "NEUTRAL", "OFF", "OFF-KILL"],
        index=["ON", "NEUTRAL", "OFF", "OFF-KILL"].index(st.session_state.previous_regime))
    weeks_pending = st.number_input("Bekleyen", 0, 5, st.session_state.weeks_in_transition)
    
    st.markdown("---")
    st.markdown("##### 💵 Döviz")
    c1, c2 = st.columns(2)
    usdtry_price = c1.number_input("USD/TRY", value=st.session_state.prev_usdtry, step=0.1, format="%.2f")
    usdtry_wchg_pct = c2.number_input("Δ%", value=0.8, step=0.1, format="%.1f")
    usdtry_wchg = usdtry_wchg_pct / 100
    
    st.markdown("##### 📈 CDS")
    c1, c2 = st.columns(2)
    cds_level = c1.number_input("Seviye", value=st.session_state.prev_cds, step=5.0, format="%.0f")
    cds_wdelta = c2.number_input("Δbp", value=0.0, step=5.0, format="%.0f")
    
    st.markdown("##### 🌍 Küresel")
    c1, c2 = st.columns(2)
    vix_last = c1.number_input("VIX", value=st.session_state.prev_vix, step=0.5, format="%.1f")
    sp500_wchg_pct = c2.number_input("S&P%", value=1.0, step=0.5, format="%.1f")
    sp500_wchg = sp500_wchg_pct / 100
    
    st.markdown("##### 🏦 BIST")
    c1, c2 = st.columns(2)
    xu100_wchg_pct = c1.number_input("XU100%", value=2.0, step=0.5, format="%.1f")
    xbank_wchg_pct = c2.number_input("XBANK%", value=2.5, step=0.5, format="%.1f")
    xu100_wchg = xu100_wchg_pct / 100
    xbank_wchg = xbank_wchg_pct / 100
    
    st.markdown("##### 💧 Diğer")
    c1, c2 = st.columns(2)
    volume_ratio = c1.number_input("Hacim", value=1.0, step=0.1, format="%.1f")
    faiz_score = c2.number_input("Faiz", value=60, step=5)
    
    st.markdown("---")
    if st.button("💾 Kaydet + Log", use_container_width=True):
        st.session_state.prev_usdtry = usdtry_price
        st.session_state.prev_cds = cds_level
        st.session_state.prev_vix = vix_last
        st.toast("✅ Kaydedildi!")


# ================================
# MFS HESAPLAMALAR
# ================================
validation = validate_data(usdtry_wchg, cds_level, cds_wdelta, vix_last, sp500_wchg)

k1_ok = usdtry_wchg < TH["K1_USDTRY_SHOCK"]
k2_ok = (cds_level < TH["K2_CDS_LEVEL"]) and (cds_wdelta < TH["K2_CDS_SPIKE"])
k3_ok = not ((vix_last > TH["K3_VIX"]) and (sp500_wchg <= TH["K3_SP500"]))
k4_ok = not ((xbank_wchg <= TH["K4_XBANK_DROP"]) and (xu100_wchg > TH["K4_XU100_STABLE"]))
k5_ok = volume_ratio >= TH["K5_VOLUME_RATIO"]

checks = {"K1": k1_ok, "K2": k2_ok, "K3": k3_ok, "K4": k4_ok, "K5": k5_ok}

# Manuel kill SADECE onaylanmışsa aktif
hard_kill = (not k1_ok) or (not k2_ok) or (not k3_ok) or (st.session_state.manual_kill and st.session_state.kill_confirmed)

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

regime, new_weeks, transition_note = get_regime_with_hysteresis(total, prev_regime, weeks_pending, hard_kill)

st.session_state.previous_regime = regime
st.session_state.weeks_in_transition = new_weeks

base_pos, base_risk = BASE_BUDGETS[regime]
if soft_reduction > 0:
    adj_pos = max(2, int(math.floor(base_pos * (1 - soft_reduction))))
    adj_risk = round(base_risk * (1 - soft_reduction), 1)
else:
    adj_pos, adj_risk = base_pos, base_risk


# ================================
# MAIN UI
# ================================
# Header
c1, c2 = st.columns([4, 1])
with c1:
    st.markdown(f"## 🎯 MFS + RAMKAR {APP_VERSION}")
with c2:
    if st.button("🚨 ACİL", type="secondary" if not st.session_state.manual_kill else "primary"):
        st.session_state.show_kill_panel = not st.session_state.show_kill_panel

# ================================
# 2 ADIMLI MANUEL KILL PANELİ
# ================================
if st.session_state.show_kill_panel:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff1744, #ad1457); 
                padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">🚨 MANUEL KILL-SWITCH PROTOKOLÜ</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.manual_kill:
        # ADIM 1: Sebep yaz
        st.markdown("### Adım 1: Sebebi Yaz")
        kill_reason = st.text_area(
            "Neden manuel kill aktifleştiriyorsun?",
            placeholder="Örn: Cumartesi gecesi savaş haberi geldi, piyasa kapalı...",
            key="kill_reason_input"
        )
        
        # ADIM 2: Onay kutusu
        st.markdown("### Adım 2: Protokol Onayı")
        confirm_check = st.checkbox(
            "Bu kill manuel protokole uygundur (Kategori A/B olay veya Duygu Kotası)",
            key="kill_confirm_check"
        )
        
        # Aktivasyon
        if kill_reason and len(kill_reason) >= 10 and confirm_check:
            if st.button("🔴 MANUEL KILL AKTİFLEŞTİR", type="primary", use_container_width=True):
                st.session_state.manual_kill = True
                st.session_state.kill_confirmed = True
                st.session_state.kill_reason = kill_reason
                
                # CSV'ye logla
                log_to_csv("MANUEL_KILL_AKTIF", kill_reason, total, regime)
                
                st.success("✅ Manuel Kill aktifleştirildi ve loglandı!")
                st.rerun()
        else:
            st.warning("⚠️ Aktivasyon için: Sebep yaz (min 10 karakter) + Onay kutusunu işaretle")
    
    else:
        # Kill aktif - kaldırma seçeneği
        st.error(f"""
        🔴 **MANUEL KILL AKTİF**
        
        **Sebep:** {st.session_state.kill_reason}
        **Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
        """)
        
        if st.button("🟢 Manuel Kill'i Kaldır", use_container_width=True):
            # Kaldırma logla
            log_to_csv("MANUEL_KILL_KALDIRILDI", f"Önceki sebep: {st.session_state.kill_reason}", total, regime)
            
            st.session_state.manual_kill = False
            st.session_state.kill_confirmed = False
            st.session_state.kill_reason = ""
            st.rerun()
    
    st.markdown("---")

# Uyarılar
if validation.errors:
    st.error(f"⛔ {' | '.join(validation.errors)}")
if st.session_state.manual_kill and st.session_state.kill_confirmed:
    st.error(f"🚨 MANUEL KILL AKTİF: {st.session_state.kill_reason[:50]}...")

# Ana Metrikler
c1, c2, c3, c4 = st.columns(4)

color = STATE_COLORS[regime]
c1.markdown(f"""<div style="background:{color}22;border:2px solid {color};border-radius:8px;padding:10px;text-align:center;">
<div style="color:#888;font-size:0.8rem;">REJİM</div>
<div style="color:{color};font-size:1.5rem;font-weight:bold;">{regime}</div>
</div>""", unsafe_allow_html=True)

sc = "#00c853" if total >= 60 else "#ffc107" if total >= 40 else "#ff1744"
c2.markdown(f"""<div style="background:#1a1a2e;border:2px solid {sc};border-radius:8px;padding:10px;text-align:center;">
<div style="color:#888;font-size:0.8rem;">SKOR</div>
<div style="color:{sc};font-size:1.5rem;font-weight:bold;">{total}</div>
</div>""", unsafe_allow_html=True)

c3.markdown(f"""<div style="background:#1a1a2e;border:2px solid #00d4ff;border-radius:8px;padding:10px;text-align:center;">
<div style="color:#888;font-size:0.8rem;">POZ</div>
<div style="color:#00d4ff;font-size:1.5rem;font-weight:bold;">{adj_pos}</div>
</div>""", unsafe_allow_html=True)

c4.markdown(f"""<div style="background:#1a1a2e;border:2px solid #00d4ff;border-radius:8px;padding:10px;text-align:center;">
<div style="color:#888;font-size:0.8rem;">RİSK</div>
<div style="color:#00d4ff;font-size:1.5rem;font-weight:bold;">{adj_risk}R</div>
</div>""", unsafe_allow_html=True)

if transition_note:
    st.info(f"🔄 {transition_note}")

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 MFS", "🔥 RAMKAR", "📋 LOG"])

with tab1:
    c1, c2 = st.columns(2)
    
    with c1:
        st.plotly_chart(create_gauge(total, regime), use_container_width=True)
        ks = " ".join([f"{'✅' if v else '❌'}{k}" for k, v in checks.items()])
        st.markdown(f"**Kill-Switch:** {ks}")
    
    with c2:
        st.plotly_chart(create_radar(scores), use_container_width=True)
    
    if regime == "ON":
        st.success(f"🟢 **YEŞİL** | Max {adj_pos} poz, {adj_risk}R")
    elif regime == "NEUTRAL":
        st.warning(f"🟡 **DİKKAT** | Max {adj_pos} poz | Sadece A kalite")
    elif regime == "OFF":
        st.error(f"🔴 **RİSKLİ** | Max {adj_pos} poz | Çok sınırlı")
    else:
        st.error(f"💀 **KİLİTLİ** | İşlem YASAK!")

with tab2:
    st.markdown("### 🔥 RAMKAR v30.4 Tarama")
    
    # ================================
    # OFF-KILL İKEN TARAMA KİLİTLİ
    # ================================
    if regime == "OFF-KILL":
        st.error("""
        ### 🔒 TARAMA KİLİTLİ
        
        MFS rejimi **OFF-KILL** durumunda.
        
        Tarama yapılamaz çünkü:
        - Tüm sinyaller geçersizdir
        - Fırsat görmenin anlamı yok
        - Önce MFS'i normale döndür
        
        **Yapman gereken:** Piyasa sakinleşene kadar bekle, manuel kill'i kaldır.
        """)
    
    elif not YF_AVAILABLE:
        st.error("yfinance yüklü değil: `pip install yfinance`")
    
    else:
        # OFF modunda uyarı
        if regime == "OFF":
            st.warning("⚠️ MFS OFF - Tarama sadece bilgi amaçlı. Yeni giriş riskli!")
        
        c1, c2, c3 = st.columns([2, 1, 1])
        
        with c1:
            scan_btn = st.button("🔄 TARA", type="primary", use_container_width=True)
        with c2:
            st.caption(f"**{len(KATILIM_HISSELERI)}** hisse")
        with c3:
            if st.session_state.last_scan:
                st.caption(f"Son: {st.session_state.last_scan.strftime('%H:%M')}")
        
        if scan_btn:
            progress = st.progress(0)
            status = st.empty()
            
            def update(pct, sym):
                progress.progress(pct)
                status.text(f"Taranıyor: {sym}")
            
            results, xu100_c, xu100_e, success, errors = run_full_scan(update)
            
            st.session_state.scan_results = results
            st.session_state.xu100_close = xu100_c
            st.session_state.xu100_ema50 = xu100_e
            st.session_state.last_scan = datetime.now()
            st.session_state.scan_errors = errors
            
            # Taramayı logla
            radar_count = len([r for r in results if r['status'] == 'RADAR'])
            log_to_csv("TARAMA", f"Başarılı:{success} Hata:{errors} RADAR:{radar_count}", total, regime)
            
            progress.empty()
            status.empty()
            st.rerun()
        
        # Sonuçlar
        if st.session_state.scan_results:
            results = st.session_state.scan_results
            
            # Hata uyarısı
            if st.session_state.scan_errors > 0:
                st.warning(f"⚠️ {st.session_state.scan_errors} hisse veri çekilemedi (Yahoo Finance)")
            
            xu100_ok = st.session_state.xu100_close and st.session_state.xu100_ema50 and \
                       st.session_state.xu100_close > st.session_state.xu100_ema50
            xu100_status = "✅" if xu100_ok else "❌"
            
            radar = [r for r in results if r['status'] == 'RADAR']
            izleme = [r for r in results if r['status'] == 'İZLEME']
            
            st.markdown(f"**XU100:** {xu100_status} | **🚀 RADAR:** {len(radar)} | **⚡ İZLEME:** {len(izleme)}")
            
            st.markdown("---")
            st.markdown("### 🏆 Top 10")
            
            # OFF modunda grileştir ama göster
            blocked = regime == "OFF"
            
            for i, r in enumerate(results[:10]):
                opacity = "0.6" if blocked else "1"
                
                crit = f"{'✅' if r['trend_ok'] else '❌'}{'✅' if r['power_ok'] else '❌'}{'✅' if r['vol_ok'] else '❌'}{'✅' if r['sar_ok'] else '❌'}{'✅' if r['dist_ok'] else '❌'}{'✅' if r['bist_ok'] else '❌'}"
                
                c1, c2, c3 = st.columns([2, 3, 2])
                
                with c1:
                    st.markdown(f"""
                    <div style="opacity:{opacity};">
                        <span style="font-size:1.2rem;font-weight:bold;color:white;">{i+1}. {r['symbol']}</span><br>
                        <span style="color:#888;">₺{r['price']:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c2:
                    st.markdown(f"""
                    <div style="opacity:{opacity};font-size:0.85rem;">
                        <span style="color:#ccc;">{crit}</span><br>
                        <span style="color:#666;">ADX:{r['adx']:.0f} MFI:{r['mfi']:.0f} Dist:{r['ema_dist']:+.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c3:
                    if blocked:
                        st.markdown(f"""
                        <div style="background:#333;border-radius:8px;padding:6px;text-align:center;">
                            <span style="color:#888;font-weight:bold;">{r['score']}/6</span><br>
                            <span style="color:#666;font-size:0.8rem;">⚠️ OFF</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:{r['status_color']}22;border:2px solid {r['status_color']};border-radius:8px;padding:6px;text-align:center;">
                            <span style="color:{r['status_color']};font-weight:bold;">{r['score']}/6</span><br>
                            <span style="color:{r['status_color']};font-size:0.8rem;">{r['status_icon']} {r['status']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<hr style='margin:4px 0;border-color:#333;'>", unsafe_allow_html=True)
            
            with st.expander("📋 Tüm Sonuçlar"):
                df = pd.DataFrame(results)
                cols = ['symbol', 'price', 'score', 'status', 'adx', 'mfi', 'ema_dist', 'atr_pct']
                df_show = df[cols].copy()
                df_show.columns = ['Hisse', 'Fiyat', 'Skor', 'Durum', 'ADX', 'MFI', 'Mesafe%', 'ATR%']
                st.dataframe(df_show, hide_index=True, use_container_width=True)
        
        else:
            st.info("👆 TARA butonuna bas")

with tab3:
    st.markdown("### 📋 Kara Kutu (Event Log)")
    
    log_df = get_log_history()
    
    if not log_df.empty:
        st.dataframe(log_df.tail(20).iloc[::-1], hide_index=True, use_container_width=True)
        
        # İndirme butonu
        csv_data = log_df.to_csv(index=False)
        st.download_button(
            "📥 CSV İndir",
            csv_data,
            "mfs_kara_kutu.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("Henüz log kaydı yok. İşlemler kaydedilecek.")
    
    st.markdown("---")
    st.markdown("""
    **Kaydedilen Olaylar:**
    - Manuel Kill aktivasyon/kaldırma
    - Tarama sonuçları
    - Haftalık veri güncellemeleri
    """)

# Footer
st.markdown("---")
st.caption(f"MFS + RAMKAR {APP_VERSION} • {datetime.now().strftime('%d/%m/%Y %H:%M')} • Kara Kutu Aktif")
