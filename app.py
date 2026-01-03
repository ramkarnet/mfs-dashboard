# app.py
# RAMKAR MFS v2.3 - Streamlit Dashboard
# Tek tuş: veri çek -> hesapla -> dashboard

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional dependencies
try:
    import yfinance as yf
except Exception:
    yf = None

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
)

APP_VERSION = "v2.3"
TZ = "Europe/Istanbul"

TICKERS = {
    "USDTRY": "USDTRY=X",
    "VIX": "^VIX",
    "SP500": "^GSPC",
    "XU100": "XU100.IS",
    "XBANK": "XBANK.IS",
}

# Thresholds
TH = {
    "K1_USDTRY_SHOCK": 0.05,       # weekly %5
    "K2_CDS_SPIKE": 100.0,         # +100bp weekly
    "K2_CDS_LEVEL": 700.0,         # 700bp level
    "K3_VIX": 35.0,
    "K3_SP500": -0.03,             # -%3 weekly
    "K4_XBANK_DROP": -0.05,        # -%5 weekly
    "K4_XU100_STABLE": -0.01,      # XU100 > -%1 means "stable"
    "K5_VOLUME_RATIO": 0.5,        # <0.5 = low liquidity
}

# Soft veto budget reductions
BUDGET_REDUCTIONS = {
    "K4": 0.25,  # 25%
    "K5": 0.15,  # 15%
}

# Weights
W = {
    "doviz": 0.30,
    "cds": 0.25,
    "global": 0.25,
    "faiz": 0.15,
    "likidite": 0.05,
}

# Risk budget by regime
BASE_BUDGETS = {
    "ON": (12, 2.5, "✅ NORMAL"),
    "NEUTRAL": (7, 1.5, "✅ SEÇİCİ"),
    "OFF": (4, 1.0, "⚠️ SINIRLI"),
    "OFF-KILL": (2, 0.5, "❌ YASAK"),
}

STATE_ICON = {"ON": "🟢", "NEUTRAL": "🟡", "OFF": "🔴", "OFF-KILL": "💀"}


# -----------------------------
# DATA MODEL
# -----------------------------
@dataclass
class MarketSnapshot:
    asof: datetime
    usdtry_close: float
    usdtry_wchg: float

    vix_last: float

    sp500_wchg: float

    xu100_wchg: float
    xbank_wchg: float

    cds_level: float
    cds_wdelta: float
    cds_is_provisional: bool

    volume_ratio: float
    faiz_proxy: float  # fixed score placeholder


# -----------------------------
# HELPERS
# -----------------------------
def safe_pct_change(last: float, prev: float) -> float:
    if prev is None or prev == 0 or np.isnan(prev):
        return np.nan
    return (last - prev) / prev

def bar10(score: int) -> str:
    score = int(max(0, min(100, score)))
    filled = score // 10
    return "█" * filled + "░" * (10 - filled)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_weekly_ohlc(ticker: str, weeks: int = 12) -> Optional[pd.DataFrame]:
    """
    Pull weekly bars. Needs internet + yfinance.
    """
    if yf is None:
        return None
    try:
        df = yf.download(ticker, period=f"{max(weeks, 4)}wk", interval="1wk", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Close"])
        return df
    except Exception:
        return None

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_daily_last(ticker: str, days: int = 10) -> Optional[pd.DataFrame]:
    if yf is None:
        return None
    try:
        df = yf.download(ticker, period=f"{max(days, 5)}d", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Close"])
        return df
    except Exception:
        return None


def compute_weekly_change_from_df(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Returns: (last_close, weekly_change) using last 2 weekly closes.
    """
    if df is None or df.empty or len(df) < 2:
        return (np.nan, np.nan)
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])
    return last, safe_pct_change(last, prev)


def score_doviz(usdtry_wchg: float) -> Tuple[int, str]:
    c = abs(usdtry_wchg)
    if np.isnan(c):
        return 60, "⚠️ Veri yok"
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
    lvl = cds_level
    if np.isnan(lvl) or lvl <= 0:
        return 50, "⚠️ Veri yok"
    if lvl < 300:
        base = 100
        status = "✅ Güvenli"
    elif lvl < 400:
        base = 70
        status = "⚠️ Normal"
    elif lvl < 500:
        base = 50
        status = "🟠 Dikkat"
    elif lvl < 600:
        base = 30
        status = "🔴 Riskli"
    elif lvl < 700:
        base = 10
        status = "💀 Kriz"
    else:
        base = 0
        status = "💀 Çöküş"

    # weekly delta penalty
    if not np.isnan(cds_wdelta) and cds_wdelta > 50:
        base = max(0, base - 20)

    return base, status

def score_global(vix_last: float, sp500_wchg: float) -> Tuple[int, str]:
    v = vix_last
    if np.isnan(v):
        base, status = 60, "⚠️ Veri yok"
    elif v < 20:
        base, status = 100, "✅ Sakin"
    elif v < 25:
        base, status = 80, "✅ Normal"
    elif v < 30:
        base, status = 60, "⚠️ Gergin"
    elif v < 35:
        base, status = 40, "🟠 Alarm"
    else:
        base, status = 20, "🔴 Panik"

    # equity drawdown penalty
    if not np.isnan(sp500_wchg):
        if sp500_wchg < -0.02:
            base = max(0, base - 20)
        elif sp500_wchg < -0.01:
            base = max(0, base - 10)

    return base, status

def score_likidite(volume_ratio: float) -> Tuple[int, str]:
    vr = volume_ratio
    if np.isnan(vr) or vr <= 0:
        return 40, "⚠️ Veri yok"
    if vr >= 1.2:
        return 100, "✅ Yüksek"
    if vr >= 0.8:
        return 70, "✅ Normal"
    if vr >= 0.5:
        return 40, "⚠️ Düşük"
    return 10, "🔴 Kritik"


def kill_switch_checks(snap: MarketSnapshot) -> Dict[str, bool]:
    # True means OK / pass
    k1_ok = (not np.isnan(snap.usdtry_wchg)) and (snap.usdtry_wchg < TH["K1_USDTRY_SHOCK"])
    # CDS kill: either spike or level
    k2_ok = (snap.cds_level < TH["K2_CDS_LEVEL"]) and (snap.cds_wdelta < TH["K2_CDS_SPIKE"])
    # Global kill: VIX > 35 AND SP500 <= -3%
    k3_ok = not ((snap.vix_last > TH["K3_VIX"]) and (snap.sp500_wchg <= TH["K3_SP500"]))

    # Soft veto checks
    k4_ok = not ((snap.xbank_wchg <= TH["K4_XBANK_DROP"]) and (snap.xu100_wchg > TH["K4_XU100_STABLE"]))
    k5_ok = snap.volume_ratio >= TH["K5_VOLUME_RATIO"]

    return {"K1": k1_ok, "K2": k2_ok, "K3": k3_ok, "K4": k4_ok, "K5": k5_ok}


def compute_soft_veto_reduction(checks: Dict[str, bool]) -> Tuple[float, List[str]]:
    reduction = 0.0
    reasons = []
    if not checks["K4"]:
        reduction += BUDGET_REDUCTIONS["K4"]
        reasons.append("K4: Banka ayrışması (XBANK çöküyor, XU100 stabil)")
    if not checks["K5"]:
        reduction += BUDGET_REDUCTIONS["K5"]
        reasons.append("K5: Likidite düşük (hacim oranı < 0.5)")
    reduction = clamp(reduction, 0.0, 0.5)
    return reduction, reasons


def compute_total_score(scores: Dict[str, int]) -> int:
    total = (
        scores["doviz"] * W["doviz"]
        + scores["cds"] * W["cds"]
        + scores["global"] * W["global"]
        + scores["faiz"] * W["faiz"]
        + scores["likidite"] * W["likidite"]
    )
    return int(round(total))


def determine_regime(hard_kill: bool, total_score: int) -> str:
    if hard_kill:
        return "OFF-KILL"
    if total_score >= 60:
        return "ON"
    if total_score >= 40:
        return "NEUTRAL"
    return "OFF"


def adjusted_budget(regime: str, soft_reduction: float) -> Tuple[Tuple[int, float, str], Tuple[int, float, str]]:
    base = BASE_BUDGETS[regime]
    if soft_reduction <= 0:
        return base, base
    max_pos, max_risk, new_entry = base
    factor = 1.0 - soft_reduction
    adj_pos = max(2, int(math.floor(max_pos * factor)))
    adj_risk = round(max_risk * factor, 1)
    adj_entry = new_entry if soft_reduction < 0.3 else "⚠️ DİKKATLİ"
    return (adj_pos, adj_risk, adj_entry), base


def top_risks(snap: MarketSnapshot) -> List[Dict[str, str]]:
    # distance to alarm thresholds
    # smaller distance = closer to risk
    risks = []

    # VIX alarm threshold 25
    vix_dist = 0 if snap.vix_last >= 25 else (25 - snap.vix_last) / 25 * 100
    risks.append({
        "name": "VIX",
        "value": f"{snap.vix_last:.2f}" if not np.isnan(snap.vix_last) else "NA",
        "threshold": "25 (alarm)",
        "distance": f"{max(0, vix_dist):.0f}%",
        "distance_num": float(max(0, vix_dist)),
    })

    # USDTRY alarm threshold 1.5%
    u = abs(snap.usdtry_wchg) if not np.isnan(snap.usdtry_wchg) else np.nan
    usd_dist = 0 if (np.isnan(u) or u >= 0.015) else (0.015 - u) / 0.015 * 100
    risks.append({
        "name": "USDTRY",
        "value": f"%{snap.usdtry_wchg*100:+.2f}" if not np.isnan(snap.usdtry_wchg) else "NA",
        "threshold": "%1.5 (alarm)",
        "distance": f"{max(0, usd_dist):.0f}%" if not np.isnan(usd_dist) else "NA",
        "distance_num": float(max(0, usd_dist)) if not np.isnan(usd_dist) else 9999.0,
    })

    # CDS alarm threshold 400
    c = snap.cds_level
    cds_dist = 0 if c >= 400 else (400 - c) / 400 * 100
    risks.append({
        "name": "CDS",
        "value": f"{c:.2f} bp" if not np.isnan(c) else "NA",
        "threshold": "400 bp (alarm)",
        "distance": f"{max(0, cds_dist):.0f}%",
        "distance_num": float(max(0, cds_dist)),
    })

    risks.sort(key=lambda x: x["distance_num"])
    return [{k: v for k, v in r.items() if k != "distance_num"} for r in risks[:2]]


def confidence_level(snap: MarketSnapshot) -> Tuple[str, List[str]]:
    reasons = []
    penalty = 0

    if snap.cds_is_provisional:
        reasons.append("CDS verisi PROVISIONAL (manuel girildi)")
        penalty += 1

    if np.isnan(snap.usdtry_wchg) or np.isnan(snap.vix_last) or np.isnan(snap.sp500_wchg):
        reasons.append("Bazı otomatik veriler eksik (yfinance / internet / ticker sorunu)")
        penalty += 2

    # faiz proxy always provisional by design
    reasons.append("Faiz modülü PROXY (TCMB entegrasyonu yok)")
    penalty += 1

    if penalty <= 1:
        return "HIGH", reasons
    if penalty <= 3:
        return "MEDIUM", reasons
    return "LOW", reasons


def plot_series(df: pd.DataFrame, title: str) -> Optional[object]:
    if go is None or df is None or df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    fig.update_layout(
        title=title,
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="",
        yaxis_title="",
    )
    return fig


# -----------------------------
# UI - HEADER
# -----------------------------
st.title(f"🎯 RAMKAR MFS {APP_VERSION} — Makro Risk Dashboard")
st.caption("Tek amaç: Makro ortam 'risk al' mı 'risk kıs' mı? RAMKAR sinyalinden bağımsız filtre.")

with st.sidebar:
    st.subheader("Kontrol Paneli")
    st.write("Çalıştır dediğinde verileri çeker (internet gerekebilir), CDS'i sen girersin.")
    run_btn = st.button("▶ ÇALIŞTIR / GÜNCELLE", use_container_width=True)

    st.divider()
    st.markdown("### CDS (manuel)")
    cds_level = st.number_input("CDS (bp)", min_value=0.0, value=203.98, step=1.0)
    cds_wdelta = st.number_input("CDS haftalık değişim (bp)", value=0.0, step=1.0)
    cds_prov = st.toggle("CDS PROVISIONAL", value=True)

    st.divider()
    st.markdown("### Likidite (manuel)")
    volume_ratio = st.number_input("BIST hacim oranı (1.0 = normal)", min_value=0.0, value=1.0, step=0.1)

    st.divider()
    st.markdown("### Faiz (şimdilik proxy)")
    faiz_score = st.slider("Faiz skoru (proxy)", min_value=0, max_value=100, value=60, step=1)

    st.divider()
    st.markdown("### Veri Kaynağı")
    st.write("- Otomatik: yfinance (USDTRY, VIX, S&P500, XU100, XBANK)")
    st.write("- Manuel: CDS, Hacim oranı")
    st.write("- Not: CDS otomasyonu stabil kaynak ister; sonra ekleriz.")


# -----------------------------
# DATA LOAD
# -----------------------------
if "last_snapshot" not in st.session_state:
    st.session_state["last_snapshot"] = None
if "score_history" not in st.session_state:
    st.session_state["score_history"] = []

def build_snapshot() -> Tuple[MarketSnapshot, Dict[str, Optional[pd.DataFrame]]]:
    now = datetime.now()

    weekly = {}
    daily = {}

    if yf is not None:
        weekly["USDTRY"] = fetch_weekly_ohlc(TICKERS["USDTRY"], weeks=12)
        weekly["SP500"] = fetch_weekly_ohlc(TICKERS["SP500"], weeks=12)
        weekly["XU100"] = fetch_weekly_ohlc(TICKERS["XU100"], weeks=12)
        weekly["XBANK"] = fetch_weekly_ohlc(TICKERS["XBANK"], weeks=12)
        daily["VIX"] = fetch_daily_last(TICKERS["VIX"], days=10)
    else:
        weekly = {k: None for k in ["USDTRY", "SP500", "XU100", "XBANK"]}
        daily = {"VIX": None}

    usdtry_close, usdtry_wchg = compute_weekly_change_from_df(weekly["USDTRY"])
    _, sp500_wchg = compute_weekly_change_from_df(weekly["SP500"])
    _, xu100_wchg = compute_weekly_change_from_df(weekly["XU100"])
    _, xbank_wchg = compute_weekly_change_from_df(weekly["XBANK"])

    vix_last = np.nan
    if daily["VIX"] is not None and not daily["VIX"].empty:
        vix_last = float(daily["VIX"]["Close"].iloc[-1])

    snap = MarketSnapshot(
        asof=now,
        usdtry_close=to_float(usdtry_close),
        usdtry_wchg=to_float(usdtry_wchg),

        vix_last=to_float(vix_last),

        sp500_wchg=to_float(sp500_wchg),

        xu100_wchg=to_float(xu100_wchg),
        xbank_wchg=to_float(xbank_wchg),

        cds_level=float(cds_level),
        cds_wdelta=float(cds_wdelta),
        cds_is_provisional=bool(cds_prov),

        volume_ratio=float(volume_ratio),
        faiz_proxy=float(faiz_score),
    )

    raw = {
        "USDTRY_weekly": weekly["USDTRY"],
        "SP500_weekly": weekly["SP500"],
        "XU100_weekly": weekly["XU100"],
        "XBANK_weekly": weekly["XBANK"],
        "VIX_daily": daily["VIX"],
    }
    return snap, raw


if run_btn or st.session_state["last_snapshot"] is None:
    with st.spinner("Veriler alınıyor ve hesaplanıyor..."):
        snap, raw = build_snapshot()
        st.session_state["last_snapshot"] = (snap, raw)
else:
    snap, raw = st.session_state["last_snapshot"]


# -----------------------------
# CORE CALC
# -----------------------------
checks = kill_switch_checks(snap)
hard_kill = (not checks["K1"]) or (not checks["K2"]) or (not checks["K3"])
soft_reduction, soft_reasons = compute_soft_veto_reduction(checks)

doviz_score, doviz_status = score_doviz(snap.usdtry_wchg)
cds_score, cds_status = score_cds(snap.cds_level, snap.cds_wdelta)
glob_score, glob_status = score_global(snap.vix_last, snap.sp500_wchg)
lik_score, lik_status = score_likidite(snap.volume_ratio)
faiz_score_fixed = int(clamp(snap.faiz_proxy, 0, 100))

scores = {
    "doviz": int(doviz_score),
    "cds": int(cds_score),
    "global": int(glob_score),
    "faiz": int(faiz_score_fixed),
    "likidite": int(lik_score),
}

total = compute_total_score(scores)
regime = determine_regime(hard_kill, total)

(adj_pos, adj_risk, adj_entry), (base_pos, base_risk, base_entry) = adjusted_budget(regime, soft_reduction)
top2 = top_risks(snap)
conf, conf_reasons = confidence_level(snap)

# trajectory history
hist = st.session_state["score_history"]
hist.append({"asof": snap.asof, "total": total})
hist = hist[-12:]  # keep last 12
st.session_state["score_history"] = hist

def trajectory_deltas(history: List[Dict]) -> Tuple[float, float, str]:
    if len(history) < 2:
        return 0.0, 0.0, "→ Stabil"
    d1 = history[-1]["total"] - history[-2]["total"]
    d4 = d1
    if len(history) >= 4:
        d4 = history[-1]["total"] - history[-4]["total"]
    if d1 > 3:
        trend = "↑ İyileşiyor"
    elif d1 < -3:
        trend = "↓ Kötüleşiyor"
    else:
        trend = "→ Stabil"
    return float(d1), float(d4), trend

d1, d4, trend = trajectory_deltas(hist)


# -----------------------------
# UI - KPI ROW
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("RiskState", f"{STATE_ICON[regime]} {regime}")
with c2:
    st.metric("Toplam Skor", f"{total} / 100", delta=f"{d1:+.0f} (1w)")
with c3:
    st.metric("Max Pozisyon", f"{adj_pos}", delta=f"{adj_pos-base_pos:+d}" if soft_reduction > 0 else None)
with c4:
    st.metric("Max Risk", f"{adj_risk}R", delta=f"{adj_risk-base_risk:+.1f}R" if soft_reduction > 0 else None)
with c5:
    st.metric("Confidence", conf)

st.divider()


# -----------------------------
# UI - MAIN LAYOUT
# -----------------------------
left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("Kill-Switch / Soft Veto Durumu")

    ks_cols = st.columns(5)
    for i, k in enumerate(["K1", "K2", "K3", "K4", "K5"]):
        ok = checks[k]
        label = {
            "K1": "K1 Döviz",
            "K2": "K2 CDS",
            "K3": "K3 Küresel",
            "K4": "K4 Banka",
            "K5": "K5 Likidite",
        }[k]
        status = "✅ OK" if ok else ("❌ KILL" if k in ["K1", "K2", "K3"] else "⚠️ SOFT")
        ks_cols[i].write(f"**{label}**")
        ks_cols[i].markdown(f"<div style='font-size:22px; font-weight:700'>{status}</div>", unsafe_allow_html=True)

    st.caption(
        "Not: K1-K2-K3 hard kill (OFF-KILL). K4-K5 soft veto (risk bütçesini düşürür, rejimi öldürmez)."
    )

    if hard_kill:
        st.error("Hard Kill aktif: Yeni giriş yasak. Rejim: OFF-KILL.")
    elif soft_reduction > 0:
        st.warning(f"Soft veto aktif: Risk bütçesi -%{int(soft_reduction*100)}. Nedenler: " + " | ".join(soft_reasons))
    else:
        st.success("Tüm kontroller normal: Hard kill yok, soft veto yok.")

    st.subheader("Faktör Skorları")
    df_scores = pd.DataFrame([
        {"Faktör": "Döviz (USDTRY)", "Skor": scores["doviz"], "Durum": doviz_status, "Detay": f"%{snap.usdtry_wchg*100:+.2f}" if not np.isnan(snap.usdtry_wchg) else "NA"},
        {"Faktör": "CDS", "Skor": scores["cds"], "Durum": cds_status + (" (PROV)" if snap.cds_is_provisional else ""), "Detay": f"{snap.cds_level:.2f} bp (Δ{snap.cds_wdelta:+.0f})"},
        {"Faktör": "Küresel (VIX+S&P)", "Skor": scores["global"], "Durum": glob_status, "Detay": f"VIX {snap.vix_last:.2f}, S&P w/w %{snap.sp500_wchg*100:+.2f}" if not np.isnan(snap.sp500_wchg) else f"VIX {snap.vix_last:.2f}"},
        {"Faktör": "Faiz", "Skor": scores["faiz"], "Durum": "⚠️ Proxy", "Detay": "TCMB entegrasyonu yok"},
        {"Faktör": "Likidite", "Skor": scores["likidite"], "Durum": lik_status, "Detay": f"Hacim oranı {snap.volume_ratio:.2f}"},
    ])

    df_scores["Bar"] = df_scores["Skor"].apply(bar10)
    df_show = df_scores[["Faktör", "Bar", "Skor", "Durum", "Detay"]]
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.subheader("Skor Trajektorisi")
    tcol1, tcol2, tcol3 = st.columns(3)
    tcol1.metric("Δ 1w", f"{d1:+.0f}")
    tcol2.metric("Δ 4w", f"{d4:+.0f}")
    tcol3.metric("Trend", trend)

    if go is not None and len(hist) >= 2:
        hist_df = pd.DataFrame(hist)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df["asof"], y=hist_df["total"], mode="lines+markers", name="Total"))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="", yaxis_title="Skor")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Grafik için plotly gerekir veya en az 2 veri noktası olmalı.")

with right:
    st.subheader("Risk Bütçesi ve Aksiyon")

    st.markdown(
        f"""
        <div style="padding:14px; border-radius:14px; border:1px solid rgba(0,0,0,0.12);">
          <div style="font-size:18px; font-weight:700">Haftalık Karar</div>
          <div style="margin-top:6px; font-size:22px; font-weight:800">{STATE_ICON[regime]} {regime}</div>
          <div style="margin-top:10px; font-size:14px;">
            <b>Max Pozisyon:</b> {adj_pos}<br/>
            <b>Max Toplam Risk:</b> {adj_risk}R<br/>
            <b>Yeni Giriş:</b> {adj_entry}<br/>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if regime == "ON":
        st.success("Makro filtre yeşil. RAMKAR teknik sinyal gelirse değerlendirilir.")
    elif regime == "NEUTRAL":
        st.warning("Seçici mod: Sadece A kalite, düşük korelasyon, düşük riskli sinyaller.")
    elif regime == "OFF":
        st.error("Makro risk yüksek: Çok sınırlı işlem / risk azalt.")
    else:
        st.error("OFF-KILL: Yeni giriş yasak. Koruma modu.")

    st.subheader("Top 2 Risk Sürücüsü")
    for r in top2:
        st.write(f"**{r['name']}**: {r['value']}")
        st.caption(f"Eşik: {r['threshold']} | Mesafe: {r['distance']}")

    st.subheader("Confidence / Data Quality")
    conf_icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}[conf]
    st.write(f"**{conf_icon} {conf}**")
    for rr in conf_reasons[:5]:
        st.caption(f"• {rr}")

    st.subheader("Veri Özeti (Asof)")
    st.write(f"**Tarih:** {snap.asof.strftime('%Y-%m-%d %H:%M')}")
    st.write(f"**USDTRY:** {snap.usdtry_close:.4f} | w/w: %{snap.usdtry_wchg*100:+.2f}" if not np.isnan(snap.usdtry_close) else "**USDTRY:** NA")
    st.write(f"**VIX:** {snap.vix_last:.2f}" if not np.isnan(snap.vix_last) else "**VIX:** NA")
    st.write(f"**S&P500 w/w:** %{snap.sp500_wchg*100:+.2f}" if not np.isnan(snap.sp500_wchg) else "**S&P500 w/w:** NA")
    st.write(f"**XU100 w/w:** %{snap.xu100_wchg*100:+.2f}" if not np.isnan(snap.xu100_wchg) else "**XU100 w/w:** NA")
    st.write(f"**XBANK w/w:** %{snap.xbank_wchg*100:+.2f}" if not np.isnan(snap.xbank_wchg) else "**XBANK w/w:** NA")
    st.write(f"**CDS:** {snap.cds_level:.2f} bp | Δ: {snap.cds_wdelta:+.0f} bp")

    st.divider()
    st.subheader("Mini Grafikler")
    if go is None:
        st.caption("Mini grafikler için plotly kurulmalı: pip install plotly")
    else:
        g1, g2 = st.columns(2)
        fig_usd = plot_series(raw.get("USDTRY_weekly"), "USDTRY (Weekly Close)")
        fig_xu = plot_series(raw.get("XU100_weekly"), "XU100 (Weekly Close)")
        if fig_usd: g1.plotly_chart(fig_usd, use_container_width=True)
        if fig_xu:  g2.plotly_chart(fig_xu, use_container_width=True)

        g3, g4 = st.columns(2)
        fig_vix = plot_series(raw.get("VIX_daily"), "VIX (Daily Close)")
        fig_xb = plot_series(raw.get("XBANK_weekly"), "XBANK (Weekly Close)")
        if fig_vix: g3.plotly_chart(fig_vix, use_container_width=True)
        if fig_xb:  g4.plotly_chart(fig_xb, use_container_width=True)


# Footer
st.divider()
st.caption(
    "⚠️ Uyarı: Bu dashboard yatırım tavsiyesi değildir. MFS sadece makro risk filtresidir; işlem kararı RAMKAR sinyali + risk yönetimi ile verilir."
)
