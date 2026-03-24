"""
streamlit_app.py
================
Climate Forecast Dashboard — Rainfall, Temperature, SPI/SPEI Drought

Run:
    streamlit run streamlit_app.py

Required files in the same folder:
    rainfall_model2.py
    temperature_model.py
    drought_model.py
    enso_helper.py
    Data/climate_master.xlsx

Usage:
    1. Climate data is preloaded from Data/climate_master.xlsx
       The file contains: state, district, year, month, rain_mean, temp_mean, latitude, longitude
    2. Select State → District from the auto-populated dropdowns.
    3. Adjust parameters, optionally upload ONI file.
    4. Click ▶ Run Forecast.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings('ignore')

import streamlit.components.v1 as components
from rainfall_model2   import run_rainfall_kalman, MONTHS
from temperature_model import run_temperature_kalman, MONTH_NAMES
from drought_model     import run_drought_model, plot_drought

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Climate Hazard Forecasting · DiCRA",
    page_icon  = "🌩",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME  — Option 1: dark header, clean cards
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Hide default Streamlit chrome ───────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Dark top header bar ─────────────────────────────────── */
.cf-header {
    background: #0F2D52;
    padding: 0.7rem 2rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    border-bottom: 3px solid #1D9E75;
}
.cf-header-logo {
    font-size: 20px;
    color: white;
    font-weight: 500;
    letter-spacing: -0.3px;
    white-space: nowrap;
}
.cf-header-sep { flex: 1; }
.cf-header-badge {
    font-size: 11px;
    color: rgba(255,255,255,0.65);
    background: rgba(255,255,255,0.08);
    padding: 3px 10px;
    border-radius: 20px;
    white-space: nowrap;
}
.cf-header-district {
    font-size: 12px;
    color: #7ECFB3;
    font-weight: 500;
    white-space: nowrap;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #F7F8FA;
    border-right: 1px solid #E2E6EA;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #6B7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 1.1rem 0 0.4rem !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
    font-size: 12px !important;
    color: #374151 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #1D4E89 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 0.5rem 1rem !important;
    width: 100% !important;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #185FA5 !important;
}
[data-testid="stSidebar"] .stButton > button:disabled {
    background: #9CA3AF !important;
    color: #E5E7EB !important;
}

/* ── Metric cards ────────────────────────────────────────── */
.cf-metric-row {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
    margin: 0.8rem 0 1.4rem;
}
.cf-metric {
    background: #F0F4F8;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    border-left: 3px solid #1D4E89;
}
.cf-metric.green  { border-left-color: #1D9E75; }
.cf-metric.purple { border-left-color: #7B2D8B; }
.cf-metric.amber  { border-left-color: #D85A30; }
.cf-metric-label {
    font-size: 10px;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 3px;
}
.cf-metric-value {
    font-size: 20px;
    font-weight: 500;
    color: #111827;
    line-height: 1.2;
}
.cf-metric-unit {
    font-size: 10px;
    color: #9CA3AF;
    margin-left: 2px;
}

/* ── Section headers ─────────────────────────────────────── */
.cf-section {
    font-size: 11px;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding-bottom: 6px;
    border-bottom: 1px solid #E5E7EB;
    margin: 1.2rem 0 0.8rem;
}

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    padding: 0;
    border-bottom: 2px solid #E5E7EB;
    margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 22px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #6B7280 !important;
    background: #F3F4F6 !important;
    border: 1.5px solid #E5E7EB !important;
    border-bottom: none !important;
    margin-bottom: -2px !important;
    transition: all 0.15s ease;
    letter-spacing: 0.01em;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #E8ECF0 !important;
    color: #374151 !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1D4E89 !important;
    border-color: #CBD5E1 !important;
    border-bottom-color: white !important;
    box-shadow: 0 -2px 0 #1D4E89 inset !important;
}

/* ── Welcome card ────────────────────────────────────────── */
.cf-welcome {
    background: linear-gradient(135deg, #0F2D52 0%, #1D4E89 100%);
    border-radius: 12px;
    padding: 2.5rem 2rem;
    color: white;
    text-align: center;
    margin: 2rem 0;
}
.cf-welcome h2 {
    font-size: 22px;
    font-weight: 500;
    margin: 0 0 0.5rem;
    color: white;
}
.cf-welcome p {
    font-size: 14px;
    color: rgba(255,255,255,0.7);
    margin: 0;
}

/* ── Info strip ──────────────────────────────────────────── */
.cf-strip {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 12px;
    color: #1E40AF;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Status / spinner ────────────────────────────────────── */
[data-testid="stStatusWidget"] {
    border-radius: 8px !important;
    border: 1px solid #E5E7EB !important;
}

/* ── Expander ────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #6B7280 !important;
    background: #F9FAFB !important;
    border-radius: 6px !important;
}

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: #F3F4F6 !important; margin: 1rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS  (shared across all plot functions)
# ─────────────────────────────────────────────────────────────────────────────

C_KF  = '#1D4E89'
C_TMP = '#7B2D8B'
C_OBS = '#D85A30'
C_LTA = '#2E8B57'
C_TR  = '#888780'


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (handles both TL schema and WB schema)
# ─────────────────────────────────────────────────────────────────────────────

def get_district_data(df, district):
    sub = (df[df['district'] == district]
           .sort_values(['year', 'month'])
           .reset_index(drop=True))

    yrs  = sub['year'].values
    mos  = sub['month'].values - 1
    rain = np.maximum(sub['rain_mean'].values.astype(float), 0.001)
    temp = sub['temp_mean'].values.astype(float)

    temp_min = sub['temp_min'].values.astype(float)
    temp_max = sub['temp_max'].values.astype(float)

    lat = float(sub['latitude'].dropna().iloc[0])
    lon = float(sub['longitude'].dropna().iloc[0])

    return yrs, mos, rain, temp, temp_min, temp_max, lat, lon


# ─────────────────────────────────────────────────────────────────────────────
# MODEL RUNNER  (cached by key parameters)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_models(yrs, mos, rain, temp, temp_min, temp_max, lat,
               train_end, n_paths, seed, dlm_q,
               q_mu0, q_gamma, enso_bytes, spi_scale):
    """
    Run all three models and return result dicts.
    Cached: re-runs only when inputs change.
    """
    train_mask = yrs <= train_end

    # ENSO (optional)
    enso_arr = None
    if enso_bytes is not None:
        try:
            from enso_helper import load_and_align as _load_enso
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix='.csv',
                                             delete=False) as tmp:
                tmp.write(enso_bytes)
                tmp_path = tmp.name
            enso_arr, _ = _load_enso(tmp_path, yrs, mos, standardise=True)
            os.unlink(tmp_path)
        except Exception as e:
            st.warning(f"ONI load failed: {e} — running without ENSO.")

    # Rainfall
    R = run_rainfall_kalman(
            rain, mos, yrs, train_mask,
            n_paths=n_paths, seed=seed,
            q_mu0=q_mu0, q_gamma=q_gamma,
            enso=enso_arr)

    # Temperature
    T = run_temperature_kalman(
            temp, mos, yrs, train_mask,
            n_paths=n_paths, seed=seed, q=dlm_q)

    # Drought
    lat_use = lat if not np.isnan(lat) else 20.0
    DR = run_drought_model(
            R, T,
            scale=spi_scale,
            lat_deg=lat_use
    )
    return R, T, DR


# ─────────────────────────────────────────────────────────────────────────────
# AXIS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _axis_labels(years_all, months_all, T_tr, T_te, ctx):
    yr_ctx  = years_all[T_tr - ctx: T_tr]
    mo_ctx  = months_all[T_tr - ctx: T_tr]
    yr_te   = years_all[T_tr:]
    mo_te   = months_all[T_tr:]
    ctx_lbl = [f"{yr_ctx[i]} {MONTH_NAMES[mo_ctx[i]]}" for i in range(ctx)]
    fc_lbl  = [f"{yr_te[i]} {MONTH_NAMES[mo_te[i]]}"   for i in range(T_te)]
    x_ctx   = np.arange(-ctx, 0)
    x_fc    = np.arange(T_te)
    xt      = list(range(-ctx, 0, 3)) + list(range(0, T_te, 3))
    xl      = ([ctx_lbl[i] for i in range(0, ctx, 3)] +
               [fc_lbl[i]  for i in range(0, T_te, 3)])
    return x_ctx, x_fc, xt, xl


def _metrics_box(ax, acc, unit):
    txt = (f"RMSE  = {acc['rmse']:.3f} {unit}\n"
           f"MAE   = {acc['mae']:.3f} {unit}\n"
           f"BIAS  = {acc['bias']:+.3f} {unit}\n"
           f"SKILL = {acc['skill']:.3f}\n"
           f"90%PI = {acc['cov90']:.0f}%")
    ax.text(0.013, 0.975, txt,
            transform=ax.transAxes, fontsize=7.5,
            va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4',
                      fc='white', ec='#bbbbbb', alpha=0.88))


# ─────────────────────────────────────────────────────────────────────────────
# PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def make_rain_paths(R, district, n_paths, ctx_months=18):
    """Plot 1.1 — all rainfall ensemble paths."""
    ctx    = min(ctx_months, R['T_tr'])
    T_te   = R['T_te']
    x_ctx, x_fc, xt, xl = _axis_labels(
        R['years_all'], R['months_all'], R['T_tr'], T_te, ctx)
    lta_fc  = R['lta_mo'][R['mo_te']]
    lta_ctx = R['lta_mo'][R['months_all'][R['T_tr'] - ctx: R['T_tr']]]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.suptitle(f"{district}  —  Plot 1.1: Rainfall Ensemble Paths  "
                 f"({n_paths} paths, Kalman Log-OU)",
                 fontsize=10, fontweight='bold')

    n_show  = min(400, n_paths)
    idx_s   = np.linspace(0, n_paths - 1, n_show, dtype=int)
    for i in idx_s:
        ax.plot(x_fc, R['X_kalman'][i], lw=0.25, color=C_KF, alpha=0.07)

    ax.plot(x_ctx, R['rain_tr'][-ctx:], lw=1.2, color=C_TR,
            ls='--', alpha=0.7, label='Training context')
    ax.axvline(-0.5, lw=1.0, color='black', alpha=0.2, ls=':')
    ax.plot(x_fc, R['fc']['mean'], lw=2.2, color=C_KF,
            zorder=5, label='Ensemble mean')
    ax.scatter(x_fc, R['rain_te'], s=38, color=C_OBS, zorder=6,
               label='Observed', edgecolors='white', linewidths=0.4)
    ax.plot(x_ctx, lta_ctx, lw=1.3, color=C_LTA, ls='-.', alpha=0.85,
            label='LTA (training)')
    ax.plot(x_fc,  lta_fc,  lw=1.3, color=C_LTA, ls='-.', alpha=0.85)

    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=38, ha='right', fontsize=7)
    ax.set_xlim(-ctx, T_te - 0.2); ax.set_ylim(bottom=0)
    ax.set_ylabel('Rainfall (mm/day)', fontsize=9)
    ax.legend(fontsize=7.5, ncol=4, loc='upper left')
    ax.grid(True, alpha=0.18)
    plt.tight_layout()
    return fig, ctx, x_ctx, x_fc, xt, xl, lta_ctx, lta_fc


def make_rain_forecast(R, district, ctx, x_ctx, x_fc, xt, xl,
                       lta_ctx, lta_fc):
    """
    Plot 1.2 — Rainfall forecast vs observed (interactive Plotly).
    Hover shows: period, observed, Kalman mean, 90% PI, LTA.
    """
    T_te     = R['T_te']
    acc      = R['acc']

    # ── Build unified string x-axis ──────────────────────────────────────────
    # ctx labels come from xl up to the split; fc labels are the rest
    n_xt     = len(xt)
    ctx_xl   = [xl[i] for i in range(len(xt)) if xt[i] < 0]
    fc_xl    = [xl[i] for i in range(len(xt)) if xt[i] >= 0]

    # Full label lists for every point
    ctx_labels = [f"{R['years_all'][R['T_tr']-ctx+i]} "
                  f"{MONTH_NAMES[R['months_all'][R['T_tr']-ctx+i]]}"
                  for i in range(ctx)]
    fc_labels  = [f"{R['years_all'][R['T_tr']+i]} "
                  f"{MONTH_NAMES[R['mo_te'][i]]}"
                  for i in range(T_te)]

    x_all    = ctx_labels + fc_labels                # full x label list
    x_ctx_s  = ctx_labels                            # context portion
    x_fc_s   = fc_labels                             # forecast portion

    # ── colour helpers ────────────────────────────────────────────────────────
    def _rgba(hex_col, alpha):
        h  = hex_col.lstrip('#')
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'

    fig = go.Figure()

    # ── 90% PI band (p05–p95) ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fc_s, y=R['fc']['p95'].tolist(),
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip', name='_p95'))

    fig.add_trace(go.Scatter(
        x=x_fc_s, y=R['fc']['p05'].tolist(),
        mode='lines', line=dict(width=0),
        fill='tonexty',
        fillcolor=_rgba(C_KF, 0.12),
        name='5th–95th (90% PI)',
        hoverinfo='skip'))

    # ── 80% PI band (p10–p90) ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fc_s, y=R['fc']['p90'].tolist(),
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip', name='_p90'))

    fig.add_trace(go.Scatter(
        x=x_fc_s, y=R['fc']['p10'].tolist(),
        mode='lines', line=dict(width=0),
        fill='tonexty',
        fillcolor=_rgba(C_KF, 0.20),
        name='10th–90th (80% PI)',
        hoverinfo='skip'))

    # ── Quantile boundary lines ───────────────────────────────────────────────
    for vals, pct, dash in [
            (R['fc']['p95'], '95th', 'dash'),
            (R['fc']['p90'], '90th', 'dot'),
            (R['fc']['p10'], '10th', 'dot'),
            (R['fc']['p05'], '5th',  'dash')]:
        fig.add_trace(go.Scatter(
            x=x_fc_s, y=vals.tolist(),
            mode='lines',
            line=dict(color=C_KF, width=0.9, dash=dash),
            opacity=0.6,
            name=pct,
            hovertemplate=f'<b>{pct}</b>: %{{y:.3f}} mm/day<extra></extra>'))

    # ── Training context ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_ctx_s, y=R['rain_tr'][-ctx:].tolist(),
        mode='lines',
        line=dict(color=C_TR, width=1.3, dash='dash'),
        opacity=0.65,
        name='Training context',
        hovertemplate='<b>%{x}</b><br>Training: %{y:.3f} mm/day<extra></extra>'))

    # ── LTA ──────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_ctx_s + x_fc_s,
        y=lta_ctx.tolist() + lta_fc.tolist(),
        mode='lines',
        line=dict(color=C_LTA, width=1.6, dash='dashdot'),
        opacity=0.9,
        name='LTA (climatology)',
        hovertemplate='<b>%{x}</b><br>LTA: %{y:.3f} mm/day<extra></extra>'))

    # ── Kalman mean ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fc_s, y=R['fc']['mean'].tolist(),
        mode='lines',
        line=dict(color=C_KF, width=2.6),
        name='Kalman mean',
        hovertemplate='<b>%{x}</b><br>Kalman mean: %{y:.3f} mm/day<extra></extra>'))

    # ── Observed points — split by inside/outside 90% PI ─────────────────────
    obs_in_x,  obs_in_y  = [], []
    obs_out_x, obs_out_y = [], []
    obs_text_in, obs_text_out = [], []

    for i in range(T_te):
        o, lo, hi = R['rain_te'][i], R['fc']['p05'][i], R['fc']['p95'][i]
        lbl = (f"<b>{x_fc_s[i]}</b><br>"
               f"Observed: {o:.3f} mm/day<br>"
               f"Kalman mean: {R['fc']['mean'][i]:.3f} mm/day<br>"
               f"90% PI: [{lo:.3f}, {hi:.3f}]<br>"
               f"LTA: {lta_fc[i]:.3f} mm/day")
        if lo <= o <= hi:
            obs_in_x.append(x_fc_s[i]); obs_in_y.append(o)
            obs_text_in.append(lbl)
        else:
            obs_out_x.append(x_fc_s[i]); obs_out_y.append(o)
            obs_text_out.append(lbl)

    if obs_in_x:
        fig.add_trace(go.Scatter(
            x=obs_in_x, y=obs_in_y,
            mode='markers',
            marker=dict(color=C_OBS, size=8,
                        line=dict(color='white', width=1)),
            name='Observed (inside PI)',
            hovertemplate='%{text}<extra></extra>',
            text=obs_text_in))

    if obs_out_x:
        fig.add_trace(go.Scatter(
            x=obs_out_x, y=obs_out_y,
            mode='markers',
            marker=dict(color=C_OBS, size=9,
                        line=dict(color='black', width=1.8)),
            name='Observed (outside 90% PI)',
            hovertemplate='%{text}<extra></extra>',
            text=obs_text_out))

    # ── Train/test vertical boundary ──────────────────────────────────────────
    fig.add_vline(
        x=ctx_labels[-1],
        line=dict(color='black', width=1.0, dash='dot'),
        opacity=0.25)

    # ── Metrics annotation ────────────────────────────────────────────────────
    metrics_txt = (
        f"<b>RMSE</b>  {acc['rmse']:.3f} mm/day<br>"
        f"<b>MAE</b>   {acc['mae']:.3f} mm/day<br>"
        f"<b>BIAS</b>  {acc['bias']:+.3f} mm/day<br>"
        f"<b>SKILL</b> {acc['skill']:.3f}<br>"
        f"<b>90%PI</b> {acc['cov90']:.0f}%")
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.01, y=0.98,
        text=metrics_txt,
        showarrow=False,
        align='left',
        bgcolor='white',
        bordercolor='#cccccc',
        borderwidth=1,
        font=dict(size=11, family='monospace'),
        opacity=0.88)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{district}</b>  —  Plot 1.2: Rainfall Forecast vs Observed"
                 f"  (Kalman Log-OU)",
            font=dict(size=14)),
        yaxis=dict(title='Rainfall (mm/day)', rangemode='tozero',
                   gridcolor='rgba(0,0,0,0.08)'),
        xaxis=dict(
            tickangle=-40,
            tickfont=dict(size=9),
            gridcolor='rgba(0,0,0,0.06)',
            categoryorder='array',
            categoryarray=x_ctx_s + x_fc_s),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01,
            xanchor='left', x=0, font=dict(size=10)),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=460,
        margin=dict(l=60, r=30, t=80, b=100))

    return fig


def make_temp_paths(T_res, district, n_paths, ctx_months=18):
    """Plot 2.1 — all temperature ensemble paths."""
    ctx   = min(ctx_months, T_res['T_tr'])
    T_te  = T_res['T_te']
    x_ctx, x_fc, xt, xl = _axis_labels(
        T_res['years_all'], T_res['months_all'],
        T_res['T_tr'], T_te, ctx)
    lta_fc  = T_res['lta_mo'][T_res['mo_te']]
    lta_ctx = T_res['lta_mo'][T_res['months_all'][T_res['T_tr'] - ctx: T_res['T_tr']]]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.suptitle(f"{district}  —  Plot 2.1: Temperature Ensemble Paths  "
                 f"({n_paths} paths, Kalman DLM-OU)",
                 fontsize=10, fontweight='bold')

    n_show = min(400, n_paths)
    idx_s  = np.linspace(0, n_paths - 1, n_show, dtype=int)
    for i in idx_s:
        ax.plot(x_fc, T_res['paths'][i], lw=0.25, color=C_TMP, alpha=0.07)

    ax.plot(x_ctx, T_res['temp_tr'][-ctx:], lw=1.2, color=C_TR,
            ls='--', alpha=0.7, label='Training context')
    ax.axvline(-0.5, lw=1.0, color='black', alpha=0.2, ls=':')
    ax.plot(x_fc, T_res['fc']['mean'], lw=2.2, color=C_TMP,
            zorder=5, label='Ensemble mean')
    ax.scatter(x_fc, T_res['temp_te'], s=38, color=C_OBS, zorder=6,
               label='Observed', edgecolors='white', linewidths=0.4)
    ax.plot(x_ctx, lta_ctx, lw=1.3, color=C_LTA, ls='-.', alpha=0.85,
            label='LTA (training)')
    ax.plot(x_fc,  lta_fc,  lw=1.3, color=C_LTA, ls='-.', alpha=0.85)

    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=38, ha='right', fontsize=7)
    ax.set_xlim(-ctx, T_te - 0.2)
    ax.set_ylabel('Temperature (°C)', fontsize=9)
    ax.legend(fontsize=7.5, ncol=4, loc='upper left')
    ax.grid(True, alpha=0.18)
    plt.tight_layout()
    return fig, ctx, x_ctx, x_fc, xt, xl, lta_ctx, lta_fc


def make_temp_forecast(T_res, district, ctx, x_ctx, x_fc, xt, xl,
                       lta_ctx, lta_fc):
    """
    Plot 2.2 — Temperature forecast vs observed (interactive Plotly).
    Hover shows: period, observed, Kalman mean, 90% PI, LTA.
    """
    T_te  = T_res['T_te']
    acc   = T_res['acc']

    # ── Build string x-axis ───────────────────────────────────────────────────
    ctx_labels = [f"{T_res['years_all'][T_res['T_tr']-ctx+i]} "
                  f"{MONTH_NAMES[T_res['months_all'][T_res['T_tr']-ctx+i]]}"
                  for i in range(ctx)]
    fc_labels  = [f"{T_res['years_all'][T_res['T_tr']+i]} "
                  f"{MONTH_NAMES[T_res['mo_te'][i]]}"
                  for i in range(T_te)]

    x_ctx_s = ctx_labels
    x_fc_s  = fc_labels

    def _rgba(hex_col, alpha):
        h = hex_col.lstrip('#')
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'

    fig = go.Figure()

    # ── 90% PI band ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fc_s, y=T_res['fc']['p95'].tolist(),
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip', name='_p95'))

    fig.add_trace(go.Scatter(
        x=x_fc_s, y=T_res['fc']['p05'].tolist(),
        mode='lines', line=dict(width=0),
        fill='tonexty',
        fillcolor=_rgba(C_TMP, 0.12),
        name='5th–95th (90% PI)',
        hoverinfo='skip'))

    # ── 80% PI band ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fc_s, y=T_res['fc']['p90'].tolist(),
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip', name='_p90'))

    fig.add_trace(go.Scatter(
        x=x_fc_s, y=T_res['fc']['p10'].tolist(),
        mode='lines', line=dict(width=0),
        fill='tonexty',
        fillcolor=_rgba(C_TMP, 0.20),
        name='10th–90th (80% PI)',
        hoverinfo='skip'))

    # ── Quantile boundary lines ───────────────────────────────────────────────
    for vals, pct, dash in [
            (T_res['fc']['p95'], '95th', 'dash'),
            (T_res['fc']['p90'], '90th', 'dot'),
            (T_res['fc']['p10'], '10th', 'dot'),
            (T_res['fc']['p05'], '5th',  'dash')]:
        fig.add_trace(go.Scatter(
            x=x_fc_s, y=vals.tolist(),
            mode='lines',
            line=dict(color=C_TMP, width=0.9, dash=dash),
            opacity=0.6,
            name=pct,
            hovertemplate=f'<b>{pct}</b>: %{{y:.2f}} °C<extra></extra>'))

    # ── Training context ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_ctx_s, y=T_res['temp_tr'][-ctx:].tolist(),
        mode='lines',
        line=dict(color=C_TR, width=1.3, dash='dash'),
        opacity=0.65,
        name='Training context',
        hovertemplate='<b>%{x}</b><br>Training: %{y:.2f} °C<extra></extra>'))

    # ── LTA ──────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_ctx_s + x_fc_s,
        y=lta_ctx.tolist() + lta_fc.tolist(),
        mode='lines',
        line=dict(color=C_LTA, width=1.6, dash='dashdot'),
        opacity=0.9,
        name='LTA (climatology)',
        hovertemplate='<b>%{x}</b><br>LTA: %{y:.2f} °C<extra></extra>'))

    # ── Kalman mean ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_fc_s, y=T_res['fc']['mean'].tolist(),
        mode='lines',
        line=dict(color=C_TMP, width=2.6),
        name='Kalman mean',
        hovertemplate='<b>%{x}</b><br>Kalman mean: %{y:.2f} °C<extra></extra>'))

    # ── Observed points — colour by inside/outside PI ─────────────────────────
    obs_in_x,  obs_in_y  = [], []
    obs_out_x, obs_out_y = [], []
    obs_text_in, obs_text_out = [], []

    for i in range(T_te):
        o  = T_res['temp_te'][i]
        lo = T_res['fc']['p05'][i]
        hi = T_res['fc']['p95'][i]
        lbl = (f"<b>{x_fc_s[i]}</b><br>"
               f"Observed: {o:.2f} °C<br>"
               f"Kalman mean: {T_res['fc']['mean'][i]:.2f} °C<br>"
               f"90% PI: [{lo:.2f}, {hi:.2f}] °C<br>"
               f"LTA: {lta_fc[i]:.2f} °C")
        if lo <= o <= hi:
            obs_in_x.append(x_fc_s[i]); obs_in_y.append(o)
            obs_text_in.append(lbl)
        else:
            obs_out_x.append(x_fc_s[i]); obs_out_y.append(o)
            obs_text_out.append(lbl)

    if obs_in_x:
        fig.add_trace(go.Scatter(
            x=obs_in_x, y=obs_in_y,
            mode='markers',
            marker=dict(color=C_OBS, size=8, symbol='triangle-up',
                        line=dict(color='white', width=1)),
            name='Observed (inside PI)',
            hovertemplate='%{text}<extra></extra>',
            text=obs_text_in))

    if obs_out_x:
        fig.add_trace(go.Scatter(
            x=obs_out_x, y=obs_out_y,
            mode='markers',
            marker=dict(color=C_OBS, size=9, symbol='triangle-up',
                        line=dict(color='black', width=1.8)),
            name='Observed (outside 90% PI)',
            hovertemplate='%{text}<extra></extra>',
            text=obs_text_out))

    # ── Train/test boundary ───────────────────────────────────────────────────
    fig.add_vline(
        x=ctx_labels[-1],
        line=dict(color='black', width=1.0, dash='dot'),
        opacity=0.25)

    # ── Metrics annotation ────────────────────────────────────────────────────
    metrics_txt = (
        f"<b>RMSE</b>  {acc['rmse']:.3f} °C<br>"
        f"<b>MAE</b>   {acc['mae']:.3f} °C<br>"
        f"<b>BIAS</b>  {acc['bias']:+.3f} °C<br>"
        f"<b>SKILL</b> {acc['skill']:.3f}<br>"
        f"<b>90%PI</b> {acc['cov90']:.0f}%")
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.01, y=0.98,
        text=metrics_txt,
        showarrow=False,
        align='left',
        bgcolor='white',
        bordercolor='#cccccc',
        borderwidth=1,
        font=dict(size=11, family='monospace'),
        opacity=0.88)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{district}</b>  —  Plot 2.2: Temperature Forecast vs Observed"
                 ,
            font=dict(size=14)),
        yaxis=dict(title='Temperature (°C)',
                   gridcolor='rgba(0,0,0,0.08)'),
        xaxis=dict(
            tickangle=-40,
            tickfont=dict(size=9),
            gridcolor='rgba(0,0,0,0.06)',
            categoryorder='array',
            categoryarray=x_ctx_s + x_fc_s),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01,
            xanchor='left', x=0, font=dict(size=10)),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=460,
        margin=dict(l=60, r=30, t=80, b=100))

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📂 Data")

    # Preload climate_master.xlsx file
    df_all = None
    try:
        with st.spinner("Loading climate data..."):
            df_all = pd.read_excel("Data/climate_master.xlsx")
            # Normalize column names to match expected format
            df_all.columns = [c.strip().lower() for c in df_all.columns]
            # Rename Latitude/Longitude to lowercase
            df_all.rename(columns={'latitude': 'latitude', 'longitude': 'longitude'}, inplace=True)

            # Ensure required columns exist
            required = ['state', 'district', 'year', 'month', 'rain_mean', 'temp_mean']
            missing = [c for c in required if c not in df_all.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                df_all = None
            else:
                # Add lat/lon columns if absent (though they should be present)
                for col in ('latitude', 'longitude'):
                    if col not in df_all.columns:
                        df_all[col] = np.nan

                # Drop rows missing key values
                df_all.dropna(subset=['year', 'month', 'rain_mean', 'temp_mean'], inplace=True)
                df_all['year'] = df_all['year'].astype(int)
                df_all['month'] = df_all['month'].astype(int)

        if df_all is not None:
            st.success(f"✓ Preloaded: {df_all['state'].nunique()} states · "
                       f"{df_all['district'].nunique()} districts · "
                       f"{len(df_all):,} rows")
        else:
            st.error("Failed to load climate data.")

    except Exception as e:
        st.error(f"Error loading climate data: {e}")
        df_all = None

    st.markdown("### 📍 Location")
    if df_all is not None:
        states_avail = sorted(df_all['state'].unique())
        sel_state    = st.selectbox("State", states_avail,
                                    label_visibility="collapsed")
        st.caption(f"State")

        dists_avail  = sorted(
            df_all[df_all['state'] == sel_state]['district'].unique())
        sel_district = st.selectbox("District", dists_avail,
                                    label_visibility="collapsed")
        st.caption(f"District")

        sub      = df_all[df_all['district'] == sel_district]
        yr_min   = int(sub['year'].min()); yr_max = int(sub['year'].max())
        lat_vals = sub['latitude'].dropna()
        lon_vals = sub['longitude'].dropna()
        lat_str  = f"{lat_vals.iloc[0]:.2f}°N" if len(lat_vals) > 0 else "—"
        lon_str  = f"{lon_vals.iloc[0]:.2f}°E" if len(lon_vals) > 0 else "—"
        st.markdown(
            f'<div style="font-size:11px;color:#6B7280;'
            f'background:#F0F4F8;border-radius:6px;padding:6px 10px;'
            f'margin-top:4px;line-height:1.7;">'
            f'📅 {yr_min}–{yr_max} &nbsp;·&nbsp; '
            f'🗺 {lat_str}, {lon_str} &nbsp;·&nbsp; '
            f'{len(sub)} months</div>', unsafe_allow_html=True)
    else:
        sel_state    = None
        sel_district = None
        st.info("Upload a file to begin.")

    st.markdown("### ⚙️ Parameters")
    train_end = st.number_input(
        "Training end year", min_value=2015, max_value=2030,
        value=2023, step=1)
    n_paths = st.select_slider(
        "Ensemble paths",
        options=[500, 1000, 2000, 3000, 4000, 5000], value=3000)
    spi_scale = st.radio(
        "SPI/SPEI scale", options=[1, 3, 6, 12],
        index=1, horizontal=True)

    with st.expander("Advanced parameters"):
        seed    = st.number_input("Random seed", value=42, step=1)
        dlm_q   = st.number_input("Temp DLM noise (q)", value=0.05,
                                   format="%.4f", step=0.01)
        q_mu0   = st.number_input("Rain Kalman Q_mu0", value=1e-4,
                                   format="%.2e", step=1e-5)
        q_gamma = st.number_input("Rain Kalman Q_gamma", value=1e-7,
                                   format="%.2e", step=1e-8)

    st.markdown("### 🌊 ENSO (optional)")
    oni_file = st.file_uploader("ONI CSV", type=['csv'],
                                 label_visibility="collapsed",
                                 help="NOAA CPC ONI CSV: SEAS, YR, TOTAL, ANOM")

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    run_btn = st.button(
        "▶  Run Forecast",
        disabled=(sel_district is None),
        use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────

# ── Dark header bar ───────────────────────────────────────────────────────────
district_label = sel_district if sel_district else "—"
state_label    = sel_state    if sel_state    else "—"
st.markdown(f"""
<div class="cf-header">
  <div class="cf-header-logo">⚡ Climate Hazard Forecasting</div>
  <div class="cf-header-sep"></div>
  <div class="cf-header-district">{district_label} · {state_label}</div>
  <div class="cf-header-badge">Kalman Log-OU · DLM-OU · SPI/SPEI</div>
  <div class="cf-header-badge">DiCRA · NABARD 2026</div>
</div>
""", unsafe_allow_html=True)

# ── Gate on data ──────────────────────────────────────────────────────────────
if df_all is None:
    st.stop()

if not run_btn:
    # Ready-to-run state — show data summary
    st.markdown('<div class="cf-strip">📁 Data loaded — select a district and click ▶ Run Forecast to generate results.</div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("States loaded",    df_all['state'].nunique())
    col2.metric("Districts loaded", df_all['district'].nunique())
    col3.metric("Total months",     f"{len(df_all):,}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# RUN MODELS
# ─────────────────────────────────────────────────────────────────────────────

yrs, mos, rain, temp,temp_min, temp_max, lat, lon = get_district_data(df_all, sel_district)
train_mask = yrs <= train_end
T_tr_total = int(train_mask.sum())
T_te_total = int((~train_mask).sum())

if T_tr_total < 24:
    st.error(f"Only {T_tr_total} training months for **{sel_district}** "
             f"(train end = {train_end}). Need ≥ 24. "
             f"Try lowering the training end year.")
    st.stop()

if T_te_total < 3:
    st.error(f"Only {T_te_total} forecast months. "
             f"Try raising the training end year.")
    st.stop()

lat_display = f"{lat:.2f}°N" if not np.isnan(lat) else "N/A (using 20.0°N)"
lat_use     = lat if not np.isnan(lat) else 20.0
oni_bytes   = oni_file.read() if oni_file is not None else None

status_box = st.status(
    f"Running forecast for **{sel_district}** …", expanded=True)
with status_box:
    st.write("📐 Calibrating Log-OU SDE (rainfall) …")
    st.write("📐 Fitting DLM-OU (temperature) …")
    st.write("🎲 Generating Monte-Carlo ensembles …")
    st.write("🌵 Computing SPI / SPEI …")
    try:
        R, T, DR = run_models(
            yrs, mos, rain, temp, temp_min, temp_max, lat_use,
            train_end, n_paths, int(seed),
            dlm_q, q_mu0, q_gamma,
            oni_bytes, spi_scale)
        status_box.update(label=f"✅ Forecast complete — {sel_district}",
                          state='complete', expanded=False)
    except Exception as e:
        status_box.update(label="❌ Model failed", state='error')
        st.exception(e)
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# INFO STRIP + METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f'<div class="cf-strip">'
    f'📍 <strong>{sel_district}</strong>, {sel_state} &nbsp;·&nbsp; '
    f'Train {yrs[train_mask][0]}–{yrs[train_mask][-1]} ({T_tr_total} mo) &nbsp;·&nbsp; '
    f'Forecast {yrs[~train_mask][0]}–{yrs[~train_mask][-1]} ({T_te_total} mo) &nbsp;·&nbsp; '
    f'Lat {lat_display} &nbsp;·&nbsp; {n_paths:,} paths'
    f'</div>', unsafe_allow_html=True)

# Drought risk badge colour
dr_pct = DR['spi_prob']['moderate'].mean()
dr_col = '#A32D2D' if dr_pct > 50 else '#854F0B' if dr_pct > 25 else '#0F6E56'

st.markdown(f"""
<div class="cf-metric-row">
  <div class="cf-metric">
    <div class="cf-metric-label">Rain RMSE</div>
    <div class="cf-metric-value">{R['acc']['rmse']:.3f}<span class="cf-metric-unit">mm/day</span></div>
  </div>
  <div class="cf-metric">
    <div class="cf-metric-label">Rain Skill</div>
    <div class="cf-metric-value">{R['acc']['skill']:.3f}</div>
  </div>
  <div class="cf-metric">
    <div class="cf-metric-label">Rain 90% PI</div>
    <div class="cf-metric-value">{R['acc']['cov90']:.0f}<span class="cf-metric-unit">%</span></div>
  </div>
  <div class="cf-metric purple">
    <div class="cf-metric-label">Temp RMSE</div>
    <div class="cf-metric-value">{T['acc']['rmse']:.3f}<span class="cf-metric-unit">°C</span></div>
  </div>
  <div class="cf-metric purple">
    <div class="cf-metric-label">Temp Skill</div>
    <div class="cf-metric-value">{T['acc']['skill']:.3f}</div>
  </div>
  <div class="cf-metric amber" style="border-left-color:{dr_col};">
    <div class="cf-metric-label">SPI-{spi_scale} Drought P</div>
    <div class="cf-metric-value" style="color:{dr_col};">{dr_pct:.0f}<span class="cf-metric-unit">%</span></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="font-size:11px;color:#9CA3AF;margin-bottom:4px;
            display:flex;align-items:center;gap:6px;">
  <span style="font-size:14px;">👇</span>
  Click a tab to explore results
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🌧  Rainfall",
    "🌡  Temperature",
    f"🏜  Drought  (SPI/SPEI-{spi_scale})",
    "📊  Summary",
])


# ══ Tab 1: Rainfall ══════════════════════════════════════════════════════════
with tab1:
    # Generate path data (used for forecast plot even though path plot is removed)
    fig11, ctx_r, x_ctx_r, x_fc_r, xt_r, xl_r, lta_ctx_r, lta_fc_r = \
        make_rain_paths(R, sel_district, n_paths)
    plt.close(fig11)

    st.markdown('<div class="cf-section">Forecast vs observed — interactive</div>',
                unsafe_allow_html=True)
    fig12 = make_rain_forecast(R, sel_district,
                               ctx_r, x_ctx_r, x_fc_r, xt_r, xl_r,
                               lta_ctx_r, lta_fc_r)
    st.plotly_chart(fig12, use_container_width=True)

    with st.expander("Rainfall metrics detail"):
        st.table(pd.DataFrame({
            'Metric': ['RMSE (mm/day)', 'MAE (mm/day)', 'BIAS (mm/day)',
                       'Skill score', '90% PI coverage',
                       'θ (reversion)', 'AR(1)', 'γ (trend/dec)'],
            'Value':  [f"{R['acc']['rmse']:.4f}",
                       f"{R['acc']['mae']:.4f}",
                       f"{R['acc']['bias']:+.4f}",
                       f"{R['acc']['skill']:.4f}",
                       f"{R['acc']['cov90']:.1f}%",
                       f"{R['sde']['theta']:.4f}",
                       f"{R['ar1']:.4f}",
                       f"{R['sde']['trend']:+.4f}"],
        }).set_index('Metric'))


# ══ Tab 2: Temperature ═══════════════════════════════════════════════════════
with tab2:
    # Generate path data (used for forecast plot even though path plot is removed)
    fig21, ctx_t, x_ctx_t, x_fc_t, xt_t, xl_t, lta_ctx_t, lta_fc_t = \
        make_temp_paths(T, sel_district, n_paths)
    plt.close(fig21)

    st.markdown('<div class="cf-section">Forecast vs observed — interactive</div>',
                unsafe_allow_html=True)
    fig22 = make_temp_forecast(T, sel_district,
                               ctx_t, x_ctx_t, x_fc_t, xt_t, xl_t,
                               lta_ctx_t, lta_fc_t)
    st.plotly_chart(fig22, use_container_width=True)

    with st.expander("Temperature metrics detail"):
        st.table(pd.DataFrame({
            'Metric': ['RMSE (°C)', 'MAE (°C)', 'BIAS (°C)',
                       'Skill score', '90% PI coverage',
                       'θ (reversion)', 'Half-life (mo)', 'σ', 'DLM q'],
            'Value':  [f"{T['acc']['rmse']:.4f}",
                       f"{T['acc']['mae']:.4f}",
                       f"{T['acc']['bias']:+.4f}",
                       f"{T['acc']['skill']:.4f}",
                       f"{T['acc']['cov90']:.1f}%",
                       f"{T['dlm'].theta:.4f}",
                       f"{T['dlm'].half_life():.2f}",
                       f"{T['dlm'].sigma:.4f}",
                       f"{dlm_q}"],
        }).set_index('Metric'))


# ══ Tab 3: Drought ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        f'<div class="cf-section">'
        f'SPI-{spi_scale} & SPEI-{spi_scale} · Gamma & log-logistic · '
        f'Thornthwaite PET · {lat_display}'
        f'</div>', unsafe_allow_html=True)

    yr_ctx_dr = R['years_all'][R['T_tr'] - DR['ctx']: R['T_tr']]
    mo_ctx_dr = R['months_all'][R['T_tr'] - DR['ctx']: R['T_tr']]
    fig3 = plot_drought(
        DR, sel_district,
        x_fc   = np.arange(R['T_te']),
        xt     = xt_r,
        xl     = xl_r,
        yr_te  = R['years_all'][R['T_tr']:],
        mo_te  = R['mo_te'],
        yr_ctx = yr_ctx_dr,
        mo_ctx = mo_ctx_dr,
    )
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    with st.expander("Monthly drought risk detail"):
        yr_te_arr = R['years_all'][R['T_tr']:]
        mo_te_arr = R['mo_te']
        rows = []
        for i in range(R['T_te']):
            rows.append({
                'Period':        f"{yr_te_arr[i]} {MONTH_NAMES[mo_te_arr[i]]}",
                'SPI mean':      f"{DR['spi_fc']['mean'][i]:.2f}",
                'SPI P(mod %)':  f"{DR['spi_prob']['moderate'][i]:.0f}",
                'SPI P(sev %)':  f"{DR['spi_prob']['severe'][i]:.0f}",
                'SPI P(ext %)':  f"{DR['spi_prob']['extreme'][i]:.0f}",
                'SPEI mean':     f"{DR['spei_fc']['mean'][i]:.2f}",
                'SPEI P(mod %)': f"{DR['spei_prob']['moderate'][i]:.0f}",
                'SPEI P(sev %)': f"{DR['spei_prob']['severe'][i]:.0f}",
                'SPEI P(ext %)': f"{DR['spei_prob']['extreme'][i]:.0f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index('Period'),
                     use_container_width=True)


# ══ Tab 4: Summary ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="cf-section">Run configuration & forecast metrics</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.table(pd.DataFrame({
            'Parameter': ['District', 'State', 'Latitude', 'Train period',
                          'Forecast period', 'Ensemble paths',
                          'SPI/SPEI scale', 'ENSO covariate'],
            'Value':     [sel_district, sel_state, lat_display,
                          f"{yrs[train_mask][0]}–{yrs[train_mask][-1]} ({T_tr_total} mo)",
                          f"{yrs[~train_mask][0]}–{yrs[~train_mask][-1]} ({T_te_total} mo)",
                          f"{n_paths:,}",
                          f"{spi_scale} month(s)",
                          "Yes" if oni_bytes else "No"],
        }).set_index('Parameter'))
    with c2:
        st.table(pd.DataFrame({
            'Metric':      ['RMSE', 'MAE', 'BIAS', 'Skill', '90% PI cov.'],
            'Rainfall':    [f"{R['acc']['rmse']:.3f} mm/day",
                            f"{R['acc']['mae']:.3f} mm/day",
                            f"{R['acc']['bias']:+.3f} mm/day",
                            f"{R['acc']['skill']:.3f}",
                            f"{R['acc']['cov90']:.0f}%"],
            'Temperature': [f"{T['acc']['rmse']:.3f} °C",
                            f"{T['acc']['mae']:.3f} °C",
                            f"{T['acc']['bias']:+.3f} °C",
                            f"{T['acc']['skill']:.3f}",
                            f"{T['acc']['cov90']:.0f}%"],
        }).set_index('Metric'))

    # ── Drought risk visual panels ────────────────────────────────────────────
    st.markdown('<div class="cf-section">Drought risk forecast — colour-coded probability panels</div>',
                unsafe_allow_html=True)

    def _obs_category(val):
        """Classify a single observed SPI/SPEI value into colour + label."""
        if np.isnan(val):
            return '#E5E7EB', '#6B7280', 'N/A'
        elif val >= 1.0:
            return '#1D9E75', 'white', 'Wet'
        elif val >= -1.0:
            return '#D1FAE5', '#065F46', 'Nrm'
        elif val >= -1.5:
            return '#F59E0B', '#4A2800', 'Mod'
        elif val >= -2.0:
            return '#D63A0C', 'white', 'Sev'
        else:
            return '#8B0000', 'white', 'Ext'

    def _drought_gauge_html(index_name, color_accent,
                             mod_p, sev_p, ext_p,
                             periods, obs_vals):
        """
        Build a self-contained HTML card with:
          - Coloured header
          - Three mean probability bars (forecast)
          - Forecast risk timeline (probabilistic — colour = highest category > 30%)
          - Observed timeline (deterministic — colour = actual SPI/SPEI category)
        """
        n = len(periods)

        # ── Forecast timeline cells ───────────────────────────────────────────
        fc_cells = ''
        for p, m, s, e in zip(periods,
                               mod_p.tolist(), sev_p.tolist(), ext_p.tolist()):
            if e >= 30:
                bg, fc = '#8B0000', 'white'
            elif s >= 30:
                bg, fc = '#D63A0C', 'white'
            elif m >= 30:
                bg, fc = '#F0A500', '#4A2800'
            else:
                bg, fc = '#E1F5EE', '#0F6E56'
            mon = p.split(' ')[1][:3]
            yr  = str(p.split(' ')[0])[2:]
            fc_cells += (
                f'<div style="flex:1;min-width:0;text-align:center;'
                f'background:{bg};border-radius:4px;padding:4px 2px;margin:1px;">'
                f'<div style="font-size:9px;color:{fc};font-weight:600;">{mon}</div>'
                f'<div style="font-size:8px;color:{fc};opacity:.8;">\'{yr}</div>'
                f'</div>')

        # ── Observed timeline cells ───────────────────────────────────────────
        obs_cells = ''
        for p, v in zip(periods, obs_vals):
            bg, fc, cat = _obs_category(float(v) if not np.isnan(v) else np.nan)
            mon = p.split(' ')[1][:3]
            yr  = str(p.split(' ')[0])[2:]
            obs_cells += (
                f'<div style="flex:1;min-width:0;text-align:center;'
                f'background:{bg};border-radius:4px;padding:4px 2px;margin:1px;">'
                f'<div style="font-size:9px;color:{fc};font-weight:600;">{mon}</div>'
                f'<div style="font-size:8px;color:{fc};opacity:.8;">{cat}</div>'
                f'</div>')

        # ── Mean probabilities ────────────────────────────────────────────────
        m_avg = float(np.mean(mod_p))
        s_avg = float(np.mean(sev_p))
        e_avg = float(np.mean(ext_p))

        # ── Forecast category counts (dominant category per month) ────────────
        fc_n_low = fc_n_mod = fc_n_sev = fc_n_ext = 0
        for m, s, e in zip(mod_p.tolist(), sev_p.tolist(), ext_p.tolist()):
            if e >= 30:   fc_n_ext += 1
            elif s >= 30: fc_n_sev += 1
            elif m >= 30: fc_n_mod += 1
            else:         fc_n_low += 1

        # ── Observed category counts ──────────────────────────────────────────
        obs_clean = [v for v in obs_vals if not np.isnan(v)]
        n_obs     = len(obs_clean) or 1
        n_mod = sum(1 for v in obs_clean if -1.5 <= v < -1.0)
        n_sev = sum(1 for v in obs_clean if -2.0 <= v < -1.5)
        n_ext = sum(1 for v in obs_clean if v < -2.0)
        n_nrm = n_obs - n_mod - n_sev - n_ext

        return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0;
       font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
  body {{ background:transparent; padding:4px; }}
  .lbl {{ font-size:10px; font-weight:700; color:#9CA3AF;
          text-transform:uppercase; letter-spacing:.07em; margin-bottom:5px; }}
  .strip {{ display:flex; gap:2px; flex-wrap:nowrap; margin-bottom:4px; }}
  .legend {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }}
  .leg-item {{ font-size:10px; }}
  .count-row {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:12px; }}
  .count-pill {{ font-size:10px; font-weight:600; padding:2px 8px;
                 border-radius:20px; }}
</style></head><body>
<div style="background:white;border:1.5px solid {color_accent};
            border-radius:10px;overflow:hidden;">

  <div style="background:{color_accent};padding:10px 14px;
              display:flex;align-items:center;gap:8px;">
    <div style="font-size:13px;font-weight:600;color:white;">{index_name}</div>
    <div style="font-size:11px;color:rgba(255,255,255,.7);flex:1;">
      {spi_scale}-month · {n} months</div>
  </div>

  <div style="padding:13px;">

    <div class="lbl">Forecast — mean drought probability</div>

    <div style="margin-bottom:7px;">
      <div style="display:flex;justify-content:space-between;
                  font-size:11px;font-weight:600;margin-bottom:3px;">
        <span style="color:#374151;">🟡 Moderate (&lt; −1.0)</span>
        <span style="color:#B45309;">{m_avg:.0f}%</span></div>
      <div style="background:#FEF3C7;border-radius:4px;height:9px;">
        <div style="width:{min(m_avg,100):.0f}%;background:#F59E0B;
                    border-radius:4px;height:9px;"></div></div>
    </div>

    <div style="margin-bottom:7px;">
      <div style="display:flex;justify-content:space-between;
                  font-size:11px;font-weight:600;margin-bottom:3px;">
        <span style="color:#374151;">🟠 Severe (&lt; −1.5)</span>
        <span style="color:#B91C1C;">{s_avg:.0f}%</span></div>
      <div style="background:#FEE2E2;border-radius:4px;height:9px;">
        <div style="width:{min(s_avg,100):.0f}%;background:#D63A0C;
                    border-radius:4px;height:9px;"></div></div>
    </div>

    <div style="margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;
                  font-size:11px;font-weight:600;margin-bottom:3px;">
        <span style="color:#374151;">🔴 Extreme (&lt; −2.0)</span>
        <span style="color:#7F1D1D;">{e_avg:.0f}%</span></div>
      <div style="background:#FEE2E2;border-radius:4px;height:9px;">
        <div style="width:{min(e_avg,100):.0f}%;background:#8B0000;
                    border-radius:4px;height:9px;"></div></div>
    </div>

    <div class="lbl">📊 Forecast risk — monthly timeline</div>
    <div class="strip">{fc_cells}</div>
    <div class="legend">
      <span class="leg-item" style="color:#0F6E56;">■ Low</span>
      <span class="leg-item" style="color:#B45309;">■ Moderate</span>
      <span class="leg-item" style="color:#D63A0C;">■ Severe</span>
      <span class="leg-item" style="color:#8B0000;">■ Extreme</span>
    </div>
    <div class="count-row">
      <span class="count-pill"
            style="background:#D1FAE5;color:#065F46;">
        Low: {fc_n_low}</span>
      <span class="count-pill"
            style="background:#FEF3C7;color:#B45309;">
        Moderate: {fc_n_mod}</span>
      <span class="count-pill"
            style="background:#FEE2E2;color:#B91C1C;">
        Severe: {fc_n_sev}</span>
      <span class="count-pill"
            style="background:#FEE2E2;color:#7F1D1D;">
        Extreme: {fc_n_ext}</span>
    </div>

    <div style="border-top:1px dashed #E5E7EB;margin:4px 0 10px;"></div>

    <div class="lbl">🔎 Observed drought — actual SPI/SPEI category</div>
    <div class="strip">{obs_cells}</div>
    <div class="legend">
      <span class="leg-item" style="color:#1D9E75;">■ Wet</span>
      <span class="leg-item" style="color:#065F46;">■ Normal</span>
      <span class="leg-item" style="color:#B45309;">■ Mod</span>
      <span class="leg-item" style="color:#D63A0C;">■ Sev</span>
      <span class="leg-item" style="color:#8B0000;">■ Ext</span>
    </div>

    <div class="count-row">
      <span class="count-pill"
            style="background:#D1FAE5;color:#065F46;">
        Normal/Wet: {n_nrm}</span>
      <span class="count-pill"
            style="background:#FEF3C7;color:#B45309;">
        Moderate: {n_mod}</span>
      <span class="count-pill"
            style="background:#FEE2E2;color:#B91C1C;">
        Severe: {n_sev}</span>
      <span class="count-pill"
            style="background:#FEE2E2;color:#7F1D1D;">
        Extreme: {n_ext}</span>
    </div>

  </div>
</div>
</body></html>"""

    # Build period labels
    yr_te_arr = R['years_all'][R['T_tr']:]
    mo_te_arr = R['mo_te']
    periods   = [f"{yr_te_arr[i]} {MONTH_NAMES[mo_te_arr[i]]}"
                 for i in range(R['T_te'])]

    # Replace NaN in observed arrays with np.nan for safe processing
    spi_obs_safe  = np.where(np.isfinite(DR['spi_obs']),
                              DR['spi_obs'], np.nan)
    spei_obs_safe = np.where(np.isfinite(DR['spei_obs']),
                              DR['spei_obs'], np.nan)

    spi_html  = _drought_gauge_html(
        f"SPI-{spi_scale}  ·  Standardised Precipitation Index",
        "#1D4E89",
        DR['spi_prob']['moderate'],
        DR['spi_prob']['severe'],
        DR['spi_prob']['extreme'],
        periods,
        spi_obs_safe)

    spei_html = _drought_gauge_html(
        f"SPEI-{spi_scale}  ·  Standardised Precipitation-Evapotranspiration Index",
        "#7B2D8B",
        DR['spei_prob']['moderate'],
        DR['spei_prob']['severe'],
        DR['spei_prob']['extreme'],
        periods,
        spei_obs_safe)

    col_spi, col_spei = st.columns(2)
    with col_spi:
        components.html(spi_html, height=560, scrolling=False)
    with col_spei:
        components.html(spei_html, height=560, scrolling=False)

    st.markdown('<div class="cf-section" style="margin-top:1.4rem;">Fitted model parameters</div>',
                unsafe_allow_html=True)
    st.table(pd.DataFrame({
        'Parameter':   ['θ (reversion speed)', 'AR(1) = exp(−θ)',
                        'γ (trend/decade)', 'σ (diffusion)',
                        'DLM half-life (mo)'],
        'Rainfall':    [f"{R['sde']['theta']:.4f}", f"{R['ar1']:.4f}",
                        f"{R['sde']['trend']:+.4f}", '—', '—'],
        'Temperature': ['—', '—', '—',
                        f"{T['dlm'].sigma:.4f}",
                        f"{T['dlm'].half_life():.2f}"],
    }).set_index('Parameter'))