"""
drought_model.py
================
Computes SPI and SPEI from Kalman forecast ensembles.

Method
------
  SPI  — Gamma distribution fitted to rolling accumulated rainfall
          (training period, per calendar month).
          Applied to all 4000 rainfall paths → 4000 SPI paths.

  SPEI — Log-logistic distribution fitted to rolling water balance
          D = rain − PET  (training period, per calendar month).
          PET computed via Thornthwaite method from temperature.
          Applied using i-th rainfall path + mean temperature forecast
          → 4000 SPEI paths.

SPI timescales
--------------
  scale=1  : short-term / agricultural drought
  scale=3  : seasonal drought (most common)
  scale=6  : medium-term water supply
  scale=12 : hydrological / groundwater drought

Public API
----------
run_drought_model(R, T, scale=3, lat_deg=17.0)  ->  dict

plot_drought(D, district, x_fc, xt, xl, scale)   ->  Figure
"""

import warnings
import numpy as np
from scipy.stats import gamma as sp_gamma, fisk as sp_fisk, norm as sp_norm

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── month constants ──────────────────────────────────────────────────────────
_DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                           dtype=float)

# SPI/SPEI drought thresholds (standard WMO/McKee 1993 categories)
THRESHOLDS = {
    'extreme_wet':    2.00,
    'severe_wet':     1.50,
    'moderate_wet':   1.00,
    'near_normal_hi': 0.00,
    'near_normal_lo': -1.00,
    'moderate':       -1.00,
    'severe':         -1.50,
    'extreme':        -2.00,
}

# ─────────────────────────────────────────────────────────────────────────────
# THORNTHWAITE PET
# ─────────────────────────────────────────────────────────────────────────────

def _daylight_hours(month_0idx, lat_deg):
    """
    Approximate mean daylight hours for a calendar month at a given latitude.
    Uses solar declination + sunset hour angle formula.
    """
    # Approximate day-of-year at mid-month
    mid_doy = [15, 46, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344]
    doy     = mid_doy[int(month_0idx) % 12]
    lat_r   = np.radians(lat_deg)
    decl    = np.radians(23.45 * np.sin(np.radians(360 * (284 + doy) / 365)))
    cos_ha  = -np.tan(lat_r) * np.tan(decl)
    cos_ha  = np.clip(cos_ha, -1.0, 1.0)
    ha_deg  = np.degrees(np.arccos(cos_ha))
    return 2.0 * ha_deg / 15.0


def thornthwaite_pet(temp_C, months_0idx, lat_deg):
    """
    Compute monthly PET (mm/month) using the Thornthwaite (1948) method.

    Parameters
    ----------
    temp_C     : (N,) array, monthly mean temperature in °C
    months_0idx: (N,) int array, 0-indexed calendar months
    lat_deg    : float, latitude in degrees (positive = North)

    Returns
    -------
    pet : (N,) array, PET in mm/month
    """
    T    = np.maximum(temp_C, 0.0)   # PET = 0 for T ≤ 0
    N    = len(T)

    # Annual heat index I (computed from monthly means of the series)
    # Use a 12-month rolling mean grouped by calendar month for robustness
    I_mo = np.zeros(12)
    for m in range(12):
        mask = months_0idx == m
        if mask.any():
            I_mo[m] = np.maximum(T[mask].mean() / 5.0, 0.0) ** 1.514
    I = I_mo.sum()
    if I < 1e-6:
        return np.zeros(N)

    a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239

    pet = np.zeros(N)
    for i in range(N):
        m  = int(months_0idx[i]) % 12
        Ti = T[i]
        if Ti <= 0 or I <= 0:
            pet[i] = 0.0
        else:
            pet_unadj = 16.0 * (10.0 * Ti / I) ** a        # mm/month (30-day)
            N_hrs     = _daylight_hours(m, lat_deg)          # daylight hours
            NDays     = _DAYS_IN_MONTH[m]
            K_m       = (N_hrs / 12.0) * (NDays / 30.0)     # correction factor
            pet[i]    = pet_unadj * K_m
    return pet


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING ACCUMULATION
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_sum(x, scale):
    """
    Rolling sum of x over `scale` steps.
    Result[t] = sum(x[t-scale+1 : t+1]).
    First (scale-1) values are NaN.
    """
    n   = len(x)
    out = np.full(n, np.nan)
    cs  = np.cumsum(x)
    out[scale - 1:] = cs[scale - 1:]
    if scale > 1:
        out[scale - 1:] -= np.r_[0.0, cs[:n - scale]]
    return out


def _rolling_sum_2d(X, scale):
    """
    Row-wise rolling sum over axis=1 for matrix X (n_paths, T).
    Returns (n_paths, T), first (scale-1) cols are NaN.
    """
    cs  = np.cumsum(X, axis=1)
    out = np.full_like(cs, np.nan)
    out[:, scale - 1:] = cs[:, scale - 1:]
    if scale > 1:
        out[:, scale - 1:] -= np.c_[np.zeros(X.shape[0]),
                                     cs[:, :X.shape[1] - scale]]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTION FITTING
# ─────────────────────────────────────────────────────────────────────────────

def _fit_gamma_per_month(accum_train, mo_train):
    """
    Fit a Gamma distribution per calendar month to rolling-accumulated rainfall.

    Parameters
    ----------
    accum_train : (T_tr,) rolling accumulated rainfall (mm/month × scale)
    mo_train    : (T_tr,) 0-indexed calendar months

    Returns
    -------
    params : list of 12 tuples (a, loc, scale) or None if insufficient data
    """
    params = [None] * 12
    for m in range(12):
        mask = (mo_train == m) & ~np.isnan(accum_train) & (accum_train > 0)
        vals = accum_train[mask]
        if len(vals) < 4:
            continue
        try:
            a, loc, sc = sp_gamma.fit(vals, floc=0.0)
            if a > 0 and sc > 0:
                params[m] = (a, loc, sc)
        except Exception:
            pass
    return params


def _fit_loglogistic_per_month(d_train, mo_train):
    """
    Fit a log-logistic (Fisk) distribution per calendar month to water balance D.

    We shift D so all values are positive before fitting, then record the shift.

    Returns
    -------
    params : list of 12 tuples (c, loc, scale, shift) or None
    """
    params = [None] * 12
    for m in range(12):
        mask = (mo_train == m) & ~np.isnan(d_train)
        vals = d_train[mask]
        if len(vals) < 4:
            continue
        try:
            shift = float(vals.min()) - 1e-3   # shift so all > 0
            v_sh  = vals - shift
            c, loc, sc = sp_fisk.fit(v_sh, floc=0.0)
            if c > 0 and sc > 0:
                params[m] = (c, loc, sc, shift)
        except Exception:
            pass
    return params


# ─────────────────────────────────────────────────────────────────────────────
# SPI TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────

def _gamma_to_spi(accum, mo, gamma_params):
    """
    Transform accumulated rainfall array to SPI using fitted Gamma params.

    Parameters
    ----------
    accum        : (T,) or (n_paths, T) accumulated rainfall
    mo           : (T,) 0-indexed calendar months
    gamma_params : list of 12 tuples from _fit_gamma_per_month

    Returns
    -------
    spi : same shape as accum
    """
    scalar_input = (accum.ndim == 1)
    if scalar_input:
        accum = accum[np.newaxis, :]   # (1, T)

    n_paths, T = accum.shape
    spi        = np.full((n_paths, T), np.nan)

    for t in range(T):
        m = int(mo[t]) % 12
        p = gamma_params[m]
        if p is None:
            continue
        a, loc, sc = p
        vals = accum[:, t]
        valid = ~np.isnan(vals) & (vals > 0)
        if valid.any():
            prob = sp_gamma.cdf(vals[valid], a, loc=loc, scale=sc)
            prob = np.clip(prob, 1e-6, 1 - 1e-6)
            spi[valid, t] = sp_norm.ppf(prob)

    return spi[0] if scalar_input else spi


# ─────────────────────────────────────────────────────────────────────────────
# SPEI TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────

def _loglogistic_to_spei(d_accum, mo, ll_params):
    """
    Transform accumulated water balance to SPEI using fitted log-logistic params.
    """
    scalar_input = (d_accum.ndim == 1)
    if scalar_input:
        d_accum = d_accum[np.newaxis, :]

    n_paths, T = d_accum.shape
    spei       = np.full((n_paths, T), np.nan)

    for t in range(T):
        m = int(mo[t]) % 12
        p = ll_params[m]
        if p is None:
            continue
        c, loc, sc, shift = p
        vals  = d_accum[:, t]
        valid = ~np.isnan(vals)
        if valid.any():
            v_sh  = vals[valid] - shift
            v_sh  = np.maximum(v_sh, 1e-6)
            prob  = sp_fisk.cdf(v_sh, c, loc=loc, scale=sc)
            prob  = np.clip(prob, 1e-6, 1 - 1e-6)
            spei[valid, t] = sp_norm.ppf(prob)

    return spei[0] if scalar_input else spei


# ─────────────────────────────────────────────────────────────────────────────
# DROUGHT PROBABILITY
# ─────────────────────────────────────────────────────────────────────────────

def drought_probabilities(index_ensemble):
    """
    Compute monthly drought exceedance probabilities from ensemble.

    Parameters
    ----------
    index_ensemble : (n_paths, T_te) SPI or SPEI ensemble

    Returns
    -------
    dict with keys 'moderate', 'severe', 'extreme' each (T_te,)
    P(index < threshold) per forecast month
    """
    valid = ~np.isnan(index_ensemble)   # (n_paths, T_te)
    n_valid = valid.sum(axis=0)         # (T_te,)
    n_valid = np.maximum(n_valid, 1)

    return dict(
        moderate = (index_ensemble < -1.0).sum(0) / n_valid * 100,
        severe   = (index_ensemble < -1.5).sum(0) / n_valid * 100,
        extreme  = (index_ensemble < -2.0).sum(0) / n_valid * 100,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def index_fc_stats(ensemble):
    """Summary statistics from index ensemble (n_paths × T), ignoring NaN."""
    return dict(
        mean = np.nanmean(ensemble, axis=0),
        med  = np.nanmedian(ensemble, axis=0),
        p05  = np.nanpercentile(ensemble,  5, axis=0),
        p10  = np.nanpercentile(ensemble, 10, axis=0),
        p90  = np.nanpercentile(ensemble, 90, axis=0),
        p95  = np.nanpercentile(ensemble, 95, axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_drought_model(R, T, scale=3, lat_deg=17.0):
    """
    Compute SPI and SPEI forecast ensembles from Kalman outputs.

    Parameters
    ----------
    R       : dict returned by run_rainfall_kalman()
    T       : dict returned by run_temperature_kalman()
    scale   : int, accumulation period in months (1, 3, 6, or 12)
    lat_deg : float, district latitude for Thornthwaite PET (degrees North)

    Returns
    -------
    dict with keys:
      spi_ensemble  (n_paths, T_te)  — SPI forecast paths
      spei_ensemble (n_paths, T_te)  — SPEI forecast paths
      spi_fc        dict             — mean, p05, p10, p90, p95
      spei_fc       dict             — mean, p05, p10, p90, p95
      spi_prob      dict             — P(moderate/severe/extreme drought)
      spei_prob     dict             — same for SPEI
      spi_obs       (T_te,)          — SPI from observed test rainfall
      spei_obs      (T_te,)          — SPEI from observed test data
      spi_ctx       (ctx,)           — SPI training context for plot
      spei_ctx      (ctx,)           — SPEI training context for plot
      pet_mean_fc   (T_te,)          — PET from mean temperature forecast
      scale, lat_deg
    """
    # ── unpack rainfall arrays ────────────────────────────────────────────────
    rain_tr     = R['rain_tr']                    # (T_tr,) mm/day
    rain_te     = R['rain_te']                    # (T_te,) mm/day
    rain_ens    = R['X_kalman']                   # (n_paths, T_te) mm/day
    mo_tr       = R['mo_tr']                      # (T_tr,) 0-indexed
    mo_te       = R['mo_te']                      # (T_te,) 0-indexed
    T_tr        = R['T_tr']
    T_te        = R['T_te']

    # ── unpack temperature arrays ─────────────────────────────────────────────
    temp_tr     = T['temp_tr']                    # (T_tr,) °C
    temp_te     = T['temp_te']                    # (T_te,) °C
    mo_tr_t     = T['mo_tr']
    mo_te_t     = T['mo_te']
    # Mean temperature forecast path  (T_te,)
    temp_mean_fc = T['fc']['mean']

    # ── convert rainfall mm/day → mm/month ───────────────────────────────────
    days_tr     = _DAYS_IN_MONTH[mo_tr]
    days_te     = _DAYS_IN_MONTH[mo_te]

    rain_tr_mm  = rain_tr * days_tr               # (T_tr,)
    rain_te_mm  = rain_te * days_te               # (T_te,)
    rain_ens_mm = rain_ens * days_te[np.newaxis, :]  # (n_paths, T_te)

    # ── PET for training + forecast ───────────────────────────────────────────
    print(f"  [Drought] Computing Thornthwaite PET (lat={lat_deg}°N) ...")
    pet_tr   = thornthwaite_pet(temp_tr, mo_tr_t, lat_deg)       # (T_tr,)
    pet_fc   = thornthwaite_pet(temp_mean_fc, mo_te_t, lat_deg)  # (T_te,) mean path

    # Water balance D = rain - PET (mm/month)
    d_tr     = rain_tr_mm - pet_tr                # (T_tr,)
    d_te     = rain_te_mm - pet_fc                # (T_te,) observed balance

    # ── build combined series with warm-up tail for rolling ──────────────────
    # We need the last (scale-1) training months to initialise the rolling sum
    # for the first forecast step.
    tail = scale - 1

    rain_combined      = np.r_[rain_tr_mm[-tail:], rain_te_mm]     # (tail+T_te,)
    rain_ens_combined  = np.c_[
        np.tile(rain_tr_mm[-tail:], (rain_ens_mm.shape[0], 1)),
        rain_ens_mm
    ]                                                               # (n_paths, tail+T_te)

    d_combined     = np.r_[d_tr[-tail:], d_te]                     # (tail+T_te,)
    # For ensemble SPEI: i-th rain path + mean PET (same for all paths)
    d_ens_combined = np.c_[
        np.tile(d_tr[-tail:], (rain_ens_mm.shape[0], 1)),
        rain_ens_mm - pet_fc[np.newaxis, :]
    ]                                                               # (n_paths, tail+T_te)

    mo_combined = np.r_[mo_tr[-tail:], mo_te]                      # (tail+T_te,)

    # ── rolling accumulations ─────────────────────────────────────────────────
    print(f"  [Drought] Rolling accumulation (scale={scale}) ...")
    # Training (for distribution fitting)
    rain_tr_roll = _rolling_sum(rain_tr_mm, scale)     # (T_tr,)
    d_tr_roll    = _rolling_sum(d_tr, scale)            # (T_tr,)

    # Observed test period
    rain_obs_roll = _rolling_sum(rain_combined, scale)[-T_te:]  # (T_te,)
    d_obs_roll    = _rolling_sum(d_combined, scale)[-T_te:]     # (T_te,)

    # Ensemble (n_paths, T_te)
    rain_ens_roll = _rolling_sum_2d(rain_ens_combined, scale)[:, -T_te:]
    d_ens_roll    = _rolling_sum_2d(d_ens_combined, scale)[:, -T_te:]

    # Training context for plot (last 18 months)
    ctx = min(18, T_tr - scale)
    rain_ctx_roll = _rolling_sum(rain_tr_mm, scale)[-ctx:]
    d_ctx_roll    = _rolling_sum(d_tr, scale)[-ctx:]
    mo_ctx        = mo_tr[-ctx:]

    # ── fit distributions on training data ───────────────────────────────────
    print(f"  [Drought] Fitting Gamma (SPI) and Log-logistic (SPEI) ...")
    gamma_params = _fit_gamma_per_month(rain_tr_roll, mo_tr)
    ll_params    = _fit_loglogistic_per_month(d_tr_roll, mo_tr)

    # ── transform to SPI/SPEI ─────────────────────────────────────────────────
    print(f"  [Drought] Transforming {rain_ens_roll.shape[0]} paths ...")

    # Ensemble
    spi_ensemble  = _gamma_to_spi(rain_ens_roll, mo_te, gamma_params)
    spei_ensemble = _loglogistic_to_spei(d_ens_roll, mo_te, ll_params)

    # Observed test
    spi_obs  = _gamma_to_spi(rain_obs_roll, mo_te, gamma_params)
    spei_obs = _loglogistic_to_spei(d_obs_roll, mo_te, ll_params)

    # Training context (for plot)
    spi_ctx  = _gamma_to_spi(rain_ctx_roll, mo_ctx, gamma_params)
    spei_ctx = _loglogistic_to_spei(d_ctx_roll, mo_ctx, ll_params)

    # ── statistics ────────────────────────────────────────────────────────────
    spi_fc   = index_fc_stats(spi_ensemble)
    spei_fc  = index_fc_stats(spei_ensemble)
    spi_prob = drought_probabilities(spi_ensemble)
    spei_prob= drought_probabilities(spei_ensemble)

    # ── summary print ─────────────────────────────────────────────────────────
    print(f"  [Drought] SPI-{scale}  mean={np.nanmean(spi_fc['mean']):.2f}  "
          f"P(moderate)={spi_prob['moderate'].mean():.0f}%  "
          f"P(severe)={spi_prob['severe'].mean():.0f}%")
    print(f"  [Drought] SPEI-{scale} mean={np.nanmean(spei_fc['mean']):.2f}  "
          f"P(moderate)={spei_prob['moderate'].mean():.0f}%  "
          f"P(severe)={spei_prob['severe'].mean():.0f}%")

    return dict(
        spi_ensemble  = spi_ensemble,
        spei_ensemble = spei_ensemble,
        spi_fc        = spi_fc,
        spei_fc       = spei_fc,
        spi_prob      = spi_prob,
        spei_prob     = spei_prob,
        spi_obs       = spi_obs,
        spei_obs      = spei_obs,
        spi_ctx       = spi_ctx,
        spei_ctx      = spei_ctx,
        pet_mean_fc   = pet_fc,
        mo_te         = mo_te,
        mo_ctx        = mo_ctx,
        ctx           = ctx,
        scale         = scale,
        lat_deg       = lat_deg,
        n_paths       = rain_ens_roll.shape[0],
    )


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3
# ─────────────────────────────────────────────────────────────────────────────

def plot_drought(D, district, x_fc, xt, xl, yr_te, mo_te,
                 yr_ctx=None, mo_ctx=None):
    """
    Plot 3: SPI and SPEI forecast ensemble with drought category bands
    and monthly drought risk probabilities.

    Parameters
    ----------
    D        : dict returned by run_drought_model()
    district : str
    x_fc     : np.arange(T_te)
    xt, xl   : x-tick positions and labels
    yr_te, mo_te : year/month arrays for test period (for annotations)
    yr_ctx, mo_ctx: year/month arrays for context (optional)

    Returns
    -------
    fig
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from temperature_model import MONTH_NAMES

    scale   = D['scale']
    n_paths = D['n_paths']
    T_te    = len(x_fc)
    ctx     = D['ctx']
    x_ctx   = np.arange(-ctx, 0)

    # ── colours ──────────────────────────────────────────────────────────────
    C_SPI   = '#1D4E89'
    C_SPEI  = '#7B2D8B'
    C_OBS   = '#D85A30'
    C_TR    = '#888780'

    # Category band colours (light, semi-transparent)
    CAT_COLORS = {
        'extreme':  ('#8B0000', 0.18),
        'severe':   ('#D63A0C', 0.15),
        'moderate': ('#F0A500', 0.12),
        'normal':   ('#AAAAAA', 0.04),
        'wet':      ('#1D9E75', 0.08),
    }

    fig = plt.figure(figsize=(15, 14))
    fig.suptitle(
        f"{district}  —  Plot 3: Probabilistic Drought Forecast  "
        f"(SPI-{scale} & SPEI-{scale})",
        fontsize=12, fontweight='bold', y=1.005)

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.52, wspace=0.30,
                           height_ratios=[2.2, 2.2, 1.4])

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_category_bands(ax, xlim):
        """Draw horizontal drought/wet category bands on axis."""
        bands = [
            (-3.5, -2.0, 'extreme',  'Extreme drought'),
            (-2.0, -1.5, 'severe',   'Severe drought'),
            (-1.5, -1.0, 'moderate', 'Moderate drought'),
            (-1.0,  1.0, 'normal',   'Near normal'),
            ( 1.0,  3.5, 'wet',      'Wet'),
        ]
        for ylo, yhi, key, lbl in bands:
            col, alp = CAT_COLORS[key]
            ax.axhspan(ylo, yhi, alpha=alp, color=col, zorder=0)
        ax.axhline(0,    color='#555555', lw=0.7, ls='--', alpha=0.5)
        ax.axhline(-1.0, color='#F0A500', lw=0.8, ls=':', alpha=0.7)
        ax.axhline(-1.5, color='#D63A0C', lw=0.8, ls=':', alpha=0.7)
        ax.axhline(-2.0, color='#8B0000', lw=0.8, ls=':', alpha=0.7)

    def _draw_index_panel(ax, ens, fc, obs, ctx_vals,
                          color, label, unit=''):
        """Draw ensemble paths, bands, mean, observed on one SPI/SPEI panel."""
        # Category shading
        _draw_category_bands(ax, (x_ctx[0], x_fc[-1]))

        # Thinned ensemble paths
        n_show = min(300, n_paths)
        idx_s  = np.linspace(0, n_paths - 1, n_show, dtype=int)
        for i in idx_s:
            row = ens[i]
            valid = ~np.isnan(row)
            if valid.any():
                ax.plot(x_fc[valid], row[valid],
                        lw=0.25, color=color, alpha=0.05)

        # 90% PI band
        ax.fill_between(x_fc, fc['p05'], fc['p95'],
                        alpha=0.18, color=color, label='5th–95th (90% PI)')
        ax.fill_between(x_fc, fc['p10'], fc['p90'],
                        alpha=0.28, color=color, label='10th–90th (80% PI)')

        # Quantile lines
        for vals, ls_, pct in [(fc['p95'], '--', '95th'),
                                (fc['p05'], '--', '5th')]:
            valid = ~np.isnan(vals)
            if valid.any():
                ax.plot(x_fc[valid], vals[valid],
                        lw=0.8, color=color, ls=ls_, alpha=0.7)
                ax.annotate(pct,
                            xy=(x_fc[valid][-1], vals[valid][-1]),
                            xytext=(x_fc[valid][-1] + 0.15, vals[valid][-1]),
                            fontsize=6.5, color=color, va='center')

        # Training context
        if ctx_vals is not None:
            valid_c = ~np.isnan(ctx_vals)
            if valid_c.any():
                ax.plot(x_ctx[valid_c], ctx_vals[valid_c],
                        lw=1.3, color=C_TR, ls='--', alpha=0.7,
                        label='Historical context')

        ax.axvline(-0.5, lw=1.0, color='black', alpha=0.2, ls=':')

        # Ensemble mean
        valid_m = ~np.isnan(fc['mean'])
        ax.plot(x_fc[valid_m], fc['mean'][valid_m],
                lw=2.5, color=color, zorder=5, label=f'{label} mean')

        # Observed
        valid_o = ~np.isnan(obs)
        if valid_o.any():
            ax.scatter(x_fc[valid_o], obs[valid_o],
                       s=45, color=C_OBS, zorder=7,
                       edgecolors='white', linewidths=0.5,
                       label='Observed')

        ax.axhline(0, color='#555555', lw=0.7, ls='--', alpha=0.4)

    # ── Panel 1: SPI ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _draw_index_panel(ax1,
                      D['spi_ensemble'], D['spi_fc'], D['spi_obs'],
                      D['spi_ctx'], C_SPI, f'SPI-{scale}')

    # Category labels on right margin
    for yc, lbl in [(-1.0, 'Moderate'), (-1.5, 'Severe'), (-2.0, 'Extreme')]:
        ax1.text(T_te - 0.1, yc + 0.05, lbl,
                 fontsize=6.5, color='#666666', va='bottom', ha='right')

    ax1.set_xticks(xt)
    ax1.set_xticklabels(xl, rotation=40, ha='right', fontsize=7)
    ax1.set_xlim(x_ctx[0] - 0.5, T_te - 0.2)
    ax1.set_ylim(-3.2, 3.2)
    ax1.set_ylabel(f'SPI-{scale}', fontsize=10)
    ax1.set_title(
        f'SPI-{scale}: Standardised Precipitation Index  '
        f'({n_paths} ensemble paths)',
        fontsize=9, loc='left')
    ax1.legend(fontsize=7.5, ncol=4, loc='upper left')
    ax1.grid(True, alpha=0.12, axis='x')

    # ── Panel 2: SPEI ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    _draw_index_panel(ax2,
                      D['spei_ensemble'], D['spei_fc'], D['spei_obs'],
                      D['spei_ctx'], C_SPEI, f'SPEI-{scale}')

    for yc, lbl in [(-1.0, 'Moderate'), (-1.5, 'Severe'), (-2.0, 'Extreme')]:
        ax2.text(T_te - 0.1, yc + 0.05, lbl,
                 fontsize=6.5, color='#666666', va='bottom', ha='right')

    ax2.set_xticks(xt)
    ax2.set_xticklabels(xl, rotation=40, ha='right', fontsize=7)
    ax2.set_xlim(x_ctx[0] - 0.5, T_te - 0.2)
    ax2.set_ylim(-3.2, 3.2)
    ax2.set_ylabel(f'SPEI-{scale}', fontsize=10)
    ax2.set_title(
        f'SPEI-{scale}: Standardised Precipitation-Evapotranspiration Index  '
        f'(rain path i + mean temp, Thornthwaite PET, lat={D["lat_deg"]}°N)',
        fontsize=9, loc='left')
    ax2.legend(fontsize=7.5, ncol=4, loc='upper left')
    ax2.grid(True, alpha=0.12, axis='x')

    # ── Panel 3a: SPI drought risk probabilities ──────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    sp  = D['spi_prob']
    x_f = np.arange(T_te)
    fc_xlbl = [f"{yr_te[i]} {MONTH_NAMES[mo_te[i]]}"
               for i in range(T_te)]
    vis_fc = list(range(0, T_te, max(1, T_te // 6)))

    ax3.fill_between(x_f, 0, sp['moderate'],
                     alpha=0.35, color='#F0A500',
                     step='mid', label='P(moderate, <−1)')
    ax3.fill_between(x_f, 0, sp['severe'],
                     alpha=0.45, color='#D63A0C',
                     step='mid', label='P(severe, <−1.5)')
    ax3.fill_between(x_f, 0, sp['extreme'],
                     alpha=0.55, color='#8B0000',
                     step='mid', label='P(extreme, <−2)')

    ax3.step(x_f, sp['moderate'], color='#F0A500', lw=1.5, where='mid')
    ax3.step(x_f, sp['severe'],   color='#D63A0C', lw=1.5, where='mid')
    ax3.step(x_f, sp['extreme'],  color='#8B0000', lw=1.5, where='mid')
    ax3.axhline(50, color='gray', lw=0.8, ls='--', alpha=0.5)

    ax3.set_xticks(vis_fc)
    ax3.set_xticklabels([fc_xlbl[i] for i in vis_fc],
                        rotation=40, ha='right', fontsize=7)
    ax3.set_xlim(-0.5, T_te - 0.5)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel('Drought probability (%)', fontsize=9)
    ax3.set_title(f'SPI-{scale}: Monthly drought risk', fontsize=9, loc='left')
    ax3.legend(fontsize=7.5, loc='upper right')
    ax3.grid(True, alpha=0.18, axis='y')

    # ── Panel 3b: SPEI drought risk probabilities ─────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ep  = D['spei_prob']

    ax4.fill_between(x_f, 0, ep['moderate'],
                     alpha=0.35, color='#F0A500',
                     step='mid', label='P(moderate, <−1)')
    ax4.fill_between(x_f, 0, ep['severe'],
                     alpha=0.45, color='#D63A0C',
                     step='mid', label='P(severe, <−1.5)')
    ax4.fill_between(x_f, 0, ep['extreme'],
                     alpha=0.55, color='#8B0000',
                     step='mid', label='P(extreme, <−2)')

    ax4.step(x_f, ep['moderate'], color='#F0A500', lw=1.5, where='mid')
    ax4.step(x_f, ep['severe'],   color='#D63A0C', lw=1.5, where='mid')
    ax4.step(x_f, ep['extreme'],  color='#8B0000', lw=1.5, where='mid')
    ax4.axhline(50, color='gray', lw=0.8, ls='--', alpha=0.5)

    ax4.set_xticks(vis_fc)
    ax4.set_xticklabels([fc_xlbl[i] for i in vis_fc],
                        rotation=40, ha='right', fontsize=7)
    ax4.set_xlim(-0.5, T_te - 0.5)
    ax4.set_ylim(0, 100)
    ax4.set_ylabel('Drought probability (%)', fontsize=9)
    ax4.set_title(f'SPEI-{scale}: Monthly drought risk', fontsize=9, loc='left')
    ax4.legend(fontsize=7.5, loc='upper right')
    ax4.grid(True, alpha=0.18, axis='y')

    plt.tight_layout()
    return fig
