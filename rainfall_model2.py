"""
rainfall_model.py
=================
Helper module: Kalman-updated Log-OU SDE for monthly rainfall.

Public API
----------
run_rainfall_kalman(rain, months, years, train_mask, n_paths, seed,
                    k_mu, k_sig, q_mu0, q_gamma) -> dict

Returned dict keys
------------------
  X_kalman   (n_paths, T_te)  — ensemble in mm/day
  fc         dict  — mean, p05, p10, p90, p95, med
  acc        dict  — rmse, mae, bias, skill, cov90
  lta_mo     (12,) — long-term average by calendar month (training)
  sde        dict  — fitted SDE parameters
  kf         dict  — Kalman filter output
  rain_tr, rain_te, mo_tr, mo_te, yr_te
  T_tr, T_te
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import shapiro

# ── Gauss-Legendre quadrature nodes & weights (4-point) ─────────────────────
GL_N = np.array([0.0694318442, 0.3300094782, 0.6699905218, 0.9305681558])
GL_W = np.array([0.1739274226, 0.3260725774, 0.3260725774, 0.1739274226])

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']

# Default Fourier orders and Kalman process noise
_K_MU   = 3
_K_SIG  = 2
_Q_MU0  = 1e-4
_Q_GAMMA = 1e-7


# ─────────────────────────────────────────────────────────────────────────────
# FOURIER HELPERS — fully vectorised
# ─────────────────────────────────────────────────────────────────────────────

def fit_fourier(vals, K):
    """Fit Fourier coefficients to a length-12 monthly array (vectorised over k)."""
    N     = 12
    k_arr = np.arange(1, K + 1)
    m_arr = np.arange(N, dtype=float)
    angs  = 2 * np.pi * k_arr[:, None] * m_arr[None, :] / N   # (K, N)
    a_k   = 2 * (vals * np.cos(angs)).sum(axis=1) / N
    b_k   = 2 * (vals * np.sin(angs)).sum(axis=1) / N
    return np.r_[vals.mean(), a_k], np.r_[0., b_k]


def eval_f(t, a, b, K, P=12.):
    """Evaluate Fourier series at arbitrary array t (vectorised over k and t)."""
    t     = np.asarray(t, dtype=float)
    k_arr = np.arange(1, K + 1)
    ax    = (slice(None),) + (None,) * t.ndim
    ang   = 2 * np.pi / P * k_arr[ax] * t
    return a[0] + (a[1:][ax] * np.cos(ang) + b[1:][ax] * np.sin(ang)).sum(axis=0)


def acf_fn(x, nlags=24):
    """FFT-based autocorrelation function."""
    x  = x - x.mean(); n = len(x)
    f  = np.fft.fft(x, n=2 * n)
    ac = np.real(np.fft.ifft(f * np.conj(f)))[:n] / n
    return ac[:nlags + 1] / ac[0]


# ─────────────────────────────────────────────────────────────────────────────
# SDE PARAMETER ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def _mom_fit(lr, mo_idx, t_abs, K_MU, K_SIG):
    """Method-of-moments starting values."""
    n_yr  = len(lr) // 12
    slope = np.polyfit(np.arange(n_yr),
                       [lr[y*12:(y+1)*12].mean() for y in range(n_yr)], 1)[0]
    trend = slope * 10
    lr_dt = lr - trend * t_abs / 120
    mu_mo = np.array([lr_dt[mo_idx == m].mean() for m in range(12)])
    mu_a, mu_b = fit_fourier(mu_mo, K_MU)
    resid  = lr_dt - eval_f(t_abs % 12, mu_a, mu_b, K_MU)
    sig_mo = np.maximum([resid[mo_idx == m].std() for m in range(12)], 0.05)
    sig_a, sig_b = fit_fourier(np.log(sig_mo), K_SIG)
    phi = max(min(acf_fn(resid, 1)[1], 0.98), 0.02)
    return dict(theta=-np.log(phi), trend=trend, beta=0.,
                mu_a=mu_a, mu_b=mu_b, sig_a=sig_a, sig_b=sig_b)


def _pack(p, K_MU, K_SIG):
    return np.concatenate([
        [np.log(p['theta'])], [p['trend']], [p['beta']],
        p['mu_a'], p['mu_b'][1:], p['sig_a'], p['sig_b'][1:]
    ])


def _unpack(x, K_MU, K_SIG):
    th = np.exp(x[0]); i = 1
    tr = x[i]; i += 1
    be = x[i]; i += 1
    ma = x[i:i + K_MU + 1]; i += K_MU + 1
    mb = np.r_[0., x[i:i + K_MU]]; i += K_MU
    sa = x[i:i + K_SIG + 1]; i += K_SIG + 1
    sb = np.r_[0., x[i:i + K_SIG]]
    return th, tr, be, ma, mb, sa, sb


def _nll_fn(x, lr, enso, K_MU, K_SIG):
    """Fully vectorised NLL — no Python time loop."""
    try:
        th, tr_p, be, ma, mb, sa, sb = _unpack(x, K_MU, K_SIG)
        if th <= 0:
            return 1e10
        T   = len(lr) - 1
        en  = np.exp(-th)
        decay_1 = np.exp(-th * (1 - GL_N))
        decay_2 = decay_1 ** 2

        t_idx = np.arange(T, dtype=float)
        s     = t_idx[:, None] + GL_N[None, :]          # (T, 4)
        s_mod = s % 12

        mu_v = (eval_f(s_mod, ma, mb, K_MU)
                + tr_p * (s / 12.0) / 10.0
                + be * enso[:T, None])
        sg_v = np.exp(eval_f(s_mod, sa, sb, K_SIG))

        E = en * lr[:T] + th * (GL_W * mu_v * decay_1).sum(axis=1)
        V = np.maximum((GL_W * sg_v ** 2 * decay_2).sum(axis=1), 1e-12)

        resid = lr[1:] - E
        return 0.5 * np.sum(np.log(2 * np.pi * V) + resid ** 2 / V)
    except Exception:
        return 1e10


def _mle_fit(lr, enso, p0, K_MU, K_SIG):
    """Two-stage MLE: Nelder-Mead → L-BFGS-B."""
    x0 = _pack(p0, K_MU, K_SIG)
    print("    Nelder-Mead ...")
    r1 = minimize(_nll_fn, x0, args=(lr, enso, K_MU, K_SIG),
                  method='Nelder-Mead',
                  options={'maxiter': 80000, 'xatol': 1e-6,
                           'fatol': 1e-6, 'adaptive': True})
    print("    L-BFGS-B ...")
    r2 = minimize(_nll_fn, r1.x, args=(lr, enso, K_MU, K_SIG),
                  method='L-BFGS-B',
                  options={'maxiter': 30000, 'ftol': 1e-12})
    best = r2.x if r2.fun < r1.fun else r1.x
    th, tr, be, ma, mb, sa, sb = _unpack(best, K_MU, K_SIG)
    m12 = np.arange(12, dtype=float)
    return dict(theta=th, trend=tr, beta=be,
                mu_a=ma, mu_b=mb, sig_a=sa, sig_b=sb,
                mu_mo=eval_f(m12, ma, mb, K_MU),
                sig_mo=np.exp(eval_f(m12, sa, sb, K_SIG)))


# ─────────────────────────────────────────────────────────────────────────────
# KALMAN FILTER
# ─────────────────────────────────────────────────────────────────────────────

def _kalman_H(t_a, Zt, theta):
    c = 1.0 - np.exp(-theta)
    return np.array([c, c * Zt, c * (t_a + 0.5) / 120.0])


def run_kalman(log_rain, t_abs_arr, enso_arr, sde, s_init, P_init, Q):
    """
    Kalman filter over the full dataset.
    Decay arrays precomputed once outside the sequential loop.
    """
    T   = len(log_rain)
    en  = np.exp(-sde['theta'])
    I3  = np.eye(3)
    decay_1 = np.exp(-sde['theta'] * (1 - GL_N))
    decay_2 = decay_1 ** 2

    s = s_init.copy(); P = P_init.copy()
    s_filt = np.zeros((T, 3));   P_filt = np.zeros((T, 3, 3))
    y_pred = np.zeros(T - 1);    y_var  = np.zeros(T - 1)
    innov  = np.zeros(T - 1)
    s_filt[0] = s;  P_filt[0] = P

    for t in range(T - 1):
        t_a = t_abs_arr[t]; Zt = enso_arr[t]
        s_gl  = t_a + GL_N;  s_mod = s_gl % 12

        mu_f  = eval_f(s_mod, sde['mu_a'], sde['mu_b'], _K_MU)
        sg_f  = np.exp(eval_f(s_mod, sde['sig_a'], sde['sig_b'], _K_SIG))
        f_int = sde['theta'] * np.dot(GL_W, mu_f * decay_1)
        R     = max(np.dot(GL_W, sg_f ** 2 * decay_2), 1e-12)

        s_pred = s.copy(); P_pred = P + Q
        H      = _kalman_H(t_a, Zt, sde['theta'])
        y_hat  = en * log_rain[t] + f_int + H @ s_pred
        S      = H @ P_pred @ H + R

        y_pred[t] = y_hat; y_var[t] = S
        innov[t]  = log_rain[t + 1] - y_hat

        K_gain = (P_pred @ H) / S
        s = s_pred + K_gain * innov[t]
        P = (I3 - np.outer(K_gain, H)) @ P_pred
        s_filt[t + 1] = s; P_filt[t + 1] = P

    return dict(s_filt=s_filt, P_filt=P_filt,
                y_pred=y_pred, y_var=y_var, innov=innov)


def kalman_forecast_ensemble(log_last, t_start, enso_fc, sde,
                              s_init, P_init, Q, n_paths, seed):
    """
    Monte-Carlo forecast. All GL integrals, H vectors, L2 and random numbers
    precomputed in one batch before the time loop.
    """
    rng = np.random.default_rng(seed)
    n   = len(enso_fc)
    en  = np.exp(-sde['theta'])
    decay_1 = np.exp(-sde['theta'] * (1 - GL_N))
    decay_2 = decay_1 ** 2

    t_steps   = t_start + np.arange(n - 1, dtype=float)
    s_all     = t_steps[:, None] + GL_N[None, :]
    s_mod     = s_all % 12
    mu_all    = eval_f(s_mod, sde['mu_a'], sde['mu_b'], _K_MU)
    sg_all    = np.exp(eval_f(s_mod, sde['sig_a'], sde['sig_b'], _K_SIG))
    f_int_all = sde['theta'] * (GL_W * mu_all * decay_1).sum(axis=1)
    R_all     = np.maximum((GL_W * sg_all ** 2 * decay_2).sum(axis=1), 1e-12)

    c     = 1.0 - en
    H_all = np.column_stack([
        np.full(n - 1, c),
        c * enso_fc[:n - 1],
        c * (t_steps + 0.5) / 120.0
    ])

    paths_log = np.zeros((n_paths, n))
    paths_log[:, 0] = log_last
    L_P    = np.linalg.cholesky(P_init + 1e-10 * np.eye(3))
    states = s_init + (L_P @ rng.standard_normal((3, n_paths))).T
    L2     = np.linalg.cholesky(Q + 1e-12 * np.eye(3))

    obs_noise   = rng.standard_normal((n - 1, n_paths))
    state_noise = (L2 @ rng.standard_normal((3, (n - 1) * n_paths))).T
    state_noise = state_noise.reshape(n - 1, n_paths, 3)

    P_curr = P_init.copy()
    for t in range(n - 1):
        H         = H_all[t]
        P_curr    = P_curr + Q
        total_var = R_all[t] + H @ P_curr @ H
        mu_det    = en * paths_log[:, t] + f_int_all[t]
        paths_log[:, t + 1] = (mu_det + states @ H
                                + np.sqrt(total_var) * obs_noise[t])
        states = states + state_noise[t]

    return np.exp(paths_log)


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS & METRICS
# ─────────────────────────────────────────────────────────────────────────────

def fc_stats(X):
    """Summary statistics from ensemble matrix (n_paths × T)."""
    return dict(
        mean = X.mean(0),
        med  = np.median(X, 0),
        p05  = np.percentile(X,  5, 0),
        p10  = np.percentile(X, 10, 0),
        p90  = np.percentile(X, 90, 0),
        p95  = np.percentile(X, 95, 0),
    )


def compute_metrics(obs, fc):
    """RMSE, MAE, BIAS, SKILL, 90% PI coverage."""
    m     = fc['mean']
    rmse  = float(np.sqrt(np.mean((obs - m) ** 2)))
    mae   = float(np.mean(np.abs(obs - m)))
    bias  = float(np.mean(m - obs))
    skill = float(1 - np.sum((obs - m) ** 2) /
                  max(np.sum((obs - obs.mean()) ** 2), 1e-12))
    cov90 = float(np.mean((obs >= fc['p05']) & (obs <= fc['p95'])) * 100)
    return dict(rmse=rmse, mae=mae, bias=bias, skill=skill, cov90=cov90)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_rainfall_kalman(rain, months, years, train_mask,
                        n_paths=4000, seed=42,
                        k_mu=_K_MU, k_sig=_K_SIG,
                        q_mu0=_Q_MU0, q_gamma=_Q_GAMMA,
                        enso=None):
    """
    Fit Kalman Log-OU and return forecast ensemble + statistics.

    Parameters
    ----------
    rain       : (N,) array of monthly rainfall (mm/day), all years
    months     : (N,) int array, 0-indexed (0=Jan … 11=Dec)
    years      : (N,) int array
    train_mask : (N,) bool, True = training, False = test/forecast
    n_paths    : number of Monte-Carlo paths
    seed       : RNG seed
    k_mu, k_sig: Fourier orders for mean / sigma
    q_mu0, q_gamma: Kalman process noise
    enso       : (N,) float array of standardised ONI anomalies, or None.
                 Pass the output of enso_helper.align_oni() to activate the
                 ENSO covariate (beta parameter).  If None, ENSO is set to
                 zeros and the model runs without the teleconnection term.

    Returns
    -------
    dict with all arrays needed for plotting
    """
    global _K_MU, _K_SIG
    _K_MU  = k_mu
    _K_SIG = k_sig

    rain_d = np.maximum(rain, 0.001)
    log_d  = np.log(rain_d)

    rain_tr = rain_d[train_mask];  rain_te = rain_d[~train_mask]
    log_tr  = log_d[train_mask];   log_te  = log_d[~train_mask]
    mo_tr   = months[train_mask];  mo_te   = months[~train_mask]
    yr_te   = years[~train_mask]
    T_tr    = int(train_mask.sum()); T_te = int((~train_mask).sum())
    t_abs_tr  = np.arange(T_tr, dtype=float)
    t_abs_all = np.arange(len(log_d), dtype=float)

    # ENSO covariate — use provided array or fall back to zeros
    if enso is not None and len(enso) == len(log_d):
        enso_all = np.asarray(enso, dtype=float)
        print("  [Rain] ENSO covariate: ACTIVE (ONI provided)")
    else:
        enso_all = np.zeros(len(log_d))
        print("  [Rain] ENSO covariate: OFF (running without ONI)")

    print("  [Rain] Calibrating SDE ...")
    p0  = _mom_fit(log_tr, mo_tr, t_abs_tr, k_mu, k_sig)
    sde = _mle_fit(log_tr, enso_all[:T_tr], p0, k_mu, k_sig)
    ar1 = float(np.exp(-sde['theta']))
    print(f"  [Rain] θ={sde['theta']:.3f}  AR(1)={ar1:.3f}  "
          f"γ={sde['trend']:+.3f}/dec")

    Q_mat = np.diag([q_mu0, 1e-5, q_gamma])
    s0    = np.array([sde['mu_a'][0], sde['beta'], sde['trend']])
    P0    = np.diag([0.5**2, 0.3**2, 0.5**2])

    print("  [Rain] Running Kalman filter ...")
    kf    = run_kalman(log_d, t_abs_all, enso_all, sde, s0, P0, Q_mat)
    s_end = kf['s_filt'][T_tr]
    P_end = kf['P_filt'][T_tr]

    print(f"  [Rain] Generating ensemble ({n_paths} paths) ...")
    enso_te  = enso_all[~train_mask]
    X_kalman = kalman_forecast_ensemble(
        log_tr[-1], T_tr - 1, enso_te, sde,
        s_end, P_end, Q_mat, n_paths, seed)

    fc  = fc_stats(X_kalman)
    acc = compute_metrics(rain_te, fc)

    # Long-term average by calendar month (training data)
    lta_mo = np.array([
        rain_tr[mo_tr == m].mean() if (mo_tr == m).any() else np.nan
        for m in range(12)
    ])

    # Kalman diagnostics
    kf_innov     = kf['innov'][:T_tr - 1]
    kf_std_innov = kf_innov / np.sqrt(np.maximum(kf['y_var'][:T_tr - 1], 1e-12))
    ac           = acf_fn(kf_std_innov, 24)
    n_i          = len(kf_std_innov)
    lb           = n_i * (n_i + 2) * sum(ac[k] ** 2 / (n_i - k) for k in range(1, 13))
    sw_p         = shapiro(kf_std_innov)[1] if n_i >= 3 else float('nan')

    print(f"  [Rain] RMSE={acc['rmse']:.3f}  MAE={acc['mae']:.3f}  "
          f"Bias={acc['bias']:+.3f}  Skill={acc['skill']:.3f}  "
          f"90%PI={acc['cov90']:.0f}%")
    print(f"  [Rain] LB(12)={'PASS' if lb < 21.03 else 'FAIL'}  "
          f"Shapiro p={sw_p:.3f}")

    return dict(
        X_kalman = X_kalman,
        fc       = fc,
        acc      = acc,
        lta_mo   = lta_mo,
        sde      = sde,
        kf       = kf,
        rain_tr  = rain_tr,
        rain_te  = rain_te,
        log_tr   = log_tr,
        mo_tr    = mo_tr,
        mo_te    = mo_te,
        yr_te    = yr_te,
        months_all = months,
        years_all  = years,
        T_tr     = T_tr,
        T_te     = T_te,
        lb       = lb,
        sw_p     = sw_p,
        ar1      = ar1,
    )
