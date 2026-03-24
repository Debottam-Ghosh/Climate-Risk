import numpy as np
from sklearn.ensemble import RandomForestRegressor
from quantile_forest import RandomForestQuantileRegressor

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']


# ─────────────────────────────────────────
# DLM CLASS
# ─────────────────────────────────────────
class SimpleDLM:
    """Simple DLM object for storing and computing temperature dynamics parameters."""
    def __init__(self, theta=0.1, sigma=0.5, q=0.05):
        self.theta = float(theta)   # Mean reversion parameter
        self.sigma = float(sigma)   # Volatility/noise parameter
        self.q = float(q)           # Process noise (Kalman)
    
    def half_life(self):
        """Compute half-life in months from mean reversion parameter."""
        if self.theta <= 0:
            return float('inf')
        # half_life ≈ ln(2) / theta (for small theta)
        return np.log(2) / self.theta


# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
def create_features(temp, months):
    n = len(temp)

    month_sin = np.sin(2 * np.pi * (months+1) / 12)
    month_cos = np.cos(2 * np.pi * (months+1) / 12)

    lag1  = np.roll(temp, 1)
    lag2  = np.roll(temp, 2)
    lag12 = np.roll(temp, 12)

    roll3 = np.array([np.mean(temp[max(0,i-3):i]) for i in range(n)])
    roll6 = np.array([np.mean(temp[max(0,i-6):i]) for i in range(n)])

    X = np.column_stack([
        month_sin, month_cos,
        lag1, lag2, lag12,
        roll3, roll6
    ])

    return X


# ─────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────
def compute_metrics(obs, fc):
    m = fc['mean']
    rmse = float(np.sqrt(np.mean((obs - m)**2)))
    mae  = float(np.mean(np.abs(obs - m)))
    bias = float(np.mean(m - obs))

    skill = float(1 - np.sum((obs - m)**2) /
                  max(np.sum((obs - obs.mean())**2), 1e-12))

    cov90 = float(np.mean((obs >= fc['p05']) & (obs <= fc['p95'])) * 100)

    return dict(rmse=rmse, mae=mae, bias=bias, skill=skill, cov90=cov90)


# ─────────────────────────────────────────
# MAIN FUNCTION (API SAME AS BEFORE)
# ─────────────────────────────────────────
def run_temperature_kalman(temp, months, years, train_mask,
                          n_paths=2000, seed=42, q=0.05):

    temp_tr = temp[train_mask]
    temp_te = temp[~train_mask]

    mo_tr = months[train_mask]
    mo_te = months[~train_mask]

    T_tr = int(train_mask.sum())
    T_te = int((~train_mask).sum())

    # ── Features
    X_all = create_features(temp, months)

    X_tr = X_all[train_mask]
    X_te = X_all[~train_mask]

    # Remove first 12 rows (lag issues)
    valid_idx = np.arange(len(temp)) >= 12
    X_tr = X_all[train_mask & valid_idx]
    y_tr = temp[train_mask & valid_idx]

    X_te = X_all[~train_mask]
    y_te = temp[~train_mask]

    # ── FAST MODEL
    model = RandomForestQuantileRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=seed,
        n_jobs=-1
    )

    model.fit(X_tr, y_tr)

    preds = model.predict(X_te, quantiles=[0.05, 0.10, 0.50, 0.90, 0.95])

    p05 = preds[:,0]
    p10 = preds[:,1]
    mean = preds[:,2]
    p90 = preds[:,3]
    p95 = preds[:,4]

    # ── Fake paths (for compatibility with plots)
    paths = np.tile(mean, (n_paths,1))

    fc = dict(
        mean = mean,
        p05  = p05,
        p10  = p10,
        p90  = p90,
        p95  = p95
    )

    acc = compute_metrics(y_te, fc)

    # LTA
    lta_mo = np.array([
        temp_tr[mo_tr == m].mean() if (mo_tr == m).any() else np.nan
        for m in range(12)
    ])

    # ── Estimate DLM parameters from training data
    # Compute lag-1 autocorrelation from detrended temperature
    temp_tr_detrend = temp_tr - np.polyfit(np.arange(len(temp_tr)), temp_tr, 1)[1]
    acf_lag1 = np.corrcoef(temp_tr_detrend[:-1], temp_tr_detrend[1:])[0, 1]
    acf_lag1 = max(min(acf_lag1, 0.99), 0.01)  # Clamp to reasonable range
    
    # theta = -ln(phi) where phi is lag-1 autocorr
    theta_est = -np.log(acf_lag1)
    
    # sigma = standard deviation of residuals
    resid = y_tr - np.mean(y_tr)
    sigma_est = float(np.std(resid))
    
    dlm_obj = SimpleDLM(theta=theta_est, sigma=sigma_est, q=q)

    return dict(
        paths = paths,
        fc = fc,
        acc = acc,
        dlm = dlm_obj,
        lta_mo = lta_mo,
        temp_tr = temp_tr,
        temp_te = temp_te,
        mo_tr = mo_tr,
        mo_te = mo_te,
        years_all = years,
        months_all = months,
        T_tr = T_tr,
        T_te = T_te,
    )