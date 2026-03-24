"""
enso_helper.py
==============
Loads the NOAA CPC Oceanic Niño Index (ONI) from a CSV file and aligns
it to any (year, month) array used by the rainfall model.

ONI CSV format expected
-----------------------
    SEAS, YR, TOTAL, ANOM
    DJF,  1950, 24.72, -1.53
    ...

The 3-month season labels (DJF, JFM, …, NDJ) are mapped to the
*middle* month they represent, following the convention used in
NOAA's CPC publications and matching the original raind2.py logic:

    DJF → Jan (0)   JFM → Feb (1)   FMA → Mar (2)
    MAM → Apr (3)   AMJ → May (4)   MJJ → Jun (5)
    JJA → Jul (6)   JAS → Aug (7)   ASO → Sep (8)
    SON → Oct (9)   OND → Nov (10)  NDJ → Dec (11) [year - 1]

Public API
----------
load_oni(filepath)
    -> dict  {(year, month_0indexed): anom_float}

align_oni(oni_dict, years, months, standardise=True)
    -> np.ndarray  shape (N,), standardised ENSO anomaly per record
       If oni_dict is None or empty → returns array of zeros.
"""

import csv
import numpy as np

# Season label → 0-indexed month it represents
_SEAS_MO = {
    'DJF':  0, 'JFM':  1, 'FMA':  2,
    'MAM':  3, 'AMJ':  4, 'MJJ':  5,
    'JJA':  6, 'JAS':  7, 'ASO':  8,
    'SON':  9, 'OND': 10, 'NDJ': 11,
}


def load_oni(filepath):
    """
    Parse the ONI CSV and return a dict keyed by (year, month_0indexed).

    The NDJ season (Nov-Dec-Jan) is assigned to December of the *previous*
    year to keep the alignment causal — the same convention used in raind2.py.

    Parameters
    ----------
    filepath : str
        Path to ONI CSV (columns: SEAS, YR, TOTAL, ANOM).

    Returns
    -------
    dict  {(int year, int month 0-indexed): float anomaly}
    """
    oni = {}
    with open(filepath, newline='') as f:
        for row in csv.DictReader(f):
            seas = row['SEAS'].strip()
            yr   = int(row['YR'])
            anom = float(row['ANOM'])
            mo   = _SEAS_MO.get(seas)
            if mo is None:
                continue
            # NDJ: the Jan-centred season belongs to Dec of the prior year
            key = (yr - 1, 11) if seas == 'NDJ' else (yr, mo)
            oni[key] = anom
    if not oni:
        raise ValueError(f"No ONI records parsed from {filepath}. "
                         f"Check the file format (expected columns: SEAS, YR, TOTAL, ANOM).")
    return oni


def align_oni(oni_dict, years, months, standardise=True):
    """
    Align ONI anomalies to a (year, month) time series.

    Missing keys (years/months outside the ONI record) are filled with 0.0
    so the model degrades gracefully near the boundaries.

    Parameters
    ----------
    oni_dict    : dict from load_oni(), or None
    years       : (N,) int array
    months      : (N,) int array, 0-indexed
    standardise : if True, divide by std of the series (makes β unitless)

    Returns
    -------
    enso : (N,) float array
        Standardised (or raw) ONI anomaly aligned to each record.
        All zeros if oni_dict is None or empty.
    """
    N = len(years)

    if not oni_dict:
        return np.zeros(N)

    raw = np.array([oni_dict.get((int(y), int(m)), 0.0)
                    for y, m in zip(years, months)])

    if standardise:
        std = raw.std()
        if std > 0.01:
            raw = raw / std

    return raw


def load_and_align(filepath, years, months, standardise=True):
    """
    Convenience wrapper: load ONI from file and align in one call.

    Parameters
    ----------
    filepath    : str  path to ONI CSV
    years       : (N,) int array
    months      : (N,) int array, 0-indexed
    standardise : bool

    Returns
    -------
    enso : (N,) float array
    oni_dict : the raw dict (useful for diagnostics)
    """
    oni_dict = load_oni(filepath)
    enso     = align_oni(oni_dict, years, months, standardise=standardise)
    n_missing = int(np.sum(enso == 0.0))
    pct_missing = 100 * n_missing / max(len(enso), 1)
    print(f"  [ENSO] Loaded {len(oni_dict)} ONI records from {filepath}")
    if pct_missing > 5:
        print(f"  [ENSO] Warning: {n_missing}/{len(enso)} records "
              f"({pct_missing:.0f}%) filled with 0 (outside ONI range)")
    else:
        print(f"  [ENSO] Aligned to {len(enso)} time steps  "
              f"(standardised={standardise})")
    return enso, oni_dict
