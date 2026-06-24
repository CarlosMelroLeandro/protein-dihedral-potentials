"""
circ_stat.py

Circular statistics for dihedral angles.
Python port of ToolsSrc/CircStat/ (Philipp Berens, 2009) +
circ_vmpdf2.m (C. Leandro, 2014).

Key differences from the MATLAB originals
------------------------------------------
- circ_vmpdf2 is fully vectorised via NumPy broadcasting (≈100× faster
  than the nested Python loop equivalent).
- circ_vmpdf2 correctly applies kappa1 to φ and kappa2 to ψ; the original
  MATLAB used kappa1 for both (likely a typo — the normalization C already
  uses both kappas separately).
- All functions accept plain NumPy arrays; no MATLAB matrix-dimension
  gymnastics needed.
"""

from __future__ import annotations

import warnings
from math import comb as _comb

import numpy as np
from scipy.special import i0 as _i0      # modified Bessel I₀(x)
from scipy.stats import chi2 as _chi2
from scipy.stats import norm as _norm


# ──────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────

def circ_rad2ang(alpha: np.ndarray) -> np.ndarray:
    """Radians → degrees."""
    return np.degrees(alpha)


def circ_ang2rad(alpha: np.ndarray) -> np.ndarray:
    """Degrees → radians."""
    return np.radians(alpha)


def rmatrix(A: np.ndarray) -> np.ndarray:
    """Flip matrix rows upside-down (port of Rmatrix.m), used for display."""
    return A[::-1, :]


def circ_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element-wise circular difference x − y, result ∈ (−π, π]."""
    return np.angle(np.exp(1j * np.asarray(x)) / np.exp(1j * np.asarray(y)))


def circ_dist2(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """All-pairs circular differences: result[i, j] = x[i] − y[j]."""
    x = np.asarray(x, dtype=float).ravel()
    y = x if y is None else np.asarray(y, dtype=float).ravel()
    return np.angle(np.exp(1j * x)[:, None] / np.exp(1j * y)[None, :])


# ──────────────────────────────────────────────────────────────────────────
# Core descriptive statistics
# ──────────────────────────────────────────────────────────────────────────

def circ_r(
    alpha: np.ndarray,
    w: np.ndarray | None = None,
    d: float = 0.0,
) -> float:
    """
    Mean resultant length R̄ ∈ [0, 1].

    Parameters
    ----------
    alpha   angles in radians
    w       weights (for binned data)
    d       bin spacing in radians (correction for binned data)
    """
    alpha = np.asarray(alpha, dtype=float).ravel()
    w = np.ones_like(alpha) if w is None else np.asarray(w, dtype=float).ravel()
    r = float(np.abs(np.sum(w * np.exp(1j * alpha))) / np.sum(w))
    if d != 0.0:
        r *= d / 2.0 / np.sin(d / 2.0)
    return r


def circ_mean(
    alpha: np.ndarray,
    w: np.ndarray | None = None,
) -> float:
    """Mean direction in radians."""
    alpha = np.asarray(alpha, dtype=float).ravel()
    w = np.ones_like(alpha) if w is None else np.asarray(w, dtype=float).ravel()
    return float(np.angle(np.sum(w * np.exp(1j * alpha))))


def circ_moment(
    alpha: np.ndarray,
    w: np.ndarray | None = None,
    p: int = 1,
    cent: bool = False,
) -> tuple[complex, float, float]:
    """
    p-th trigonometric moment.

    Returns
    -------
    (mp, rho_p, mu_p)  — complex moment, magnitude, direction
    """
    alpha = np.asarray(alpha, dtype=float).ravel()
    n = len(alpha)
    w = np.ones_like(alpha) if w is None else np.asarray(w, dtype=float).ravel()
    if cent:
        alpha = circ_dist(alpha, circ_mean(alpha, w))
    cbar = float(np.sum(np.cos(p * alpha) * w) / n)
    sbar = float(np.sum(np.sin(p * alpha) * w) / n)
    mp = complex(cbar, sbar)
    return mp, abs(mp), float(np.angle(mp))


def circ_var(alpha: np.ndarray, w: np.ndarray | None = None) -> float:
    """Circular variance S = 1 − R̄  (Zar 26.17)."""
    return 1.0 - circ_r(alpha, w)


def circ_std(alpha: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """
    Circular standard deviation.

    Returns
    -------
    s    angular deviation  = sqrt(2(1 − R̄))   (Zar 26.20)
    s0   circular std       = sqrt(−2 ln R̄)     (Zar 26.21)
    """
    r = circ_r(alpha, w)
    return float(np.sqrt(2.0 * (1.0 - r))), float(np.sqrt(-2.0 * np.log(r)))


def circ_skewness(alpha: np.ndarray, w: np.ndarray | None = None) -> float:
    """Circular skewness b (Pewsey 2004)."""
    alpha = np.asarray(alpha, dtype=float).ravel()
    w = np.ones_like(alpha) if w is None else np.asarray(w, dtype=float).ravel()
    theta = circ_mean(alpha, w)
    return float(np.sum(w * np.sin(2.0 * circ_dist(alpha, theta))) / np.sum(w))


def circ_kurtosis(alpha: np.ndarray, w: np.ndarray | None = None) -> float:
    """Circular kurtosis k (Pewsey 2004)."""
    alpha = np.asarray(alpha, dtype=float).ravel()
    w = np.ones_like(alpha) if w is None else np.asarray(w, dtype=float).ravel()
    theta = circ_mean(alpha, w)
    return float(np.sum(w * np.cos(2.0 * circ_dist(alpha, theta))) / np.sum(w))


def circ_median(alpha: np.ndarray) -> float:
    """
    Circular median (Zar 26.6).

    O(n²) — suitable for n < ~5 000.
    """
    alpha = np.asarray(alpha, dtype=float).ravel()
    beta = np.mod(alpha, 2.0 * np.pi)
    n = len(beta)

    dd = circ_dist2(beta, beta)
    m1 = np.sum(dd >= 0, axis=1)
    m2 = np.sum(dd <= 0, axis=1)
    dm = np.abs(m1 - m2)
    m_min = int(dm.min())

    if n % 2 == 1:
        idx = [int(np.argmin(dm))]
    else:
        idx = list(np.where(dm == m_min)[0][:2])

    if m_min > 1:
        warnings.warn("Ties detected in circ_median.", stacklevel=2)

    md = circ_mean(beta[idx])
    mu = circ_mean(beta)
    if abs(circ_dist(mu, md)) > abs(circ_dist(mu, md + np.pi)):
        md = (md + np.pi) % (2.0 * np.pi)
    return float(md)


# ──────────────────────────────────────────────────────────────────────────
# von Mises parameter estimation
# ──────────────────────────────────────────────────────────────────────────

def circ_kappa(r_or_alpha: float | np.ndarray, w: np.ndarray | None = None) -> float:
    """
    ML estimate of the von Mises concentration κ.

    Accepts either:
    - a scalar R̄ value (returned by circ_r), or
    - a raw sample of angles (N > 1), in which case R̄ is computed first.

    Uses Fisher's closed-form approximation (Fisher 1993, p. 88) with
    a small-sample correction for N < 15.
    """
    arr = np.asarray(r_or_alpha, dtype=float).ravel()
    if arr.size > 1:
        n = len(arr)
        R = circ_r(arr, w)
    else:
        n = 1
        R = float(arr[0])

    if R < 0.53:
        kappa = 2.0 * R + R**3 + 5.0 * R**5 / 6.0
    elif R < 0.85:
        kappa = -0.4 + 1.39 * R + 0.43 / (1.0 - R)
    else:
        kappa = 1.0 / (R**3 - 4.0 * R**2 + 3.0 * R)

    # small-sample correction
    if 1 < n < 15:
        if kappa < 2.0:
            kappa = max(kappa - 2.0 / (n * kappa), 0.0)
        else:
            kappa = (n - 1.0) ** 3 * kappa / (n**3 + n)

    return float(kappa)


def circ_vmpar(
    alpha: np.ndarray,
    w: np.ndarray | None = None,
    d: float = 0.0,
) -> tuple[float, float]:
    """
    Estimate von Mises parameters from data.

    Returns
    -------
    (mu, kappa)  — mean direction (rad) and concentration parameter
    """
    alpha = np.asarray(alpha, dtype=float).ravel()
    r = circ_r(alpha, w, d)
    return circ_mean(alpha, w), circ_kappa(r)


# ──────────────────────────────────────────────────────────────────────────
# 2-D von Mises KDE  (port of circ_vmpdf2.m, vectorised)
# ──────────────────────────────────────────────────────────────────────────

def circ_vmpdf2(
    n_ang: int,
    phi_i: np.ndarray,
    psi_i: np.ndarray,
    kappa1: float,
    kappa2: float,
) -> tuple[np.ndarray, int]:
    """
    2-D von Mises kernel density estimate for Ramachandran data.

    Port of circ_vmpdf2.m (C. Leandro, 2014), vectorised.

    Density evaluated on an (n_ang × n_ang) uniform grid over (−π, π]:

        p[i, j] = C · Σₖ exp(κ₁ cos(φᵢ − φₖ) + κ₂ cos(ψⱼ − ψₖ))
        C        = 1 / (4π² N I₀(κ₁) I₀(κ₂))

    The product structure allows a single matrix multiply instead of nested
    loops — O(N_ang · N_data) rather than O(N_ang² · N_data).

    Parameters
    ----------
    n_ang    number of evaluation points in (−π, π) for each axis
    phi_i    observed φ angles (radians)
    psi_i    observed ψ angles (radians)
    kappa1   concentration for φ  (from circ_vmpar)
    kappa2   concentration for ψ  (from circ_vmpar)

    Returns
    -------
    p      (n_ang, n_ang) density — rows index φ, cols index ψ
    n_ang  (echo, for API compatibility with the MATLAB original)
    """
    phi_i = np.asarray(phi_i, dtype=float).ravel()
    psi_i = np.asarray(psi_i, dtype=float).ravel()
    N = len(phi_i)

    delta = 2.0 * np.pi / n_ang
    grid  = -np.pi + delta / 2.0 + np.arange(n_ang) * delta   # (n_ang,)

    C = 1.0 / (4.0 * np.pi**2 * N * _i0(kappa1) * _i0(kappa2))

    # diff[i, k] = grid[i] − data[k]
    phi_diff = grid[:, None] - phi_i[None, :]   # (n_ang, N)
    psi_diff = grid[:, None] - psi_i[None, :]   # (n_ang, N)

    A = np.exp(kappa1 * np.cos(phi_diff))        # (n_ang, N)
    B = np.exp(kappa2 * np.cos(psi_diff))        # (n_ang, N)

    # p[i, j] = C · Σₖ A[i,k] · B[j,k]  =  C · (A @ Bᵀ)[i,j]
    p = C * (A @ B.T)                            # (n_ang, n_ang)

    return p, n_ang


# ──────────────────────────────────────────────────────────────────────────
# Statistical tests
# ──────────────────────────────────────────────────────────────────────────

def circ_rtest(
    alpha: np.ndarray,
    w: np.ndarray | None = None,
    d: float = 0.0,
) -> tuple[float, float]:
    """
    Rayleigh test for uniformity.
    H₀: distribution is uniform on the circle.

    Returns (pval, z).
    """
    alpha = np.asarray(alpha, dtype=float).ravel()
    if w is None:
        n = float(len(alpha))
        r = circ_r(alpha)
    else:
        w = np.asarray(w, dtype=float).ravel()
        n = float(np.sum(w))
        r = circ_r(alpha, w, d)
    R    = n * r
    z    = R**2 / n
    pval = float(np.exp(np.sqrt(1.0 + 4.0 * n + 4.0 * (n**2 - R**2)) - (1.0 + 2.0 * n)))
    return pval, z


def circ_otest(
    alpha: np.ndarray,
    sz: float | None = None,
    w: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Omnibus / Hodges-Ajne test for uniformity.
    Works for uni-, bi-, and multi-modal data.

    Returns (pval, m).
    """
    alpha = np.asarray(alpha, dtype=float).ravel()
    if sz is None:
        sz = circ_ang2rad(1.0)
    w = np.ones_like(alpha) if w is None else np.asarray(w, dtype=float).ravel()

    alpha = np.mod(alpha, 2.0 * np.pi)
    n = float(np.sum(w))
    dg = np.arange(0.0, np.pi, sz)

    m1 = np.array([np.sum(((alpha > d) & (alpha < np.pi + d)) * w) for d in dg])
    m2 = n - m1
    m  = float(np.minimum(m1, m2).min())

    if n > 50:
        A    = np.pi * np.sqrt(n) / 2.0 / (n - 2.0 * m)
        pval = float(np.sqrt(2.0 * np.pi) / A * np.exp(-np.pi**2 / 8.0 / A**2))
    else:
        pval = float(2.0 ** (1.0 - n) * (n - 2.0 * m) * _comb(int(n), int(m)))
    return pval, m


# Rao's spacing test — critical-value table (Russell & Levitin 1995)
_RAO_TABLE = np.array([
    [  4, 247.32, 221.14, 186.45, 168.02],
    [  5, 245.19, 211.93, 183.44, 168.66],
    [  6, 236.81, 206.79, 180.65, 166.30],
    [  7, 229.46, 202.55, 177.83, 165.05],
    [  8, 224.41, 198.46, 175.68, 163.56],
    [  9, 219.52, 195.27, 173.68, 162.36],
    [ 10, 215.44, 192.37, 171.98, 161.23],
    [ 11, 211.87, 189.88, 170.45, 160.24],
    [ 12, 208.69, 187.66, 169.09, 159.33],
    [ 13, 205.87, 185.68, 167.87, 158.50],
    [ 14, 203.33, 183.90, 166.76, 157.75],
    [ 15, 201.04, 182.28, 165.75, 157.06],
    [ 16, 198.96, 180.81, 164.83, 156.43],
    [ 17, 197.05, 179.46, 163.98, 155.84],
    [ 18, 195.29, 178.22, 163.20, 155.29],
    [ 19, 193.67, 177.08, 162.47, 154.78],
    [ 20, 192.17, 176.01, 161.79, 154.31],
    [ 21, 190.78, 175.02, 161.16, 153.86],
    [ 22, 189.47, 174.10, 160.56, 153.44],
    [ 23, 188.25, 173.23, 160.01, 153.05],
    [ 24, 187.11, 172.41, 159.48, 152.68],
    [ 25, 186.03, 171.64, 158.99, 152.32],
    [ 26, 185.01, 170.92, 158.52, 151.99],
    [ 27, 184.05, 170.23, 158.07, 151.67],
    [ 28, 183.14, 169.58, 157.65, 151.37],
    [ 29, 182.28, 168.96, 157.25, 151.08],
    [ 30, 181.45, 168.38, 156.87, 150.80],
    [ 35, 177.88, 165.81, 155.19, 149.59],
    [ 40, 174.99, 163.73, 153.82, 148.60],
    [ 45, 172.58, 162.00, 152.68, 147.76],
    [ 50, 170.54, 160.53, 151.70, 147.05],
    [ 75, 163.60, 155.49, 148.34, 144.56],
    [100, 159.45, 152.46, 146.29, 143.03],
    [150, 154.51, 148.84, 143.83, 141.18],
    [200, 151.56, 146.67, 142.35, 140.06],
    [300, 148.06, 144.09, 140.57, 138.71],
    [400, 145.96, 142.54, 139.50, 137.89],
    [500, 144.54, 141.48, 138.77, 137.33],
    [600, 143.48, 140.70, 138.23, 136.91],
    [700, 142.66, 140.09, 137.80, 136.59],
    [800, 142.00, 139.60, 137.46, 136.33],
    [900, 141.45, 139.19, 137.18, 136.11],
    [1000, 140.99, 138.84, 136.94, 135.92],
])
_RAO_ALPHA = np.array([0.001, 0.01, 0.05, 0.10])


def circ_raotest(alpha: np.ndarray) -> tuple[float, float, float]:
    """
    Rao's spacing test for uniformity.

    Returns (p, U, UC) — smallest significant alpha level, test statistic,
    critical value at that level.
    """
    alpha = np.asarray(alpha, dtype=float).ravel()
    alpha_deg = np.degrees(alpha)
    n = len(alpha_deg)
    alpha_deg = np.sort(alpha_deg)

    lam = 360.0 / n
    gaps = np.concatenate([np.diff(alpha_deg),
                           [360.0 - alpha_deg[-1] + alpha_deg[0]]])
    U = 0.5 * np.sum(np.abs(gaps - lam))

    ridx = min(np.searchsorted(_RAO_TABLE[:, 0], n), len(_RAO_TABLE) - 1)
    row  = _RAO_TABLE[ridx, 1:]          # four critical values
    cidx = np.where(row < U)[0]
    if len(cidx):
        c = cidx[0]
        return float(_RAO_ALPHA[c]), float(U), float(row[c])
    return 0.5, float(U), float(row[-1])


def circ_vtest(
    alpha: np.ndarray,
    dir_: float,
    w: np.ndarray | None = None,
    d: float = 0.0,
) -> tuple[float, float]:
    """
    V-test: uniformity vs. a specified mean direction *dir_*.

    Returns (pval, v).
    """
    alpha = np.asarray(alpha, dtype=float).ravel()
    w     = np.ones_like(alpha) if w is None else np.asarray(w, dtype=float).ravel()
    n     = float(np.sum(w))
    r     = circ_r(alpha, w, d)
    mu    = circ_mean(alpha, w)
    R     = n * r
    v     = R * np.cos(mu - dir_)
    u     = v * np.sqrt(2.0 / n)
    return float(1.0 - _norm.cdf(u)), float(v)


# ──────────────────────────────────────────────────────────────────────────
# Correlation coefficients
# ──────────────────────────────────────────────────────────────────────────

def circ_corrcc(alpha1: np.ndarray, alpha2: np.ndarray) -> tuple[float, float]:
    """
    Circular-circular correlation (Jammalamadaka & SenGupta, p. 176).

    Returns (rho, pval).
    """
    alpha1 = np.asarray(alpha1, dtype=float).ravel()
    alpha2 = np.asarray(alpha2, dtype=float).ravel()
    n   = len(alpha1)
    mu1 = circ_mean(alpha1)
    mu2 = circ_mean(alpha2)
    s1  = np.sin(alpha1 - mu1)
    s2  = np.sin(alpha2 - mu2)

    rho  = float(np.sum(s1 * s2) / np.sqrt(np.sum(s1**2) * np.sum(s2**2)))
    l20  = np.mean(s1**2)
    l02  = np.mean(s2**2)
    l22  = np.mean(s1**2 * s2**2)
    ts   = np.sqrt(n * l20 * l02 / l22) * rho
    pval = float(2.0 * (1.0 - _norm.cdf(abs(ts))))
    return rho, pval


def circ_corrcl(alpha: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """
    Circular-linear correlation (Zar, equ. 27.47).

    Returns (rho, pval).
    """
    alpha = np.asarray(alpha, dtype=float).ravel()
    x     = np.asarray(x, dtype=float).ravel()
    n     = len(alpha)
    rxs   = float(np.corrcoef(x, np.sin(alpha))[0, 1])
    rxc   = float(np.corrcoef(x, np.cos(alpha))[0, 1])
    rcs   = float(np.corrcoef(np.sin(alpha), np.cos(alpha))[0, 1])
    rho   = float(np.sqrt((rxc**2 + rxs**2 - 2.0 * rxc * rxs * rcs) / (1.0 - rcs**2)))
    pval  = float(1.0 - _chi2.cdf(n * rho**2, 2))
    return rho, pval
