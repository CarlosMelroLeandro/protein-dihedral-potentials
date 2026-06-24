"""
analysis.py

Statistical potential computation — Phase 2:
  Load per-amino CSV  →  fit 2D von Mises KDE  →  compute -log(p) maps
  and their gradients  →  save .npz files to contData/.

Port of ToolsSrc/ForcesCont.m + ToolsSrc/Analysis.m (statistical section).

Dihedral pairs computed (when enough data is available):
  1. phi–psi          (Ramachandran backbone)
  2. phi–chi1         (backbone–rotamer coupling)
  3. psi–chi1
  4. chi1–chi2        (rotamer–rotamer coupling)
  5. chi2–chi3
  6. chi3–chi4

Output files
------------
contData/Mises<AMINO><n_ang>.npz  — compressed NumPy archive with:
  n_total        int      total residues loaded from CSV
  n_ang          int      KDE grid resolution
  n_potentials   int      number of dihedral pairs computed
  pair_k         (2,)     angle names for pair k  (e.g. ['phi', 'psi'])
  E_k            (n,n)    -log p  potential map for pair k
  Gx_k           (n,n)    gradient in x (column) direction
  Gy_k           (n,n)    gradient in y (row) direction
  kappa_k        (2,)     fitted von Mises concentrations [kappa1, kappa2]
  n_k            int      residues used for pair k
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .circ_stat import circ_vmpar, circ_vmpdf2
from .circ_stat import (
    circ_mean, circ_r, circ_var, circ_std,
    circ_skewness, circ_kurtosis,
)
from .pdb_tools import AMINO_ACIDS

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 5   # minimum points required to compute a potential

SINGLE_COLS = [
    "phi", "psi", "dC_CA", "dN_CA", "dN_C",
    "dP_plane", "ang_N_CA_C", "tempFactor",
    "chi1", "chi2", "chi3", "chi4",
]

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_amino_data(csv_path: str | Path) -> np.ndarray | None:
    """
    Load a per-amino CSV file produced by extractor.py.

    Returns an (N, 12) float array with columns:
      phi, psi, dC_CA, dN_CA, dN_C, dP_plane, ang_N_CA_C, tempFactor,
      chi1, chi2, chi3, chi4

    Rows with NaN phi or NaN psi are dropped.
    Returns None if the file is absent or empty after filtering.
    """
    path = Path(csv_path)
    if not path.exists():
        return None

    data = np.genfromtxt(str(path), delimiter=",", dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    if data.size == 0:
        return None

    # Drop rows where phi (col 0) or psi (col 1) is NaN
    mask = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    data = data[mask]
    return data if len(data) > 0 else None


# ──────────────────────────────────────────────────────────────────────────────
# Core potential computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_potential(
    a1: np.ndarray,
    a2: np.ndarray,
    n_ang: int = 180,
) -> dict | None:
    """
    Compute 2D von Mises KDE and return the -log(p) potential and gradients.

    Parameters
    ----------
    a1, a2   1-D angle arrays (radians); NaN values are removed pairwise.
    n_ang    KDE grid resolution (number of bins per axis).

    Returns
    -------
    dict with keys E, Gx, Gy, kappa1, kappa2, n   or   None (too few data).
    """
    mask = np.isfinite(a1) & np.isfinite(a2)
    a1, a2 = a1[mask], a2[mask]
    if len(a1) < _MIN_SAMPLES:
        return None

    _, kappa1 = circ_vmpar(a1)
    _, kappa2 = circ_vmpar(a2)

    p, _ = circ_vmpdf2(n_ang, a1, a2, kappa1, kappa2)

    # Clamp to avoid log(0)
    p_floor = float(p[p > 0].min()) * 1e-3 if (p > 0).any() else 1e-30
    E = -np.log(np.maximum(p, p_floor))

    # Gradient: column direction (x) and row direction (y)
    # Equivalent to MATLAB imgradientxy (central differences)
    Gx = np.gradient(E, axis=1)
    Gy = np.gradient(E, axis=0)

    return {
        "E": E.astype(np.float64),
        "Gx": Gx.astype(np.float64),
        "Gy": Gy.astype(np.float64),
        "kappa1": float(kappa1),
        "kappa2": float(kappa2),
        "n": int(len(a1)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-amino analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyse_amino(
    amino_name: str,
    csv_path: str | Path,
    n_ang: int = 180,
) -> dict | None:
    """
    Compute all dihedral pair potentials for one amino acid.

    Port of ToolsSrc/ForcesCont.m.

    Parameters
    ----------
    amino_name   3-letter code (e.g. "ALA")
    csv_path     path to the per-amino CSV produced by extractor.py
    n_ang        KDE grid resolution (default 180, matching MATLAB binAng=180)

    Returns
    -------
    dict with keys:
        amino        str
        n_ang        int
        n_total      int  — total rows loaded from CSV
        potentials   list[dict]  — one entry per dihedral pair

    Each potential dict has:
        pair         (str, str)  — angle names
        E            (n_ang, n_ang)  — -log p map
        Gx, Gy       (n_ang, n_ang)  — gradient components
        kappa1, kappa2  float
        n            int  — number of data points used

    Returns None if the CSV is absent or no rows survive filtering.
    """
    data = load_amino_data(csv_path)
    if data is None:
        logger.warning("%s: no data loaded from %s", amino_name, csv_path)
        return None

    n_total = len(data)
    phi  = data[:, 0]
    psi  = data[:, 1]
    chi1 = data[:, 8]
    chi2 = data[:, 9]
    chi3 = data[:, 10]
    chi4 = data[:, 11]

    potentials: list[dict] = []

    def _add(pair_names, a1, a2):
        res = compute_potential(a1, a2, n_ang)
        if res is not None:
            res["pair"] = pair_names
            potentials.append(res)
            logger.debug(
                "%s: %s  n=%d  κ₁=%.2f  κ₂=%.2f",
                amino_name, pair_names, res["n"], res["kappa1"], res["kappa2"],
            )

    # ── 1: phi–psi ─────────────────────────────────────────────────────────
    logger.info("%s: computing phi–psi", amino_name)
    _add(("phi", "psi"), phi, psi)

    # ── 2–3: phi/psi – chi1 ────────────────────────────────────────────────
    if np.isfinite(chi1).sum() >= _MIN_SAMPLES:
        logger.info("%s: computing phi–chi1 / psi–chi1", amino_name)
        _add(("phi", "chi1"), phi, chi1)
        _add(("psi", "chi1"), psi, chi1)

    # ── 4: chi1–chi2 ───────────────────────────────────────────────────────
    if np.isfinite(chi2).sum() >= _MIN_SAMPLES:
        logger.info("%s: computing chi1–chi2", amino_name)
        _add(("chi1", "chi2"), chi1, chi2)

    # ── 5: chi2–chi3 ───────────────────────────────────────────────────────
    if np.isfinite(chi3).sum() >= _MIN_SAMPLES:
        logger.info("%s: computing chi2–chi3", amino_name)
        _add(("chi2", "chi3"), chi2, chi3)

    # ── 6: chi3–chi4 ───────────────────────────────────────────────────────
    if np.isfinite(chi4).sum() >= _MIN_SAMPLES:
        logger.info("%s: computing chi3–chi4", amino_name)
        _add(("chi3", "chi4"), chi3, chi4)

    logger.info(
        "%s: %d potentials computed from %d residues",
        amino_name, len(potentials), n_total,
    )
    return {
        "amino": amino_name,
        "n_ang": n_ang,
        "n_total": n_total,
        "potentials": potentials,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Batch analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyse_all(
    amino_dir: str | Path,
    cont_dir: str | Path | None = None,
    *,
    n_ang: int = 180,
    amino_list: list[str] | None = None,
) -> dict[str, dict]:
    """
    Run :func:`analyse_amino` for every amino acid whose CSV is present.

    Parameters
    ----------
    amino_dir    directory containing <AMINO>.csv files
    cont_dir     if given, each result is saved as a .npz file (may be None)
    n_ang        KDE grid resolution
    amino_list   subset of amino acids to process (default: all 20)

    Returns
    -------
    dict mapping amino-acid name → result dict (same as analyse_amino)
    """
    amino_dir = Path(amino_dir)
    if amino_list is None:
        amino_list = list(AMINO_ACIDS)

    results: dict[str, dict] = {}
    for amino in amino_list:
        csv_path = amino_dir / f"{amino}.csv"
        if not csv_path.exists():
            logger.debug("%s: CSV absent — skipped", amino)
            continue

        result = analyse_amino(amino, csv_path, n_ang=n_ang)
        if result is None:
            continue

        results[amino] = result
        if cont_dir is not None:
            save_potential(result, cont_dir)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Save / load .npz potential files
# ──────────────────────────────────────────────────────────────────────────────

def save_potential(result: dict, cont_dir: str | Path) -> Path:
    """
    Save one amino acid's analysis result to a compressed .npz file.

    File: <cont_dir>/Mises<AMINO><n_ang>.npz
    """
    cont_dir = Path(cont_dir)
    cont_dir.mkdir(parents=True, exist_ok=True)

    amino = result["amino"]
    n_ang = result["n_ang"]
    path = cont_dir / f"Mises{amino}{n_ang}.npz"

    arrays: dict[str, np.ndarray] = {
        "n_total": np.array(result["n_total"], dtype=np.int64),
        "n_ang": np.array(n_ang, dtype=np.int64),
        "n_potentials": np.array(len(result["potentials"]), dtype=np.int64),
    }
    for k, pot in enumerate(result["potentials"]):
        p1, p2 = pot["pair"]
        arrays[f"pair_{k}"]  = np.array([p1, p2])
        arrays[f"E_{k}"]     = pot["E"]
        arrays[f"Gx_{k}"]    = pot["Gx"]
        arrays[f"Gy_{k}"]    = pot["Gy"]
        arrays[f"kappa_{k}"] = np.array([pot["kappa1"], pot["kappa2"]])
        arrays[f"n_{k}"]     = np.array(pot["n"], dtype=np.int64)

    np.savez_compressed(path, **arrays)
    logger.debug("Saved %s", path)
    return path


def load_potential(
    cont_dir: str | Path,
    amino_name: str,
    n_ang: int = 180,
) -> dict | None:
    """
    Load a previously saved potential from a .npz file.

    Returns None if the file does not exist.
    """
    path = Path(cont_dir) / f"Mises{amino_name}{n_ang}.npz"
    if not path.exists():
        return None

    raw = np.load(path, allow_pickle=False)
    n_pot = int(raw["n_potentials"])

    potentials = []
    for k in range(n_pot):
        pair = tuple(str(s) for s in raw[f"pair_{k}"])
        kappas = raw[f"kappa_{k}"]
        potentials.append({
            "pair": pair,
            "E":  raw[f"E_{k}"],
            "Gx": raw[f"Gx_{k}"],
            "Gy": raw[f"Gy_{k}"],
            "kappa1": float(kappas[0]),
            "kappa2": float(kappas[1]),
            "n": int(raw[f"n_{k}"]),
        })

    return {
        "amino": amino_name,
        "n_ang": int(raw["n_ang"]),
        "n_total": int(raw["n_total"]),
        "potentials": potentials,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Descriptive circular statistics (port of Analysis.m statistical section)
# ──────────────────────────────────────────────────────────────────────────────

def circular_stats(alpha: np.ndarray) -> dict:
    """
    Compute descriptive circular statistics for an angle array.

    Returns a dict with keys:
        n, mean_deg, R, var, std_deg, std0_deg, skewness, kurtosis
    Returns an empty dict if fewer than 2 finite values are present.
    """
    valid = np.asarray(alpha).ravel()
    valid = valid[np.isfinite(valid)]
    if len(valid) < 2:
        return {}

    mu = circ_mean(valid)
    R  = circ_r(valid)
    s, s0 = circ_std(valid)

    return {
        "n":         int(len(valid)),
        "mean_deg":  float(np.degrees(mu)),
        "R":         float(R),
        "var":       float(circ_var(valid)),
        "std_deg":   float(np.degrees(s)),
        "std0_deg":  float(np.degrees(s0)),
        "skewness":  float(circ_skewness(valid)),
        "kurtosis":  float(circ_kurtosis(valid)),
    }


def amino_stats(csv_path: str | Path) -> dict:
    """
    Full descriptive statistics for one amino acid CSV.

    Returns a dict with keys:
        n_total,
        phi, psi       — circular_stats dicts
        chi1..chi4     — circular_stats dicts (empty if not present)
        dC_CA, dN_CA, dP_plane  — linear stats dicts
    """
    data = load_amino_data(csv_path)
    if data is None:
        return {}

    def _lin(col):
        x = data[:, col]
        x = x[np.isfinite(x)]
        if len(x) < 2:
            return {}
        return {
            "n": len(x), "mean": float(np.mean(x)),
            "median": float(np.median(x)), "std": float(np.std(x)),
            "min": float(np.min(x)), "max": float(np.max(x)),
        }

    return {
        "n_total":  len(data),
        "phi":      circular_stats(data[:, 0]),
        "psi":      circular_stats(data[:, 1]),
        "dC_CA":    _lin(2),
        "dN_CA":    _lin(3),
        "dP_plane": _lin(5),
        "chi1":     circular_stats(data[:, 8]),
        "chi2":     circular_stats(data[:, 9]),
        "chi3":     circular_stats(data[:, 10]),
        "chi4":     circular_stats(data[:, 11]),
    }
