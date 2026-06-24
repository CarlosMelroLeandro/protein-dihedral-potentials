"""
pdb_tools.py

Backbone and side-chain dihedral geometry for PDB structures.
Python port of ToolsSrc/pdbTools/:
  calcDihedral.m, calcDihedrals.m, findAngle.m, Torsion.m, Rotamer.m

Column layout of per-residue output (mirrors aminoData/<AMINO>.csv):
  0  phi          backbone dihedral (rad)
  1  psi          backbone dihedral (rad)
  2  dC_CA        C–CA bond length (Å)
  3  dN_CA        N–CA bond length (Å)
  4  dN_C         intra-residue N–C distance (Å)
  5  dP_plane     peptide-bond length prev-C → curr-N (Å)
  6  ang_N_CA_C   N–CA–C angle (rad)
  7  tempFactor   max B-factor in window of 3 residues
  8  chi1 … chi4  side-chain dihedrals (rad, NaN when absent)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUTOFF: float = 6.5  # Å – bond-length sanity check (matches MATLAB default)

AMINO_ACIDS: list[str] = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

# Chi-angle atom quadruples, per residue (from Rotamer.m)
CHIS: dict[str, list[tuple[str, str, str, str]]] = {
    "ARG": [("N", "CA", "CB", "CG"),  ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")],
    "ASN": [("N", "CA", "CB", "CG"),  ("CA", "CB", "CG", "OD1")],
    "ASP": [("N", "CA", "CB", "CG"),  ("CA", "CB", "CG", "OD1")],
    "CYS": [("N", "CA", "CB", "SG")],
    "GLN": [("N", "CA", "CB", "CG"),  ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")],
    "GLU": [("N", "CA", "CB", "CG"),  ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")],
    "HIS": [("N", "CA", "CB", "CG"),  ("CA", "CB", "CG", "ND1")],
    "ILE": [("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")],
    "LEU": [("N", "CA", "CB", "CG"),  ("CA", "CB", "CG", "CD1")],
    "LYS": [("N", "CA", "CB", "CG"),  ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")],
    "MET": [("N", "CA", "CB", "CG"),  ("CA", "CB", "CG", "SD"),
            ("CB", "CG", "SD", "CE")],
    "PHE": [("N", "CA", "CB", "CG"),  ("CA", "CB", "CG", "CD1")],
    "PRO": [("N", "CA", "CB", "CG"),  ("CA", "CB", "CG", "CD")],
    "SER": [("N", "CA", "CB", "OG")],
    "THR": [("N", "CA", "CB", "OG1")],
    "TRP": [("N", "CA", "CB", "CG"),  ("CA", "CB", "CG", "CD1")],
    "TYR": [("N", "CA", "CB", "CG"),  ("CA", "CB", "CG", "CD1")],
    "VAL": [("N", "CA", "CB", "CG1")],
}


# ---------------------------------------------------------------------------
# Geometry primitives  (port of findAngle.m / calcDihedral.m)
# ---------------------------------------------------------------------------

def find_angle(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in radians between two 3-D vectors (port of findAngle.m)."""
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0.0:
        return np.nan
    return float(np.arccos(np.clip(np.dot(u, v) / denom, -1.0, 1.0)))


def calc_dihedral(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    cutoff: float = np.inf,
) -> float:
    """
    Signed dihedral angle (radians) defined by four points.
    Returns NaN when any sequential bond exceeds *cutoff* Å.

    Port of calcDihedral.m — sign convention is identical:
    looking down p2→p3, positive = clockwise from the p1-plane to the p4-plane.
    """
    ab = p1 - p2
    cb = p3 - p2
    db = p4 - p3

    if cutoff < np.inf:
        lengths = (np.linalg.norm(ab), np.linalg.norm(cb), np.linalg.norm(db))
        if max(lengths) > cutoff:
            return np.nan

    u = np.cross(ab, cb)
    v = np.cross(db, cb)
    w = np.cross(u, v)

    ang = find_angle(u, v)

    # Sign: if cb and w point in opposite directions, negate
    if find_angle(cb, w) > 0.001:
        ang = -ang

    return float(ang)


def calc_dihedrals(
    prev_c: np.ndarray,
    curr_n: np.ndarray,
    curr_ca: np.ndarray,
    curr_c: np.ndarray,
    next_n: np.ndarray,
    cutoff: float = CUTOFF,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Backbone angles and bond metrics for one residue.
    Port of calcDihedrals.m.

    Returns
    -------
    phi, psi, dC_CA, dN_CA, dN_C, dP_plane, ang_N_CA_C
    """
    b = curr_n - curr_ca    # N→CA
    c = curr_c - curr_ca    # C→CA
    e = curr_n - curr_c     # intra-residue N–C vector
    f = prev_c - curr_n     # peptide-bond vector (prev C → curr N)

    dC_CA      = float(np.linalg.norm(c))
    dN_CA      = float(np.linalg.norm(b))
    dN_C       = float(np.linalg.norm(e))
    dP_plane   = float(np.linalg.norm(f))
    ang_N_CA_C = find_angle(c, b)

    phi = calc_dihedral(prev_c, curr_n, curr_ca, curr_c, cutoff=cutoff)
    psi = calc_dihedral(curr_n, curr_ca, curr_c, next_n, cutoff=cutoff)

    return phi, psi, dC_CA, dN_CA, dN_C, dP_plane, ang_N_CA_C


# ---------------------------------------------------------------------------
# Structure-level extraction  (port of Torsion.m / Rotamer.m)
# ---------------------------------------------------------------------------

def _iter_standard_residues(structure):
    """Yield (model_id, chain_id, residue) for ATOM records only."""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":   # ' ' = ATOM record (not HETATM)
                    yield model.id, chain.id, residue


def torsion(structure) -> list[dict]:
    """
    Backbone torsion angles and distances for every standard residue.
    Port of Torsion.m.

    Parameters
    ----------
    structure : Bio.PDB.Structure

    Returns
    -------
    List of dicts with keys:
        chain, resName, model, resSeq,
        phi, psi, omega,
        dC_CA, dN_CA, dN_C, dP_plane, ang_N_CA_C,
        tempFactor
    All angles in radians; NaN at chain boundaries or when atoms are missing.
    """
    # ---- Pass 1: collect backbone atom coords in residue order -------------
    records: list[tuple] = []
    # Each record: (model_id, chain_id, resname, resseq, N, CA, C, max_bf)

    for mid, cid, residue in _iter_standard_residues(structure):
        resname  = residue.resname.strip()
        n_xyz = ca_xyz = c_xyz = None
        max_bf = 0.0

        for atom in residue.get_atoms():
            bf = atom.bfactor or 0.0
            max_bf = max(max_bf, bf)
            name = atom.get_name().strip()
            if   name == "N":  n_xyz  = atom.coord.copy()
            elif name == "CA": ca_xyz = atom.coord.copy()
            elif name == "C":  c_xyz  = atom.coord.copy()

        records.append((mid, cid, resname, residue.id[1],
                        n_xyz, ca_xyz, c_xyz, max_bf))

    # ---- Pass 2: compute angles -------------------------------------------
    results: list[dict] = []
    n = len(records)

    for i, (mid, cid, resname, resseq, n_xyz, ca_xyz, c_xyz, bf) in enumerate(records):
        entry: dict = dict(
            chain=cid, resName=resname, model=mid, resSeq=resseq,
            phi=np.nan, psi=np.nan, omega=np.nan,
            dC_CA=np.nan, dN_CA=np.nan, dN_C=np.nan,
            dP_plane=np.nan, ang_N_CA_C=np.nan,
            tempFactor=bf,
        )

        # Interior residues in the same model+chain
        if 0 < i < n - 1:
            p_mid, p_cid = records[i - 1][0], records[i - 1][1]
            nx_mid, nx_cid = records[i + 1][0], records[i + 1][1]
            if mid == p_mid == nx_mid and cid == p_cid == nx_cid:
                prev_c  = records[i - 1][6]   # C of residue i-1
                prev_ca = records[i - 1][5]   # CA of residue i-1 (for omega)
                next_n  = records[i + 1][4]   # N of residue i+1

                if all(x is not None
                       for x in (prev_c, n_xyz, ca_xyz, c_xyz, next_n)):
                    phi, psi, dC_CA, dN_CA, dN_C, dP_plane, ang_N_CA_C = (
                        calc_dihedrals(prev_c, n_xyz, ca_xyz, c_xyz, next_n)
                    )
                    omega = np.nan
                    if prev_ca is not None:
                        omega = calc_dihedral(
                            prev_ca, prev_c, n_xyz, ca_xyz, cutoff=CUTOFF
                        )
                    tf = max(records[i - 1][7], bf, records[i + 1][7])
                    entry.update(dict(
                        phi=phi, psi=psi, omega=omega,
                        dC_CA=dC_CA, dN_CA=dN_CA, dN_C=dN_C,
                        dP_plane=dP_plane, ang_N_CA_C=ang_N_CA_C,
                        tempFactor=tf,
                    ))

        results.append(entry)

    return results


def rotamer(structure) -> list[dict]:
    """
    Side-chain chi angles for every standard residue.
    Port of Rotamer.m.

    Parameters
    ----------
    structure : Bio.PDB.Structure

    Returns
    -------
    List of dicts with keys:
        chain, resName, resSeq, tempFactor, chi1, chi2, chi3, chi4
    All angles in radians; NaN when atoms are absent or too far apart.
    """
    results: list[dict] = []

    for mid, cid, residue in _iter_standard_residues(structure):
        resname  = residue.resname.strip()
        atom_xyz = {a.get_name().strip(): a.coord.copy() for a in residue}
        max_bf   = max((a.bfactor or 0.0 for a in residue), default=0.0)

        chis: list[float] = [np.nan, np.nan, np.nan, np.nan]
        if resname in CHIS:
            for k, quad in enumerate(CHIS[resname]):
                if all(a in atom_xyz for a in quad):
                    chis[k] = calc_dihedral(
                        atom_xyz[quad[0]], atom_xyz[quad[1]],
                        atom_xyz[quad[2]], atom_xyz[quad[3]],
                        cutoff=CUTOFF,
                    )

        results.append(dict(
            chain=cid, resName=resname, resSeq=residue.id[1],
            tempFactor=max_bf,
            chi1=chis[0], chi2=chis[1], chi3=chis[2], chi4=chis[3],
        ))

    return results


# ---------------------------------------------------------------------------
# Convenience: merge torsion + rotamer into a single DataFrame row per residue
# ---------------------------------------------------------------------------

def extract(structure) -> "pd.DataFrame":
    """
    Full per-residue feature extraction: backbone + side-chain.

    Returns a DataFrame with columns:
        chain, resName, resSeq, model,
        phi, psi, omega, dC_CA, dN_CA, dN_C, dP_plane, ang_N_CA_C,
        tempFactor, chi1, chi2, chi3, chi4
    """
    import pandas as pd

    tor = torsion(structure)
    rot = rotamer(structure)

    if len(tor) != len(rot):
        raise ValueError(
            f"torsion ({len(tor)}) and rotamer ({len(rot)}) lengths differ — "
            "verify the structure has a single consistent model/chain ordering."
        )

    rows = []
    for t, r in zip(tor, rot):
        if t["resName"] != r["resName"] or t["resSeq"] != r["resSeq"]:
            warnings.warn(
                f"Residue mismatch: torsion={t['resName']}{t['resSeq']} "
                f"rotamer={r['resName']}{r['resSeq']}",
                stacklevel=2,
            )
        row = {**t, "chi1": r["chi1"], "chi2": r["chi2"],
               "chi3": r["chi3"], "chi4": r["chi4"]}
        rows.append(row)

    df = pd.DataFrame(rows)

    # Drop boundary/missing rows (phi or psi is NaN)
    return df.dropna(subset=["phi", "psi"]).reset_index(drop=True)
