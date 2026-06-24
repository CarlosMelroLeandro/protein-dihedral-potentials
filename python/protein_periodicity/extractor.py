"""
extractor.py

Batch pipeline — Phase 1:
  Read a list of PDB IDs  →  download structures  →  append per-residue
  measurements to aminoData/<AMINO>.csv files.

Port of ToolsSrc/Protein.m.

Output files
------------
aminoData/<AMINO>.csv    — 12 columns per row:
    phi, psi, dC_CA, dN_CA, dN_C, dP_plane, ang_N_CA_C, tempFactor,
    chi1, chi2, chi3, chi4

aminoData/<AA1><AA2>.csv — 16 columns per row (consecutive residue pair):
    phi1, psi1, dC_CA1, dN_CA1, dN_C1, dP_plane1, ang_N_CA_C1, tempFactor_max,
    phi2, psi2, omega2, dC_CA2, dN_CA2, dN_C2, dP_plane2, ang_N_CA_C2

rotamer_w3.csv           — 12 columns per row (consecutive residue triplet):
    resName1, resName2, resName3,
    phi1, psi1,
    phi2, psi2, omega2,
    phi3, psi3, omega3,
    tempFactor_max

NaN chi angles (GLY, ALA, …) are written as 'nan'.
Pairs and triplets are only written when the window stays within the same
model + chain (stricter than the MATLAB original, which ignores chain breaks).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from Bio.PDB import PDBList, PDBParser

from .pdb_tools import torsion, rotamer, AMINO_ACIDS

logger = logging.getLogger(__name__)

# ── CSV column headers (not written to file, for documentation) ────────────
SINGLE_COLS = [
    "phi", "psi", "dC_CA", "dN_CA", "dN_C",
    "dP_plane", "ang_N_CA_C", "tempFactor",
    "chi1", "chi2", "chi3", "chi4",
]
PAIR_COLS = [
    "phi1", "psi1", "dC_CA1", "dN_CA1", "dN_C1",
    "dP_plane1", "ang_N_CA_C1", "tempFactor_max",
    "phi2", "psi2", "omega2", "dC_CA2", "dN_CA2", "dN_C2",
    "dP_plane2", "ang_N_CA_C2",
]
TRIPLET_COLS = [
    "resName1", "resName2", "resName3",
    "phi1", "psi1",
    "phi2", "psi2", "omega2",
    "phi3", "psi3", "omega3",
    "tempFactor_max",
]

_NUCLEIC_KEYWORDS = frozenset(["rna", "dna", "dna/rna hybrid", "dna/rna"])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(x) -> str:
    """Format one CSV field: float (possibly NaN) or string."""
    if isinstance(x, str):
        return x
    try:
        if math.isnan(x):
            return "nan"
        return f"{float(x):.6f}"
    except (TypeError, ValueError):
        return str(x)


def _append_row(path: Path, row: list) -> None:
    """Append one CSV row to *path* (creating the file if absent)."""
    with path.open("a") as fh:
        fh.write(",".join(_fmt(v) for v in row) + "\n")


def is_nucleic(structure) -> bool:
    """Return True if the structure is classified as DNA or RNA."""
    head = str(structure.header.get("head", "")).lower()
    return any(kw in head for kw in _NUCLEIC_KEYWORDS)


def read_pdb_ids(tsv_path: str | Path) -> list[str]:
    """Read PDB IDs from a TSV file (one 4-char ID per line)."""
    path = Path(tsv_path)
    ids = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.append(line[:4].upper())
    return ids


# ──────────────────────────────────────────────────────────────────────────────
# Per-structure processing
# ──────────────────────────────────────────────────────────────────────────────

def process_structure(
    structure,
    amino_dir: Path,
    rotamer_w3_path: Path | None = None,
    *,
    pairs: bool = True,
    triplets: bool = True,
) -> dict[str, int]:
    """
    Extract angles from one structure and append rows to CSV files.

    Parameters
    ----------
    structure        Bio.PDB.Structure
    amino_dir        directory for per-amino CSV files
    rotamer_w3_path  path to the triplet CSV (None → skip triplets)
    pairs            write pair CSVs
    triplets         write triplet CSV

    Returns
    -------
    dict mapping amino-acid name → number of rows written
    """
    amino_dir.mkdir(parents=True, exist_ok=True)

    tor_records = torsion(structure)
    rot_records = rotamer(structure)

    if len(tor_records) != len(rot_records):
        raise ValueError(
            f"torsion ({len(tor_records)}) ≠ rotamer ({len(rot_records)}) — "
            "structure iteration mismatch"
        )

    # Merge into unified record list (same positional order)
    records = []
    for t, r in zip(tor_records, rot_records):
        records.append({
            **t,
            "chi1": r["chi1"],
            "chi2": r["chi2"],
            "chi3": r["chi3"],
            "chi4": r["chi4"],
        })

    n = len(records)
    counts: dict[str, int] = defaultdict(int)

    for i, rec in enumerate(records):
        phi  = rec["phi"]
        psi  = rec["psi"]

        # Only write if this residue has valid backbone angles
        if math.isnan(phi) or math.isnan(psi):
            continue

        amino = rec["resName"]
        if amino not in AMINO_ACIDS:
            continue

        # ── single-amino CSV ─────────────────────────────────────────────────
        row_single = [
            phi,           psi,           rec["dC_CA"],
            rec["dN_CA"],  rec["dN_C"],   rec["dP_plane"],
            rec["ang_N_CA_C"], rec["tempFactor"],
            rec["chi1"],   rec["chi2"],   rec["chi3"],   rec["chi4"],
        ]
        _append_row(amino_dir / f"{amino}.csv", row_single)
        counts[amino] += 1

        # ── pair CSV (i, i+1) — same model + chain ───────────────────────────
        if pairs and i + 1 < n:
            r2 = records[i + 1]
            if r2["model"] == rec["model"] and r2["chain"] == rec["chain"]:
                amino2 = r2["resName"]
                if amino2 in AMINO_ACIDS:
                    tf_max = max(rec["tempFactor"], r2["tempFactor"])
                    row_pair = [
                        phi,           psi,           rec["dC_CA"],
                        rec["dN_CA"],  rec["dN_C"],   rec["dP_plane"],
                        rec["ang_N_CA_C"], tf_max,
                        r2["phi"],     r2["psi"],     r2["omega"],
                        r2["dC_CA"],   r2["dN_CA"],   r2["dN_C"],
                        r2["dP_plane"], r2["ang_N_CA_C"],
                    ]
                    _append_row(amino_dir / f"{amino}{amino2}.csv", row_pair)

        # ── triplet CSV (i, i+1, i+2) — same model + chain ──────────────────
        if triplets and rotamer_w3_path is not None and i + 2 < n:
            r2, r3 = records[i + 1], records[i + 2]
            same = (
                r2["model"] == rec["model"] == r3["model"]
                and r2["chain"] == rec["chain"] == r3["chain"]
            )
            if same:
                amino2 = r2["resName"]
                amino3 = r3["resName"]
                if amino2 in AMINO_ACIDS and amino3 in AMINO_ACIDS:
                    tf_max = max(
                        rec["tempFactor"], r2["tempFactor"], r3["tempFactor"]
                    )
                    row_triplet = [
                        amino, amino2, amino3,
                        phi,           psi,
                        r2["phi"],     r2["psi"],     r2["omega"],
                        r3["phi"],     r3["psi"],     r3["omega"],
                        tf_max,
                    ]
                    _append_row(rotamer_w3_path, row_triplet)

    return dict(counts)


# ──────────────────────────────────────────────────────────────────────────────
# Batch extractor
# ──────────────────────────────────────────────────────────────────────────────

def extract_pdb_list(
    pdb_ids: list[str],
    *,
    pdb_dir: str | Path = "proteinData",
    amino_dir: str | Path = "aminoData",
    rotamer_w3: str | Path | None = "aminoData/rotamer_w3.csv",
    skip_existing: bool = True,
    pairs: bool = True,
    triplets: bool = True,
    max_count: int | None = None,
) -> dict[str, int]:
    """
    Process a list of PDB IDs and accumulate per-amino CSV files.

    Parameters
    ----------
    pdb_ids       list of 4-character PDB IDs (uppercase)
    pdb_dir       directory for cached PDB files (downloaded if absent)
    amino_dir     output directory for per-amino CSV files
    rotamer_w3    path for triplet CSV (None to disable)
    skip_existing skip PDB IDs that already have a cached .ent file
    pairs         write pair CSVs
    triplets      write triplet CSV
    max_count     stop after this many successfully processed structures

    Returns
    -------
    Total per-amino counts across all processed structures.
    """
    pdb_dir  = Path(pdb_dir)
    amino_dir = Path(amino_dir)
    w3_path  = Path(rotamer_w3) if rotamer_w3 else None
    pdb_dir.mkdir(parents=True, exist_ok=True)
    amino_dir.mkdir(parents=True, exist_ok=True)

    pdbl   = PDBList(verbose=False)
    parser = PDBParser(QUIET=True)
    totals: dict[str, int] = defaultdict(int)
    n_done = 0

    for pdb_id in pdb_ids:
        if max_count is not None and n_done >= max_count:
            break

        pdb_id = pdb_id.upper()
        ent_path = pdb_dir / f"pdb{pdb_id.lower()}.ent"

        # ── download if missing ───────────────────────────────────────────
        if not ent_path.exists():
            try:
                fetched = pdbl.retrieve_pdb_file(
                    pdb_id, pdir=str(pdb_dir), file_format="pdb"
                )
                ent_path = Path(fetched)
            except Exception as exc:
                logger.warning("%s: download failed — %s", pdb_id, exc)
                continue
        elif skip_existing:
            logger.debug("%s: cached at %s", pdb_id, ent_path)

        # ── parse ─────────────────────────────────────────────────────────
        try:
            structure = parser.get_structure(pdb_id, str(ent_path))
        except Exception as exc:
            logger.warning("%s: parse failed — %s", pdb_id, exc)
            continue

        # ── skip nucleic acids ────────────────────────────────────────────
        if is_nucleic(structure):
            logger.debug("%s: nucleic acid — skipped", pdb_id)
            continue

        # ── extract ───────────────────────────────────────────────────────
        try:
            counts = process_structure(
                structure,
                amino_dir,
                rotamer_w3_path=w3_path,
                pairs=pairs,
                triplets=triplets,
            )
        except Exception as exc:
            logger.warning("%s: extraction failed — %s", pdb_id, exc)
            continue

        total = sum(counts.values())
        logger.info("%s: %d residues written %s", pdb_id, total, counts)
        for aa, c in counts.items():
            totals[aa] += c
        n_done += 1

    return dict(totals)


def extract_from_file(
    tsv_path: str | Path,
    *,
    pdb_dir: str | Path = "proteinData",
    amino_dir: str | Path = "aminoData",
    rotamer_w3: str | Path | None = "aminoData/rotamer_w3.csv",
    skip_existing: bool = True,
    pairs: bool = True,
    triplets: bool = True,
    max_count: int | None = None,
) -> dict[str, int]:
    """
    Read PDB IDs from *tsv_path* (one per line) and run the batch extractor.

    Thin wrapper around :func:`extract_pdb_list`.
    """
    ids = read_pdb_ids(tsv_path)
    logger.info("Loaded %d PDB IDs from %s", len(ids), tsv_path)
    return extract_pdb_list(
        ids,
        pdb_dir=pdb_dir,
        amino_dir=amino_dir,
        rotamer_w3=rotamer_w3,
        skip_existing=skip_existing,
        pairs=pairs,
        triplets=triplets,
        max_count=max_count,
    )
