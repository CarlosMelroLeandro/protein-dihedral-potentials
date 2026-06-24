# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Statistical potential pipeline for protein dihedral angle distributions. Extracts backbone (φ, ψ) and side-chain (χ₁–χ₄) dihedral angles from PDB crystal structures, fits them as 2D von Mises KDE potentials, and exports potential maps and gradients for use in rigid-body protein dynamics simulations.

The Python package (`python/protein_periodicity/`) is a port of the original MATLAB pipeline (`matlab/`).

## Setup and running

```bash
cd python/
pip install -r requirements.txt

# Full pipeline (downloads ~600 PDB structures on first run, takes hours)
python run_pipeline.py

# Quick test run with a small subset
python run_pipeline.py --max-pdbs 10

# Change grid resolution (default 180 → 2° per cell, use 360 for 1° per cell)
python run_pipeline.py --n-ang 360
```

Outputs:
- `python/aminoData/<AA>.csv` — per-residue measurements (Phase 1)
- `python/aminoData/<AA1><AA2>.csv` — consecutive residue pair data
- `python/aminoData/rotamer_w3.csv` — triplet window rotamer data
- `python/contData/Mises<AA>180.npz` — statistical potential maps (Phase 2)
- `python/proteinData/` — downloaded raw PDB files (cached, skip_existing=True)

## Notebooks

```bash
cd python/
jupyter notebook
```

- `01_pdb_tools.ipynb` — walkthrough and validation of `pdb_tools.py`
- `02_amino_analysis.ipynb` — per-amino-acid analysis; change `AMINO = "TYR"` cell to target any of the 20 standard amino acids

## Architecture

### Two-phase pipeline

**Phase 1 — Extraction** (`extractor.py` / `Protein.m`):
- Reads PDB IDs from `data/pdb1A.tsv` (607 curated structures)
- Downloads PDB files via BioPython's `PDBList`
- Calls `pdb_tools.py` to compute per-residue geometry
- Appends rows to `aminoData/<AA>.csv`, `aminoData/<AA1><AA2>.csv`, and `rotamer_w3.csv`
- Chain-break detection is stricter than the original MATLAB (pair/triplet windows must stay within the same model+chain)

**Phase 2 — Potential maps** (`analysis.py` / `ForcesCont.m`):
- Loads each `aminoData/<AA>.csv`
- Fits 2D von Mises KDE via `circ_stat.circ_vmpar` + `circ_vmpdf2`
- Computes −log(p) potential `E`, gradients `Gx`/`Gy`, concentration params `kappa`, and sample count `n`
- Saves to `contData/Mises<AA><n_ang>.npz`

### Module responsibilities

| Module | Ports from | Role |
|---|---|---|
| `pdb_tools.py` | `pdbTools/`, `calcDihedral.m`, `Rotamer.m` | Dihedral and bond geometry from BioPython structure objects |
| `circ_stat.py` | `CircStat/` (Berens 2009), `circ_vmpdf2.m` | Circular statistics: mean, variance, uniformity tests, 2D von Mises KDE |
| `extractor.py` | `Protein.m` | Batch PDB download → CSV |
| `analysis.py` | `ForcesCont.m`, `Analysis.m` | CSV → `.npz` potential maps |

### `.npz` file structure

Each `contData/Mises<AA><n_ang>.npz` stores arrays keyed by pair index `k`:
- `pair_k` — dihedral names, e.g. `['phi', 'psi']`
- `E_k` — potential −log(p), shape `(n_ang, n_ang)`
- `Gx_k`, `Gy_k` — gradient components
- `kappa_k` — fitted von Mises concentrations `[κ₁, κ₂]`
- `n_k` — sample count

Loading:
```python
from protein_periodicity.analysis import load_potential
data = load_potential("python/contData", "ALA", n_ang=180)
E = data["potentials"][0]["E"]   # shape (180, 180)
```

### Key constants

- `CUTOFF = 6.5` Å — bond-length sanity check in `pdb_tools.py` (matches MATLAB default)
- `_MIN_SAMPLES = 5` — minimum points to compute a potential in `analysis.py`
- Angular resolution default: `n_ang=180` (2° per cell); use 360 for 1° per cell

### Notable deviation from MATLAB

`circ_vmpdf2` in `circ_stat.py` is fully vectorised via NumPy broadcasting and fixes a bug in the original MATLAB where `kappa1` was used for both axes instead of `kappa1` for φ and `kappa2` for ψ.

## Data files

- `data/pdb1A.tsv` — 607 curated PDB IDs (primary reference set, no header, one ID per line)
- `data/pdb1AB.tsv` — extended PDB ID list
