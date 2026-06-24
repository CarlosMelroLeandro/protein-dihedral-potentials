"""
run_pipeline.py

Full pipeline: 607 PDB IDs -> aminoData/*.csv -> contData/*.npz

Usage:
    cd python/
    python3 run_pipeline.py [--n-ang 180] [--max-pdbs N]

Directory layout expected (relative to repo root):
    data/pdb1A.tsv          — PDB ID list
    python/aminoData/       — extracted CSV files (created on first run)
    python/contData/        — statistical potential maps (created on first run)
    python/proteinData/     — downloaded PDB files (created on first run)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── paths relative to this script ────────────────────────────────────────────
ROOT      = Path(__file__).parent          # python/
REPO_ROOT = ROOT.parent                    # repo root
TSV_PATH  = REPO_ROOT / "data" / "pdb1A.tsv"
PDB_DIR   = ROOT / "proteinData"
AMINO_DIR = ROOT / "aminoData"
CONT_DIR  = ROOT / "contData"
LOG_FILE  = ROOT / "pipeline.log"

def setup_logging():
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ang",   type=int, default=180)
    parser.add_argument("--max-pdbs", type=int, default=None,
                        help="Limit number of PDB structures (for testing)")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("Protein Periodicity — Full Pipeline")
    log.info("  TSV:      %s", TSV_PATH)
    log.info("  pdb_dir:  %s", PDB_DIR)
    log.info("  amino_dir:%s", AMINO_DIR)
    log.info("  cont_dir: %s", CONT_DIR)
    log.info("  n_ang:    %d", args.n_ang)
    if args.max_pdbs:
        log.info("  max_pdbs: %d", args.max_pdbs)
    log.info("=" * 60)

    sys.path.insert(0, str(ROOT))
    from protein_periodicity.extractor import extract_from_file
    from protein_periodicity.analysis  import analyse_all

    # ── Phase 1: extraction ──────────────────────────────────────────────────
    log.info("")
    log.info("PHASE 1 — PDB extraction")
    t0 = time.time()

    totals = extract_from_file(
        TSV_PATH,
        pdb_dir=PDB_DIR,
        amino_dir=AMINO_DIR,
        rotamer_w3=AMINO_DIR / "rotamer_w3.csv",
        skip_existing=True,
        pairs=True,
        triplets=True,
        max_count=args.max_pdbs,
    )

    dt1 = time.time() - t0
    grand_total = sum(totals.values())
    log.info("")
    log.info("Phase 1 done in %.0f s", dt1)
    log.info("Residues per amino acid:")
    for aa in sorted(totals):
        log.info("  %s: %d", aa, totals[aa])
    log.info("Grand total: %d residues", grand_total)

    # ── Phase 2: potential maps ──────────────────────────────────────────────
    log.info("")
    log.info("PHASE 2 — Statistical potential computation (n_ang=%d)", args.n_ang)
    t0 = time.time()

    results = analyse_all(
        AMINO_DIR,
        CONT_DIR,
        n_ang=args.n_ang,
    )

    dt2 = time.time() - t0
    log.info("")
    log.info("Phase 2 done in %.0f s", dt2)
    log.info("Potentials computed for %d amino acids:", len(results))
    for aa, res in sorted(results.items()):
        pairs = ["-".join(p["pair"]) for p in res["potentials"]]
        log.info("  %s: n=%d  pairs=%s", aa, res["n_total"], ", ".join(pairs))

    log.info("")
    log.info("Pipeline complete.")
    log.info("  aminoData/ : %d CSV files", len(list(AMINO_DIR.glob("*.csv"))))
    log.info("  contData/  : %d .npz files", len(list(CONT_DIR.glob("*.npz"))))

if __name__ == "__main__":
    main()
