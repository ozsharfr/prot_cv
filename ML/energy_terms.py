"""
energy_terms.py — Parse all 15 FoldX energy term columns from existing
                  per-structure result directories.

FoldX Average_.fxout columns (tab-separated):
  0  pdb_name
  1  SD
  2  total_energy      ← currently the only column parsed by foldx.py
  3  backbone_hbond
  4  sidechain_hbond
  5  vdw
  6  electrostatics
  7  solvation_polar
  8  solvation_hydrophobic
  9  vdw_clashes
 10  entropy_sidechain
 11  entropy_mainchain
 12  sloop_entropy
 13  mloop_entropy
 14  cis_bond
 15  torsional_clash
 16  backbone_clash
 17  helix_dipole
 18  water_bridge
 19  disulfide
 20  electrostatic_kon
 21  partial_covalent

Only columns 2–17 are used (the 15 main terms that are consistently present).

Output
------
results/energy_terms.parquet
  Columns: pdb_id, chain, resnum, wt_aa, mut_aa + 15 energy term columns
"""

import sys
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import os
os.chdir(_PROJECT_ROOT)        # must happen before local imports so relative paths resolve correctly
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np
from config import FOLDX_WORK_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

ENERGY_COLS = [
    "total_energy",
    "backbone_hbond",
    "sidechain_hbond",
    "vdw",
    "electrostatics",
    "solvation_polar",
    "solvation_hydrophobic",
    "vdw_clashes",
    "entropy_sidechain",
    "entropy_mainchain",
    "sloop_entropy",
    "mloop_entropy",
    "cis_bond",
    "torsional_clash",
    "backbone_clash",
    "helix_dipole",
]

# Prefixed versions as they'll appear in the feature matrix
ENERGY_FEATURE_COLS = [f"et_{c}" for c in ENERGY_COLS]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_energy_terms(fxout_file: Path) -> dict | None:
    """
    Parse all energy term columns from a FoldX Average_.fxout file.
    Returns dict of {col_name: mean_value} or None on failure.
    """
    if not fxout_file.exists():
        return None

    rows = []
    with open(fxout_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 17:
                continue
            try:
                # cols 2–17 = 16 values (total + 15 terms)
                vals = [float(parts[i]) for i in range(2, 18)]
                rows.append(vals)
            except ValueError:
                continue   # header or footer line

    if not rows:
        return None

    means = np.mean(rows, axis=0)
    return dict(zip(ENERGY_COLS, means))


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_energy_terms(results_dir: Path = RESULTS_DIR,
                       work_dir: Path   = FOLDX_WORK_DIR) -> pd.DataFrame:
    """
    Walk through FOLDX_WORK_DIR subdirectories (one per mutation), parse
    energy terms, and join back to the per-structure CSV data.

    Returns DataFrame with columns:
        pdb_id, chain, resnum, wt_aa, mut_aa, et_total_energy, et_vdw, ...
    """
    import re
    # Load all per-structure results to get mutation metadata
    csvs = [f for f in results_dir.glob("*.csv")
            if re.match(r"^[A-Z0-9]{4}\.csv$", f.name)]
    if not csvs:
        log.error("No per-structure CSVs found in %s", results_dir)
        return pd.DataFrame()

    mutations_df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    log.info("Loaded %d mutations from %d structures", len(mutations_df), len(csvs))

    records = []
    missing = 0

    for _, row in mutations_df.iterrows():
        pdb_id  = row["pdb_id"]
        chain   = row["chain"]
        wt_aa   = row["wt_aa"]
        mut_aa  = row["mut_aa"]
        resnum  = int(row["resnum"])

        # FoldX mutation string e.g. RA45K
        mut_str = f"{wt_aa}{chain}{resnum}{mut_aa}"

        # Look for the archived fxout file
        mut_dir  = work_dir / mut_str
        # Repaired PDB stem varies per structure — search for any Average_ file
        fxout_candidates = list(mut_dir.glob("Average_*.fxout"))

        if not fxout_candidates:
            missing += 1
            log.debug("No fxout found for %s", mut_str)
            records.append({
                "pdb_id": pdb_id, "chain": chain,
                "resnum": resnum, "wt_aa": wt_aa, "mut_aa": mut_aa,
                **{c: np.nan for c in ENERGY_COLS}
            })
            continue

        terms = parse_energy_terms(fxout_candidates[0])
        if terms is None:
            missing += 1
            terms = {c: np.nan for c in ENERGY_COLS}

        records.append({
            "pdb_id": pdb_id, "chain": chain,
            "resnum": resnum, "wt_aa": wt_aa, "mut_aa": mut_aa,
            **terms
        })

    log.info("Parsed energy terms: %d ok, %d missing", len(records) - missing, missing)

    df = pd.DataFrame(records)
    # Rename to feature-safe names with et_ prefix
    rename = {c: f"et_{c}" for c in ENERGY_COLS}
    df = df.rename(columns=rename)
    # Ensure correct types for parquet serialisation
    df["pdb_id"]  = df["pdb_id"].astype(str)
    df["chain"]   = df["chain"].astype(str)
    df["wt_aa"]   = df["wt_aa"].astype(str)
    df["mut_aa"]  = df["mut_aa"].astype(str)
    df["resnum"]  = pd.to_numeric(df["resnum"], errors="coerce").astype("int64")
    for col in [f"et_{c}" for c in ENERGY_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    out_path = RESULTS_DIR / "energy_terms.parquet"
    if out_path.exists():
        log.info("energy_terms.parquet already exists — delete to recompute")
        return

    df = build_energy_terms()
    if df.empty:
        log.error("No energy terms extracted — aborting")
        return

    df.to_parquet(out_path, index=False)
    log.info("Saved: %s (%d rows, %d columns)", out_path, len(df), len(df.columns))


if __name__ == "__main__":
    main()