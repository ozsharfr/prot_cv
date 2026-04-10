"""
assemble_features.py — Build the extended feature matrix from scratch.

Reads:
  results/<PDB_ID>.csv          (per-structure results from pipeline.py)
  results/scan_features.parquet (built by scan_features.py)
  results/energy_terms.parquet  (built by energy_terms.py)

Writes:
  results/feature_matrix_extended_<target>.parquet

Self-contained — does NOT depend on a previously saved base parquet.
Can be run immediately after pipeline.py + run_extended_pipeline.py.
"""

import sys
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import os
os.chdir(_PROJECT_ROOT)        # must happen before local imports so relative paths resolve correctly
sys.path.insert(0, str(_PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from config import RESULTS_DIR
from features import build_features
from common import load_results, build_pdb_paths

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

AA_20 = list("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_or_warn(path: Path, name: str):
    if not path.exists():
        log.warning("%s not found at %s — skipping these features", name, path)
        return None
    df = pd.read_parquet(path)
    log.info("Loaded %s: %d rows, %d columns", name, len(df), len(df.columns))
    return df


def summarise_scan(df: pd.DataFrame) -> pd.DataFrame:
    """Add summary statistics over the 20-dim scan DDG vectors."""
    result = df.copy()
    for tag, cols in [
        ("self", [f"scan_{aa}" for aa in AA_20]),
        ("nb",   [f"scan_nb_{aa}" for aa in AA_20]),
    ]:
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        sub = df[present]
        result[f"scan_{tag}_mean"]     = sub.mean(axis=1)
        result[f"scan_{tag}_std"]      = sub.std(axis=1)
        result[f"scan_{tag}_min"]      = sub.min(axis=1)
        result[f"scan_{tag}_max"]      = sub.max(axis=1)
        result[f"scan_{tag}_mean_abs"] = sub.abs().mean(axis=1)
        result[f"scan_{tag}_n_destab"] = (sub > 0.5).sum(axis=1)
        result[f"scan_{tag}_n_stab"]   = (sub < -0.5).sum(axis=1)
    return result


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def assemble(target: str = "prediction_error") -> pd.DataFrame:
    """
    Build extended feature matrix from scratch.
    1. Load per-structure CSVs and compute base features
    2. Join scan features
    3. Join energy terms
    Returns full DataFrame including target, pdb_id, chain, resnum, wt_aa, mut_aa.
    """
    # Step 1 — base features from raw CSVs
    df_raw    = load_results(RESULTS_DIR)
    pdb_paths = build_pdb_paths(df_raw)
    df = build_features(df_raw, pdb_paths=pdb_paths if pdb_paths else None)

    if target not in df.columns:
        log.error("Target '%s' not in data. Available: %s",
                  target, [c for c in ["prediction_error", "DDG"] if c in df.columns])
        return pd.DataFrame()

    df = df.dropna(subset=[target, "pdb_id"]).copy()
    df["pdb_id"] = df["pdb_id"].astype(str)
    # Normalise resnum to int to ensure consistent merge keys
    if "resnum" in df.columns:
        df["resnum"] = pd.to_numeric(df["resnum"], errors="coerce").astype("int64")
    log.info("Base features: %d rows x %d columns", len(df), len(df.columns))

    # Step 2 — scan features (one row per position, join on pdb_id+chain+resnum)
    scan = load_or_warn(RESULTS_DIR / "scan_features.parquet", "scan features")
    if scan is not None:
        scan = summarise_scan(scan)
        if "resnum" in scan.columns:
            scan["resnum"] = pd.to_numeric(scan["resnum"], errors="coerce").astype("int64")
        keys = [c for c in ["pdb_id", "chain", "resnum"]
                if c in df.columns and c in scan.columns]
        if keys:
            log.info("Merging scan features on: %s", keys)
            # Force plain Python types to avoid any hidden type issues
            for k in keys:
                df[k]   = df[k].astype(str) if k != "resnum" else df[k].astype(int)
                scan[k] = scan[k].astype(str) if k != "resnum" else scan[k].astype(int)
            df = df.reset_index(drop=True)
            scan = scan.reset_index(drop=True)
            df = pd.merge(df, scan, on=keys, how="left")
            n_matched = df["scan_self_mean"].notna().sum() if "scan_self_mean" in df.columns else 0
            log.info("Added %d scan columns | %d/%d rows matched",
                     len([c for c in df.columns if "scan_" in c]), n_matched, len(df))
        else:
            log.warning("No common merge keys for scan — skipping")

    # Step 3 — energy terms (one row per mutation, join on pdb_id+chain+resnum+wt_aa+mut_aa)
    et = load_or_warn(RESULTS_DIR / "energy_terms.parquet", "energy terms")
    if et is not None:
        if "resnum" in et.columns:
            et["resnum"] = pd.to_numeric(et["resnum"], errors="coerce").astype("int64")
        # Deduplicate energy terms to avoid row multiplication on merge
        et_keys = [c for c in ["pdb_id", "chain", "resnum", "wt_aa", "mut_aa"]
                   if c in et.columns]
        et = et.drop_duplicates(subset=et_keys)
        keys = [c for c in et_keys if c in df.columns]
        if keys:
            n_before = len(df)
            log.info("Merging energy terms on: %s", keys)
            df = df.merge(et, on=keys, how="left", suffixes=("", "_et"))
            if len(df) != n_before:
                log.warning("Row count changed after energy merge: %d → %d (check for duplicate keys)",
                            n_before, len(df))
            n_matched = df["et_total_energy"].notna().sum() if "et_total_energy" in df.columns else 0
            log.info("Added %d energy term columns | %d/%d rows matched",
                     len([c for c in df.columns if c.startswith("et_")]), n_matched, len(df))
        else:
            log.warning("No common merge keys for energy terms — skipping")

    # Drop columns that pyarrow cannot serialise (mixed object types from raw SKEMPI)
    problem_cols = [c for c in df.columns
                    if df[c].dtype == object and c not in
                    ["pdb_id", "chain", "wt_aa", "mut_aa", "resnum_str",
                     "iMutation_Location(s)", "Mutation(s)_PDB"]]
    if problem_cols:
        log.info("Dropping %d non-serialisable object columns: %s", len(problem_cols), problem_cols)
        df = df.drop(columns=problem_cols)

    log.info("Extended feature matrix: %d rows x %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Assemble extended feature matrix")
    parser.add_argument("--target", default="prediction_error",
                        choices=["prediction_error", "DDG"])
    args = parser.parse_args()

    out_path = RESULTS_DIR / f"feature_matrix_extended_{args.target}.parquet"
    if out_path.exists():
        log.info("%s already exists — delete to recompute", out_path.name)
        return

    df = assemble(args.target)
    if df.empty:
        log.error("Assembly failed — aborting")
        return

    df.to_parquet(out_path, index=False)
    log.info("Saved: %s", out_path)

    cols      = df.columns.tolist()
    scan_cols = [c for c in cols if "scan_" in c]
    et_cols   = [c for c in cols if c.startswith("et_")]
    base_cols = [c for c in cols if c not in scan_cols and c not in et_cols]
    log.info("Feature groups: base=%d | scan=%d | energy_terms=%d | total=%d",
             len(base_cols), len(scan_cols), len(et_cols), len(cols))


if __name__ == "__main__":
    main()