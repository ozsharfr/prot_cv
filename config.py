"""
config.py — Central configuration for the flexibility vs. DDG pipeline.
Edit these values before running.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FOLDX_BIN      = Path(os.environ.get("FOLDX_BIN", "/usr/local/bin/foldx"))
FOLDX_BIN = Path(r"C:\Users\ozsha\Downloads\foldxWindows\foldx_20261231.exe")
SKEMPI_CSV     = Path("data/skempi_v2.csv")
STRUCTURES_DIR = Path("data/structures")
FOLDX_WORK_DIR = Path("data/foldx_runs")
RESULTS_DIR    = Path("results")
FIGURES_DIR    = Path("results/figures")

PILOT_PDB_IDS =['3SGB' , '3S9D','1JTG' ,'1AO7','2FTL','1A22','1JRH', '1CHO', '1R0R', '1PPF', 
                '3BT1', '3SE3','2NZ9','3HFM','1BRS','4BFI','2WPT','2JEL','1CBW',
                '4RS1','3EQS','3MZG','1DAN','3QDG','3BN9'] + [
                '3SE4','4P5T', '3NPS', '4I77', '1EMV','1IAR', '1MHP','3C60','3NGB',
                '1B41','1DQJ','1TM1','2NYY','1E50','1FC2','1FSS','1OGA','1LFD','3SZK','1BP3',
                '2G2U','1KTZ','1DVF','1VFB', '1GC1','3QHY','4P23','4G0N','1A4Y','1JTD']
    # "1CSE",   # Subtilisin / eglin c — many SKEMPI entries, classic benchmark
    # "1VFB",   # Antibody / lysozyme — good interface, canonical test case
    # "1A22",   # Human growth hormone / receptor — well-studied, good mutation coverage




# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

RESOLUTION_CUTOFF = 4.0   # Angstrom — max crystal resolution to include
INTERFACE_CUTOFF  = 12.0   # Angstrom — residue counts as interface if within this distance of partner chain

# ---------------------------------------------------------------------------
# ANM
# ---------------------------------------------------------------------------

ANM_MODES = 10   # number of slowest modes to include in MSF calculation
