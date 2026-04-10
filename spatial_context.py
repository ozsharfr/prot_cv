"""
spatial_context.py — Find spatially adjacent residues for a given position.

Given a (pdb_id, chain, resnum), returns the nearest residue on the same chain
by Cα–Cα distance, excluding the residue itself.

Used by scan_features.py to define the "neighbouring position" for the
positional scan, as an alternative to simple sequence ±1.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def get_spatial_neighbour(
    pdb_id: str,
    chain_id: str,
    resnum: int,
    struct_cache: dict,
    exclude_self: bool = True,
) -> Optional[int]:
    """
    Return the resnum of the spatially nearest residue on the same chain.

    Parameters
    ----------
    pdb_id       : PDB identifier (used as key into struct_cache)
    chain_id     : Chain identifier
    resnum       : Residue number of the query position
    struct_cache : Dict {pdb_id: BioPython Model} — pre-parsed structures
    exclude_self : If True, exclude the query residue itself (default True)

    Returns
    -------
    Nearest resnum (int) or None if not found.
    """
    model = struct_cache.get(pdb_id)
    if model is None:
        log.debug("Structure not in cache: %s", pdb_id)
        return None

    try:
        chain = model[chain_id]
    except KeyError:
        log.debug("Chain %s not found in %s", chain_id, pdb_id)
        return None

    # Collect all Cα coordinates on this chain
    residues = [
        r for r in chain.get_residues()
        if r.get_id()[0] == " " and "CA" in r   # ATOM records only
    ]
    if len(residues) < 2:
        return None

    # Find query residue Cα
    query_ca = None
    for r in residues:
        if r.get_id()[1] == resnum:
            query_ca = r["CA"].get_vector().get_array()
            break

    if query_ca is None:
        log.debug("Residue %d not found in chain %s of %s", resnum, chain_id, pdb_id)
        return None

    # Find nearest residue by Cα distance
    best_dist   = np.inf
    best_resnum = None

    for r in residues:
        rn = r.get_id()[1]
        if exclude_self and rn == resnum:
            continue
        ca   = r["CA"].get_vector().get_array()
        dist = float(np.linalg.norm(ca - query_ca))
        if dist < best_dist:
            best_dist   = dist
            best_resnum = rn

    log.debug("%s/%s/%d → nearest neighbour: %s (%.2f Å)",
              pdb_id, chain_id, resnum, best_resnum, best_dist)
    return best_resnum


def build_struct_cache(pdb_paths: dict) -> dict:
    """
    Parse all PDB files into BioPython models and return a cache dict.

    Parameters
    ----------
    pdb_paths : {pdb_id: Path}

    Returns
    -------
    {pdb_id: BioPython Model (model[0])}
    """
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    cache  = {}

    for pdb_id, path in pdb_paths.items():
        path = Path(path)
        if not path.exists():
            log.warning("PDB file not found, skipping: %s", path)
            continue
        try:
            structure    = parser.get_structure(pdb_id, str(path))
            cache[pdb_id] = structure[0]
            log.debug("Cached structure: %s", pdb_id)
        except Exception as e:
            log.warning("Failed to parse %s: %s", path, e)

    log.info("Loaded %d structures into cache", len(cache))
    return cache
