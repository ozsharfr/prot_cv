import logging

import numpy as np
import pandas as pd

from .splits import make_pos_keys

logger = logging.getLogger(__name__)

# Standard amino acid single-letter codes
ALL_AA = list("ACDEFGHIKLMNPQRSTVWY")


def build_sibling_features(
    dff,
    groups,
    resnums,
    mut_to,
    ddg_col="measured_ddg",
    mode="trees",      # 'trees' (NaN for missing) or 'lr' (0 + indicator)
    aa_list=None,      # subset of AAs to include; None = all 20
):
    """
    For each mutation row, add sibling DDG features from other mutations
    at the same (structure, resnum) position.

    For mode='trees' (RF, GB):
        - One column per AA: sibling_ddg_W, sibling_ddg_Y, ...
        - NaN where sibling not observed at this position

    For mode='lr' (Logistic Regression):
        - Same DDG columns but NaN filled with 0
        - Additional binary indicator: sibling_obs_W, sibling_obs_Y, ...
          (1 = sibling was measured, 0 = not available)

    Parameters
    ----------
    dff      : pd.DataFrame  source dataframe (must contain ddg_col and mut_to)
    groups   : pd.Series     structure IDs
    resnums  : pd.Series     residue numbers
    mut_to   : pd.Series     substituted amino acid (e.g. 'W', 'Y', 'R')
    ddg_col  : str           column name for measured DDG
    mode     : str           'trees' or 'lr'
    aa_list  : list or None  amino acids to include (default: all 20)

    Returns
    -------
    pd.DataFrame with sibling features, indexed like dff
    """
    if aa_list is None:
        aa_list = ALL_AA

    pos_keys = make_pos_keys(groups, resnums)

    df = pd.DataFrame({
        "pos_key": pos_keys,
        "mut_to":  mut_to,
        "ddg":     dff[ddg_col],
    }, index=dff.index)

    # Build lookup: pos_key → {aa: ddg}
    pos_aa_ddg = (
        df.groupby(["pos_key", "mut_to"])["ddg"]
        .mean()   # if duplicates, take mean
        .unstack("mut_to")
        .reindex(columns=aa_list)
    )

    # For each row, look up sibling DDGs at same position
    # excluding the current mutation itself (LOO — no self-leakage)
    result = pd.DataFrame(index=dff.index)

    for aa in aa_list:
        col = f"sibling_ddg_{aa}"
        # Map position key to that AA's DDG at the position
        raw = pos_keys.map(pos_aa_ddg[aa].to_dict()) if aa in pos_aa_ddg.columns else np.nan

        # Zero out self — if this row IS the →aa mutation, its own DDG is not a sibling
        is_self = (mut_to == aa)
        if hasattr(raw, 'where'):
            raw = raw.where(~is_self, other=np.nan)
        else:
            raw = pd.Series(raw, index=dff.index).where(~is_self, other=np.nan)

        result[col] = raw

    if mode == "lr":
        # Fill NaN with 0 and add observation indicator
        for aa in aa_list:
            ddg_col_name = f"sibling_ddg_{aa}"
            obs_col_name = f"sibling_obs_{aa}"
            result[obs_col_name] = result[ddg_col_name].notna().astype(int)
            result[ddg_col_name] = result[ddg_col_name].fillna(0)

    elif mode == "trees":
        pass  # keep NaN as-is

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'trees' or 'lr'.")

    # Log coverage stats
    n_rows        = len(result)
    ddg_cols      = [c for c in result.columns if c.startswith("sibling_ddg_")]
    n_observed    = result[ddg_cols].notna().sum().sum()
    n_total       = n_rows * len(ddg_cols)
    coverage      = n_observed / n_total if n_total > 0 else 0
    positions_with_any = (result[ddg_cols].notna().any(axis=1)).sum()

    logger.info(
        f"Sibling features (mode='{mode}'): "
        f"{len(ddg_cols)} AA columns | "
        f"coverage={coverage:.1%} | "
        f"{positions_with_any}/{n_rows} mutations have ≥1 sibling"
    )

    return result


def add_sibling_features(X, dff, groups, resnums, mut_to,
                         ddg_col="measured_ddg", mode="trees", aa_list=None):
    """
    Convenience wrapper: build sibling features and concatenate with X.

    Returns
    -------
    pd.DataFrame: X with sibling columns appended
    """
    sib = build_sibling_features(
        dff, groups, resnums, mut_to,
        ddg_col=ddg_col, mode=mode, aa_list=aa_list
    )
    return pd.concat([X, sib], axis=1)