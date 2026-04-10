import logging
import random

import numpy as np
import pandas as pd

from .splits import make_pos_keys, random_split, sample_position_holdouts
from .train import _fit_one, _make_sample_weights

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _logloss_per_mutation(clf, X_te, y_te):
    """Per-mutation logloss as a pd.Series indexed like y_te."""
    eps    = 1e-7
    proba  = np.clip(clf.predict_proba(X_te)[:, 1], eps, 1 - eps)
    losses = -(y_te.values * np.log(proba) + (1 - y_te.values) * np.log(1 - proba))
    return pd.Series(losses, index=y_te.index)


def filter_multi_mutation_positions(groups, resnums, min_mutations=3):
    """
    Return a boolean mask keeping only mutations at positions
    that have at least `min_mutations` measurements.
    """
    pos_keys = make_pos_keys(groups, resnums)
    counts   = pos_keys.map(pos_keys.value_counts())
    mask     = counts >= min_mutations

    n_pos_before = pos_keys.nunique()
    n_pos_after  = pos_keys[mask].nunique()
    logger.info(
        f"Position filter (>={min_mutations} mutations): "
        f"{n_pos_after}/{n_pos_before} positions kept, "
        f"{mask.sum()}/{len(mask)} mutations kept"
    )
    return mask


def _get_outer_split_masks(X, y, groups, resnums, split_mode, fold_seed):
    """
    Generate a single train/test boolean mask pair.

        'random'   — random 80/20, no leakage constraint (M1)
        'position' — position-level holdout, siblings cannot leak (M2)
    """
    if split_mode == "random":
        train_idx, test_idx = random_split(len(X), test_frac=0.2, seed=fold_seed)
        train_mask = pd.Series(False, index=X.index)
        test_mask  = pd.Series(False, index=X.index)
        train_mask.iloc[train_idx] = True
        test_mask.iloc[test_idx]   = True
        return train_mask, test_mask

    if split_mode == "position":
        holdouts, pos_keys_full = sample_position_holdouts(
            groups, resnums, n_holdouts=1, holdout_size=3, seed=fold_seed
        )
        held       = holdouts[0]
        test_mask  = pos_keys_full.isin(held)
        train_mask = ~test_mask
        return train_mask, test_mask

    raise ValueError(
        f"Unknown split_mode: {split_mode!r}. Choose 'random' or 'position'."
    )


# ── Main LOO experiment ───────────────────────────────────────────────────────

def run_loo_retraining(
    X, y, groups, resnums, mut_to,
    models,
    split_mode="random",
    class_weight_map=None,
    n_splits=50,
    min_mutations=3,
    min_pos_train=3,
    min_pos_test=1,
    seed=42,
):
    """
    For each mutation S in the train set, remove it, retrain, and measure
    the logloss delta on all test mutations.

    Records (s_mut_to, t_mut_to) pairs so results can be aggregated into
    a heatmap: rows = mut_to of left-out S, cols = mut_to of predicted T.

    Two split modes:
        'random'   — M1: random split, siblings of S may appear in test
        'position' — M2: position-level split, siblings cannot be in test

    Parameters
    ----------
    X             : pd.DataFrame  features
    y             : pd.Series     binary labels
    groups        : pd.Series     structure IDs
    resnums       : pd.Series     residue numbers
    mut_to        : pd.Series     substituted amino acid (e.g. 'W', 'Y', 'R')
    models        : dict          {"name": sklearn_estimator}
    split_mode    : str           'random' (M1) | 'position' (M2)
    n_splits      : int           number of outer splits
    min_mutations : int           min mutations per position to include (default 3)
    """
    pos_keys = make_pos_keys(groups, resnums)

    # Filter to positions with >= min_mutations
    pos_filter    = filter_multi_mutation_positions(groups, resnums, min_mutations)
    eligible_idx  = set(X.index[pos_filter])
    logger.info(
        f"split_mode='{split_mode}' | "
        f"eligible mutations (>={min_mutations} per position): {len(eligible_idx)}"
    )

    # Work only on filtered data
    X_f        = X[pos_filter]
    y_f        = y[pos_filter]
    groups_f   = groups[pos_filter]
    resnums_f  = resnums[pos_filter]
    mut_to_f   = mut_to[pos_filter]
    pos_keys_f = pos_keys[pos_filter]

    rng           = random.Random(seed)
    records       = []
    skipped_folds = 0

    for split_i in range(n_splits):
        fold_seed = rng.randint(0, 999999)

        train_mask, test_mask = _get_outer_split_masks(
            X_f, y_f, groups_f, resnums_f, split_mode, fold_seed
        )

        # Basic fold validity
        if y_f[train_mask].nunique() < 2 or y_f[test_mask].nunique() < 2:
            skipped_folds += 1
            continue
        if (np.sum(y_f[train_mask] == 1) < min_pos_train
                or np.sum(y_f[test_mask]  == 1) < min_pos_test):
            skipped_folds += 1
            continue

        X_tr_full = X_f[train_mask]
        y_tr_full = y_f[train_mask]
        X_te      = X_f[test_mask]
        y_te      = y_f[test_mask]

        logger.info(
            f"Split {split_i+1}/{n_splits} | "
            f"train={len(X_tr_full)} test={len(X_te)}"
        )

        sw_full = _make_sample_weights(y_tr_full, class_weight_map)

        for name, clf in models.items():

            # ── Baseline: train on full train set ─────────────────────────
            _fit_one(clf, X_tr_full, y_tr_full, sw_full)
            baseline_loss    = _logloss_per_mutation(clf, X_te, y_te)
            baseline_pred    = pd.Series(clf.predict(X_te), index=X_te.index)
            baseline_correct = (baseline_pred == y_te).astype(int)

            # ── LOO: remove each training mutation S one at a time ────────
            fits = 0
            for s_idx in X_tr_full.index:

                loo_mask = X_tr_full.index != s_idx
                X_tr_loo = X_tr_full.loc[loo_mask]
                y_tr_loo = y_tr_full.loc[loo_mask]

                if y_tr_loo.nunique() < 2:
                    continue

                sw_loo = _make_sample_weights(y_tr_loo, class_weight_map)
                _fit_one(clf, X_tr_loo, y_tr_loo, sw_loo)
                loo_loss    = _logloss_per_mutation(clf, X_te, y_te)
                loo_pred    = pd.Series(clf.predict(X_te), index=X_te.index)
                loo_correct = (loo_pred == y_te).astype(int)

                # positive = removing S hurt prediction
                delta_loss    = loo_loss    - baseline_loss
                delta_correct = baseline_correct - loo_correct  # +1=flipped correct->wrong
                fits += 1

                for t_idx in X_te.index:
                    records.append({
                        "split":         split_i,
                        "model":         name,
                        "s_idx":         s_idx,
                        "s_mut_to":      mut_to_f.loc[s_idx],
                        "s_structure":   groups_f.loc[s_idx],
                        "s_pos":         pos_keys_f.loc[s_idx],
                        "t_idx":         t_idx,
                        "t_mut_to":      mut_to_f.loc[t_idx],
                        "t_structure":   groups_f.loc[t_idx],
                        "t_pos":         pos_keys_f.loc[t_idx],
                        "same_position": pos_keys_f.loc[s_idx] == pos_keys_f.loc[t_idx],
                        "delta_logloss": delta_loss.loc[t_idx],
                        "delta_correct": delta_correct.loc[t_idx],
                        "y_true":        y_te.loc[t_idx],
                    })

            logger.info(
                f"  Model '{name}': {fits} LOO fits, "
                f"{fits * len(X_te)} records added"
            )

    logger.info(
        f"Done. Total records: {len(records)} | "
        f"Skipped folds: {skipped_folds}/{n_splits}"
    )
    return pd.DataFrame(records)
