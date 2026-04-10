import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, log_loss

from .splits import (
    is_valid_fold,
    make_pos_keys,
    sample_structure_holdouts,
    sample_position_holdouts,
)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _fit_one(clf, X_tr, y_tr, sw=None):
    """Fit estimator, routing sample_weight only when class_weight is absent."""
    inner = clf.estimator if hasattr(clf, "estimator") else clf
    has_cw = hasattr(inner, "class_weight") and inner.class_weight is not None
    fit_kwargs = {} if has_cw else ({"sample_weight": sw} if sw is not None else {})
    clf.fit(X_tr, y_tr, **fit_kwargs)


def _record(clf, name, fold_id, X_te, y_te, include_per_mutation=False):
    """
    Compute fold-level metrics. Optionally include per-mutation logloss
    (needed for LOO delta computation).
    """
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]

    base = {
        "model":   name,
        "fold_id": fold_id,
        "auc":     roc_auc_score(y_te, y_proba),
        "bal_acc": balanced_accuracy_score(y_te, y_pred),
        "logloss": log_loss(y_te, y_proba),
        "n":       len(y_te),
    }

    if include_per_mutation:
        # Per-mutation logloss: clip for numerical safety
        eps = 1e-7
        p = np.clip(y_proba, eps, 1 - eps)
        per_mut = -(y_te.values * np.log(p) + (1 - y_te.values) * np.log(1 - p))
        base["per_mutation_logloss"] = dict(zip(X_te.index, per_mut))

    return base


def _make_sample_weights(y_tr, class_weight_map):
    if class_weight_map:
        return y_tr.map(class_weight_map).values
    return None


# ── CV: hold out full structures ──────────────────────────────────────────────

def run_cv_structures(
    X, y, groups,
    models,
    class_weight_map=None,
    n_holdouts=100,
    holdout_size=3,
    min_pos_train=3,
    min_pos_test=3,
    seed=42,
):
    """
    Leave-k-structures-out CV.
    Same structure cannot appear in both train and test.
    """
    holdout_sets = sample_structure_holdouts(groups, n_holdouts, holdout_size, seed)
    sw_full = _make_sample_weights(y, class_weight_map)

    records = []
    for held_out in holdout_sets:
        train_mask = ~groups.isin(held_out)
        test_mask  =  groups.isin(held_out)
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        if not is_valid_fold(X_tr, X_te, y_tr, y_te, min_pos_train, min_pos_test):
            continue

        sw = sw_full[train_mask] if sw_full is not None else None
        for name, clf in models.items():
            _fit_one(clf, X_tr, y_tr, sw)
            records.append(_record(clf, name, str(held_out), X_te, y_te))

    return pd.DataFrame(records)


# ── CV: hold out positions ────────────────────────────────────────────────────

def run_cv_positions(
    X, y, groups, resnums,
    models,
    class_weight_map=None,
    n_holdouts=100,
    holdout_size=3,
    min_pos_train=3,
    min_pos_test=3,
    seed=42,
):
    """
    Leave-k-positions-out CV (one position per structure).
    Same structure may appear in train and test, but not the same position.
    """
    holdout_sets, pos_keys = sample_position_holdouts(
        groups, resnums, n_holdouts, holdout_size, seed
    )
    sw_full = _make_sample_weights(y, class_weight_map)

    records = []
    for held_out in holdout_sets:
        test_mask  =  pos_keys.isin(held_out)
        train_mask = ~pos_keys.isin(held_out)
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        if not is_valid_fold(X_tr, X_te, y_tr, y_te, min_pos_train, min_pos_test):
            continue

        sw = sw_full[train_mask] if sw_full is not None else None
        for name, clf in models.items():
            _fit_one(clf, X_tr, y_tr, sw)
            records.append(_record(clf, name, str(held_out), X_te, y_te))

    return pd.DataFrame(records)


# ── CV: stratified k-fold (no leakage constraint) ─────────────────────────────

def _get_feature_importances(clf, feature_names, X_val=None, y_val=None,
                              use_permutation=False, perm_seed=42):
    """
    Extract feature importances from a fitted estimator.

    Priority:
        1. native feature_importances_ (RF, GB, trees)
        2. absolute coef_ (LR, linear models)
        3. permutation importance — used when use_permutation=True
           or when neither of the above is available (e.g. HGB)

    Parameters
    ----------
    use_permutation : bool
        If True, always use permutation importance (slower but model-agnostic)
    X_val, y_val    : validation data required for permutation importance

    Returns pd.Series indexed by feature name, or None if not computable.
    """
    from sklearn.inspection import permutation_importance as sklearn_perm

    inner = clf.best_estimator_ if hasattr(clf, 'best_estimator_') else clf

    if not use_permutation:
        if hasattr(inner, 'feature_importances_'):
            return pd.Series(inner.feature_importances_, index=feature_names)
        elif hasattr(inner, 'coef_'):
            coef = np.abs(inner.coef_[0]) if inner.coef_.ndim > 1 else np.abs(inner.coef_)
            return pd.Series(coef, index=feature_names)

    # Fallback or explicit: permutation importance
    if X_val is None or y_val is None:
        return None
    result = sklearn_perm(clf, X_val, y_val, n_repeats=5,
                          random_state=perm_seed, scoring='roc_auc')
    return pd.Series(result.importances_mean, index=feature_names)


def run_cv_kfold(
    X, y,
    models,
    groups=None,
    class_weight_map=None,
    n_splits=5,
    seed=42,
    raw=False,
    feature_importance=False,
    use_permutation=False,
):
    """
    Stratified KFold CV — no position or structure leakage constraint.
    Use this when sibling features are included, so mutations from the
    same position can appear in both train and test (siblings in train
    inform prediction of held-out mutation at same position).

    Parameters
    ----------
    groups             : pd.Series, optional
        Structure IDs — stored per sample when raw=True
    raw                : bool
        If False (default): returns fold-level aggregate metrics (for summarize_cv)
        If True: returns one row per test sample with y_true, y_pred, y_proba,
                 structure — for custom per-structure analysis
    feature_importance : bool
        If True: also returns a separate DataFrame with per-feature mean and std
                 importance across folds. Return value becomes (scores_df, imp_df).
    use_permutation    : bool
        If True: use permutation importance for all models (slower, model-agnostic).
        If False (default): use native importances where available (RF, LR),
                 fall back to permutation for models without native support (HGB).

    Returns
    -------
    scores_df                          if feature_importance=False (default)
    (scores_df, importance_df)         if feature_importance=True
    """
    from sklearn.model_selection import StratifiedKFold

    skf            = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    records        = []
    imp_by_fold    = {name: [] for name in models}  # name -> list of pd.Series per fold

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_te = X.iloc[te_idx]
        y_te = y.iloc[te_idx]

        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue

        sw = _make_sample_weights(y_tr, class_weight_map)
        for name, clf in models.items():
            _fit_one(clf, X_tr, y_tr, sw)

            if raw:
                y_proba = clf.predict_proba(X_te)[:, 1]
                y_pred  = clf.predict(X_te)
                y_proba_train = clf.predict_proba(X_tr)[:, 1]
                y_pred_train  = clf.predict(X_tr)
                print (f"Test set : Fold  {fold_i}, model {name}: AUC={roc_auc_score(y_te, y_proba):.3f}, "
                       f"BalAcc={balanced_accuracy_score(y_te, y_pred):.3f}, "
                       f"LogLoss={log_loss(y_te, y_proba):.3f}")
                print (f"Train set : Fold  {fold_i}, model {name}: AUC={roc_auc_score(y_tr, y_proba_train):.3f}, "
                       f"BalAcc={balanced_accuracy_score(y_tr, y_pred_train):.3f}, "
                       f"LogLoss={log_loss(y_tr, y_proba_train):.3f}")
                for idx, prob, pred, true in zip(X_te.index, y_proba, y_pred, y_te):
                    records.append({
                        "fold":      fold_i,
                        "model":     name,
                        "idx":       idx,
                        "y_true":    true,
                        "y_pred":    pred,
                        "y_proba":   prob,
                        "structure": groups.loc[idx] if groups is not None else None,
                    })
            else:
                records.append(_record(clf, name, fold_i, X_te, y_te))

            if feature_importance:
                imp = _get_feature_importances(clf, X.columns,
                                               X_val=X_te, y_val=y_te,
                                               use_permutation=use_permutation)
                if imp is not None:
                    imp_by_fold[name].append(imp)

    scores_df = pd.DataFrame(records)

    if not feature_importance:
        return scores_df

    # Build importance DataFrame: mean and std across folds per model
    imp_rows = []
    for name, folds in imp_by_fold.items():
        if not folds:
            continue
        imp_mat = pd.concat(folds, axis=1)  # features x folds
        imp_rows.append(
            pd.DataFrame({
                "model":   name,
                "feature": imp_mat.index,
                "mean":    imp_mat.mean(axis=1).values,
                "std":     imp_mat.std(axis=1).values,
            })
        )

    importance_df = (
        pd.concat(imp_rows)
        .sort_values(["model", "mean"], ascending=[True, False])
        .reset_index(drop=True)
    ) if imp_rows else pd.DataFrame()

    return scores_df, importance_df