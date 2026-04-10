import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ── CV summary ────────────────────────────────────────────────────────────────

def summarize_cv(scores_df):
    """Weighted-average AUC and balanced accuracy per model, sorted by AUC."""
    rows = []
    for name, grp in scores_df.groupby("model"):
        w = grp["n"].values
        rows.append({
            "model":        name,
            "auc_mean":     np.average(grp["auc"],     weights=w),
            "bal_acc_mean": np.average(grp["bal_acc"], weights=w),
            "logloss_mean": np.average(grp["logloss"], weights=w),
            "auc_std":      grp["auc"].std(),
            "n_folds":      len(grp),
        })
    return (pd.DataFrame(rows)
              .sort_values("auc_mean", ascending=False)
              .reset_index(drop=True))


# ── LOO delta summaries ───────────────────────────────────────────────────────

def _cell_stats(deltas):
    """Wilcoxon signed-rank test + effect size r for a vector of deltas."""
    from scipy.stats import wilcoxon
    n = len(deltas)
    if n < 5:
        return dict(n=n, mean=float('nan'), median=float('nan'),
                    r=float('nan'), p_value=float('nan'))
    try:
        stat, p = wilcoxon(deltas)
        # Effect size r = Z / sqrt(n), where Z approximated from W statistic
        # scipy returns the W statistic; convert to Z via normal approximation
        import numpy as np
        mu    = n * (n + 1) / 4
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z     = (stat - mu) / sigma
        r     = abs(z) / np.sqrt(n)
    except Exception:
        p, r = float('nan'), float('nan')
    return dict(n=n, mean=float(deltas.mean()), median=float(deltas.median()),
                r=r, p_value=p)


def summarize_loo_logloss(loo_df, same_position_only=False, min_n=5, alpha=0.05):
    """
    Table 1: Wilcoxon signed-rank on delta_logloss per (model, s_mut_to, t_mut_to).
    Effect size r = |Z| / sqrt(n).  Positive mean_delta = removing S hurt logloss.

    Parameters
    ----------
    same_position_only : only include (S, T) pairs at the same position
    min_n              : minimum observations to include a cell
    alpha              : significance threshold for 'significant' flag
    """
    import pandas as pd
    df = loo_df[loo_df["same_position"]] if same_position_only else loo_df
    rows = []
    for (model, s_aa, t_aa), grp in df.groupby(["model", "s_mut_to", "t_mut_to"]):
        stats = _cell_stats(grp["delta_logloss"])
        if stats["n"] < min_n:
            continue
        rows.append({
            "model":       model,
            "s_mut_to":    s_aa,
            "t_mut_to":    t_aa,
            "mean_delta":  stats["mean"],
            "median_delta": stats["median"],
            "effect_r":    stats["r"],
            "p_value":     stats["p_value"],
            "significant": stats["p_value"] < alpha,
            "n":           stats["n"],
        })
    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out

    # FDR correction (Benjamini-Hochberg) per model — less conservative than Bonferroni
    corrected = df_out.copy()
    for model_name, grp in df_out.groupby("model"):
        valid = grp["p_value"].notna()
        if valid.sum() < 2:
            continue
        # BH procedure: sort p-values, compare to (rank/n)*alpha
        p_vals = grp.loc[valid, "p_value"].values
        n_tests = len(p_vals)
        order   = np.argsort(p_vals)
        ranked  = np.empty_like(order)
        ranked[order] = np.arange(1, n_tests + 1)
        p_fdr = np.minimum(1.0, p_vals * n_tests / ranked)
        # Enforce monotonicity (cumulative minimum from largest to smallest)
        p_fdr = np.minimum.accumulate(p_fdr[order][::-1])[::-1][np.argsort(order)]
        corrected.loc[grp[valid].index, "p_fdr"] = p_fdr
    if "p_fdr" not in corrected.columns:
        corrected["p_fdr"] = corrected["p_value"]
    corrected["p_fdr"] = corrected["p_fdr"].fillna(1.0)
    corrected["significant"] = corrected["p_fdr"] < alpha

    return (corrected
              .sort_values("effect_r", ascending=False)
              .reset_index(drop=True))


def summarize_loo_correct(loo_df, same_position_only=False, min_n=5, min_flip_rate=0.05):
    """
    Table 2: mean delta_correct per (model, s_mut_to, t_mut_to).
    delta_correct = correct_baseline - correct_loo.
    Positive mean = removing S flipped more correct predictions to wrong than vice versa.
    No significance test — raw mean flip rate only.

    Parameters
    ----------
    same_position_only : only include (S, T) pairs at the same position
    min_n              : minimum observations to include a cell
    min_flip_rate      : only include cells where abs(mean_flip) > this threshold
    """
    df = loo_df[loo_df["same_position"]] if same_position_only else loo_df
    rows = []
    for (model, s_aa, t_aa), grp in df.groupby(["model", "s_mut_to", "t_mut_to"]):
        if len(grp) < min_n:
            continue
        mean_flip = grp["delta_correct"].mean()
        if abs(mean_flip) <= min_flip_rate:
            continue
        rows.append({
            "model":       model,
            "s_mut_to":    s_aa,
            "t_mut_to":    t_aa,
            "mean_flip":   mean_flip,
            "n":           len(grp),
        })
    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out
    return (df_out
              .sort_values("mean_flip", ascending=False)
              .reset_index(drop=True))


def _pivot_significance(summary_df, model, value_col, min_n=5, alpha=0.05):
    """Pivot with significance masking — for logloss effect r panels."""
    if summary_df.empty or "model" not in summary_df.columns:
        return None, None
    sub = summary_df[
        (summary_df["model"] == model) &
        (summary_df["significant"]) &
        (summary_df["n"] >= min_n)
    ]
    if sub.empty:
        return None, None
    pivot_val = sub.pivot_table(index="s_mut_to", columns="t_mut_to",
                                values=value_col, aggfunc="mean")
    pivot_n   = sub.pivot_table(index="s_mut_to", columns="t_mut_to",
                                values="n", aggfunc="sum")
    return pivot_val, pivot_n


def _pivot_raw(summary_df, model, value_col, min_n=5):
    """Pivot without significance masking — for flip rate panels."""
    if summary_df.empty or "model" not in summary_df.columns:
        return None, None
    sub = summary_df[
        (summary_df["model"] == model) &
        (summary_df["n"] >= min_n)
    ]
    if sub.empty:
        return None, None
    pivot_val = sub.pivot_table(index="s_mut_to", columns="t_mut_to",
                                values=value_col, aggfunc="mean")
    pivot_n   = sub.pivot_table(index="s_mut_to", columns="t_mut_to",
                                values="n", aggfunc="sum")
    return pivot_val, pivot_n


def _draw_heatmap(ax, pivot_val, pivot_n, title, val_col):
    """Draw a single annotated heatmap panel."""
    import numpy as np
    import seaborn as sns

    annot = pivot_val.copy().astype(object)
    for r in pivot_val.index:
        for c in pivot_val.columns:
            try:
                v = pivot_val.loc[r, c]
                n = pivot_n.loc[r, c]
                annot.loc[r, c] = f"{v:.3f}\n(n={int(n)})" if not np.isnan(v) else ""
            except KeyError:
                annot.loc[r, c] = ""

    vmax = np.nanmax(np.abs(pivot_val.values))
    sns.heatmap(
        pivot_val, annot=annot, fmt="", center=0,
        vmin=-vmax, vmax=vmax, cmap="RdBu_r", ax=ax,
        cbar_kws={"label": val_col},
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted mutation (mut_to)")
    ax.set_ylabel("Left-out mutation (mut_to)")


def plot_loo_heatmaps(loo_m1, loo_m2, model,
                      same_position_only=False, min_n=5, alpha=0.05,
                      min_flip_rate=0.05):
    """
    Four heatmaps in a 2x2 grid:
        Row 1: logloss effect r  (FDR-corrected Wilcoxon, significant cells only)  M1 | M2
        Row 2: mean flip rate    (raw mean, cells with abs > min_flip_rate only)    M1 | M2

    Parameters
    ----------
    loo_m1             : run_loo_retraining output with split_mode='random'
    loo_m2             : run_loo_retraining output with split_mode='position'
    model              : model name string
    same_position_only : restrict to same-position (S,T) pairs
    min_n              : minimum observations per cell
    alpha              : FDR significance threshold for logloss panels only
    min_flip_rate      : minimum abs(mean_flip) to show a cell in flip rate panels
    """
    import matplotlib.pyplot as plt

    sum_loss_m1 = summarize_loo_logloss(loo_m1, same_position_only, min_n, alpha)
    sum_loss_m2 = summarize_loo_logloss(loo_m2, same_position_only, min_n, alpha)
    sum_corr_m1 = summarize_loo_correct(loo_m1, same_position_only, min_n, min_flip_rate)
    sum_corr_m2 = summarize_loo_correct(loo_m2, same_position_only, min_n, min_flip_rate)

    pos_label = " (same position)" if same_position_only else ""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Row 1: logloss effect r — significance masked
    for ax, summary, title in [
        (axes[0, 0], sum_loss_m1, f"{model} — M1 random — logloss effect r{pos_label}"),
        (axes[0, 1], sum_loss_m2, f"{model} — M2 position — logloss effect r{pos_label}"),
    ]:
        pivot_val, pivot_n = _pivot_significance(summary, model, "effect_r", min_n, alpha)
        if pivot_val is None:
            ax.set_title(f"{title}\n(no significant cells)")
            ax.axis("off")
        else:
            _draw_heatmap(ax, pivot_val, pivot_n, title, "effect_r")

    # Row 2: flip rate — raw mean, no masking
    for ax, summary, title in [
        (axes[1, 0], sum_corr_m1, f"{model} — M1 random — mean flip rate{pos_label}"),
        (axes[1, 1], sum_corr_m2, f"{model} — M2 position — mean flip rate{pos_label}"),
    ]:
        pivot_val, pivot_n = _pivot_raw(summary, model, "mean_flip", min_n)
        if pivot_val is None:
            ax.set_title(f"{title}\n(no data)")
            ax.axis("off")
        else:
            _draw_heatmap(ax, pivot_val, pivot_n, title, "mean_flip")

    plt.tight_layout()
    plt.show()
    return fig


# kept for backward compat
def summarize_loo_delta(loo_df, same_position_only=False):
    return summarize_loo_logloss(loo_df, same_position_only)

