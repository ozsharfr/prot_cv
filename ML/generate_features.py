"""
generate_features.py — Prepares feature matrix for models predicting DDG or prediction error.

Validation: Leave-One-Protein-Out (LOPO) CV + per-structure GroupKFold CV.
Run: python generate_features.py --target prediction_error
     python generate_features.py --target DDG
     python generate_features.py --target DDG --include-foldx
"""

from common import (
    _PROJECT_ROOT, log, np, pd, plt,
    RESULTS_DIR, FIGURES_DIR, DEFAULT_TARGET,
    load_results, prepare_xy, build_pdb_paths,
    lopo_cv, per_structure_cv,
    get_feature_importances, plot_importances, plot_per_structure_results,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Random Forest regressor for DDG prediction")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        choices=["prediction_error", "DDG"])
    parser.add_argument("--include-foldx", action="store_true",
                        help="Add ddg_foldx as feature when target=DDG")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df        = load_results(RESULTS_DIR)
    pdb_paths = build_pdb_paths(df)

    X, y, groups, feature_names, df_valid = prepare_xy(
        df, target=args.target,
        include_foldx=args.include_foldx,
        pdb_paths=pdb_paths or None
    )

    # Save feature matrix (include mutation identifiers for downstream assembly joins)
    feature_df = X.copy()
    feature_df[args.target] = y.values
    feature_df["pdb_id"]    = groups.values
    for col in ["chain", "resnum", "wt_aa", "mut_aa"]:
        if col in df_valid.columns:
            feature_df[col] = df_valid[col].values
    out_parquet = RESULTS_DIR / f"feature_matrix_{args.target}.parquet"
    feature_df.to_parquet(out_parquet, index=False)
    log.info("Feature matrix saved: %s (%d rows × %d features)",
             out_parquet, len(feature_df), len(feature_names))

    


if __name__ == "__main__":
    main()