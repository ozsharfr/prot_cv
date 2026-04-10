import random
import numpy as np
import pandas as pd


def make_pos_keys(groups, resnums):
    """Canonical (structure, resnum) key used throughout the package."""
    return groups.astype(str) + "__" + resnums.astype(str)


def is_valid_fold(X_tr, X_te, y_tr, y_te, min_pos_train=3, min_pos_test=3):
    return (
        X_te.shape[0] >= 3
        and y_te.nunique() >= 2
        and np.sum(y_tr == 1) >= min_pos_train
        and np.sum(y_te == 1) >= min_pos_test
    )


def sample_structure_holdouts(groups, n_holdouts, holdout_size, seed):
    structures = groups.unique().tolist()
    rng = random.Random(seed)
    return [tuple(rng.sample(structures, holdout_size)) for _ in range(n_holdouts)]


def sample_position_holdouts(groups, resnums, n_holdouts, holdout_size, seed):
    pos_keys = make_pos_keys(groups, resnums)
    structure_to_positions = (
        pd.DataFrame({"structure": groups, "pos": pos_keys})
        .groupby("structure")["pos"]
        .apply(list)
        .to_dict()
    )
    all_structures = list(structure_to_positions.keys())
    rng = random.Random(seed)

    holdouts = []
    for _ in range(n_holdouts):
        sampled_structures = rng.sample(all_structures, holdout_size)
        held_positions = tuple(
            rng.choice(structure_to_positions[s]) for s in sampled_structures
        )
        holdouts.append(held_positions)
    return holdouts, pos_keys


def random_split(n, test_frac=0.2, seed=42):
    """Simple random train/test index split."""
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    split = int(n * (1 - test_frac))
    return indices[:split], indices[split:]
