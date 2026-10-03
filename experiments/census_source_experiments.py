
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# CENSUS SOURCE-DATA EXPERIMENT
# =========================================================
#
# Purpose:
#   Add Census KDD as a SOURCE dataset for the selector without
#   rerunning the enormous full Census corruption experiment.
#
# Fixed Census source subset:
#   10,000 stratified rows from the project training split.
#
# Existing full Census test set remains untouched.
#
# We generate only the corruption responses actually needed by
# the selector's combined-condition formulas:
#
#   Single:
#       missing_0.10
#       label_noise_0.10
#       duplicates_0.10
#       outliers_0.10
#
#   Combined:
#       missing_noise
#       missing_noise_duplicates
#       missing_noise_outliers
#
# 3 seeds are used for corruption randomization.
#
# Total:
#   5 clean models
#   + 4 single conditions * 5 models * 3 seeds
#   + 3 combined conditions * 5 models * 3 seeds
#   = 110 model fits
#
# This is intentionally much smaller than the full Census
# corruption experiment.
# =========================================================


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Reuse the project's existing implementations.
from experiments.corruption_experiment import (
    get_models,
    train_evaluate,
)

from experiments.multi_seed_validation import (
    create_combined_data,
    CONDITIONS,
)


# =========================================================
# PATHS
# =========================================================

SPLIT_DIR = ROOT / "data" / "processed" / "splits"
RESULTS_DIR = ROOT / "results" / "tables"
OUT_DIR = RESULTS_DIR / "census_source_experiments"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_FILE = SPLIT_DIR / "census_income_kdd_train.csv"
TEST_FILE = SPLIT_DIR / "census_income_kdd_test.csv"

CLEAN_OUTPUT = OUT_DIR / "census_source_clean_baseline.csv"
SINGLE_OUTPUT = OUT_DIR / "census_source_single_results.csv"
COMBINED_OUTPUT = OUT_DIR / "census_source_combined_results.csv"


# =========================================================
# SETTINGS
# =========================================================

DATASET = "census_income_kdd"
TARGET = "CLASS"

SUBSET_SIZE = 10000
SUBSET_RANDOM_STATE = 42

SEEDS = [42, 123, 2026]

SINGLE_CONDITIONS = [
    ("missing", 0.10),
    ("label_noise", 0.10),
    ("duplicates", 0.10),
    ("outliers", 0.10),
]

COMBINED_CONDITIONS = [
    "missing_noise",
    "missing_noise_duplicates",
    "missing_noise_outliers",
]


# =========================================================
# LOAD FIXED CENSUS SUBSET
# =========================================================

def load_fixed_data():

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training split not found:\n{TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test split not found:\n{TEST_FILE}"
        )

    full_train = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    if TARGET not in full_train.columns:
        raise KeyError(
            f"Target '{TARGET}' not found in training data."
        )

    if TARGET not in test_df.columns:
        raise KeyError(
            f"Target '{TARGET}' not found in test data."
        )

    if len(full_train) < SUBSET_SIZE:
        raise ValueError(
            f"Census training data has only {len(full_train):,} rows; "
            f"{SUBSET_SIZE:,} are required."
        )

    train_subset, _ = train_test_split(
        full_train,
        train_size=SUBSET_SIZE,
        stratify=full_train[TARGET],
        random_state=SUBSET_RANDOM_STATE,
    )

    train_subset = train_subset.reset_index(drop=True)

    return train_subset, test_df


# =========================================================
# CLEAN BASELINE
# =========================================================

def run_clean_baseline(
    train_df,
    test_df,
):
    print("\n" + "=" * 70)
    print("CENSUS SOURCE CLEAN BASELINE")
    print("=" * 70)

    rows = []

    for model_name, model in get_models().items():

        print("\n" + "-" * 70)
        print(f"Clean baseline: {model_name}")
        print("-" * 70)

        result = train_evaluate(
            train_df,
            test_df,
            TARGET,
            model_name,
            model,
        )

        print(
            f"Accuracy:          {result['accuracy']:.6f}"
        )
        print(
            f"Balanced Accuracy: {result['balanced_accuracy']:.6f}"
        )
        print(
            f"Macro-F1:          {result['macro_f1']:.6f}"
        )

        rows.append({
            "dataset": DATASET,
            "model": model_name,
            "accuracy": result["accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "macro_f1": result["macro_f1"],
        })

    clean_df = pd.DataFrame(rows)

    clean_df.to_csv(
        CLEAN_OUTPUT,
        index=False,
    )

    return clean_df


# =========================================================
# SINGLE CORRUPTION
# =========================================================

def run_single_experiments(
    train_df,
    test_df,
):
    print("\n" + "=" * 70)
    print("CENSUS SOURCE SINGLE-CORRUPTION EXPERIMENTS")
    print("=" * 70)

    rows = []

    # Use the same corruption application used by the main
    # project through multi_seed_validation.
    from experiments.multi_seed_validation import apply_corruption

    for seed in SEEDS:

        print("\n" + "-" * 70)
        print(f"Seed: {seed}")
        print("-" * 70)

        for corruption, severity in SINGLE_CONDITIONS:

            print(
                f"  Condition: "
                f"{corruption}_{severity}"
            )

            corrupted_train = apply_corruption(
                train_df,
                TARGET,
                corruption,
                severity,
                seed,
            )

            for model_name, model in get_models().items():

                result = train_evaluate(
                    corrupted_train,
                    test_df,
                    TARGET,
                    model_name,
                    model,
                )

                rows.append({
                    "seed": seed,
                    "dataset": DATASET,
                    "model": model_name,
                    "corruption": corruption,
                    "severity": severity,
                    "macro_f1": result["macro_f1"],
                    "accuracy": result["accuracy"],
                    "balanced_accuracy": result[
                        "balanced_accuracy"
                    ],
                })

    single_df = pd.DataFrame(rows)

    single_df.to_csv(
        SINGLE_OUTPUT,
        index=False,
    )

    return single_df


# =========================================================
# COMBINED CORRUPTION
# =========================================================

def run_combined_experiments(
    train_df,
    test_df,
):
    print("\n" + "=" * 70)
    print("CENSUS SOURCE COMBINED-CORRUPTION EXPERIMENTS")
    print("=" * 70)

    rows = []

    for seed in SEEDS:

        print("\n" + "-" * 70)
        print(f"Seed: {seed}")
        print("-" * 70)

        for condition in COMBINED_CONDITIONS:

            components = CONDITIONS[condition]

            print(
                f"  Combined condition: {condition}"
            )

            corrupted_train = create_combined_data(
                train_df,
                TARGET,
                components,
                seed,
            )

            print(
                f"  Training rows: "
                f"{len(corrupted_train)}"
            )

            for model_name, model in get_models().items():

                result = train_evaluate(
                    corrupted_train,
                    test_df,
                    TARGET,
                    model_name,
                    model,
                )

                rows.append({
                    "seed": seed,
                    "dataset": DATASET,
                    "condition": condition,
                    "model": model_name,
                    "macro_f1": result["macro_f1"],
                    "accuracy": result["accuracy"],
                    "balanced_accuracy": result[
                        "balanced_accuracy"
                    ],
                })

    combined_df = pd.DataFrame(rows)

    combined_df.to_csv(
        COMBINED_OUTPUT,
        index=False,
    )

    return combined_df


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("CENSUS KDD SOURCE-DATA EXPERIMENT")
    print("=" * 70)

    train_df, test_df = load_fixed_data()

    print(
        f"\nFull Census train rows: "
        f"{len(pd.read_csv(TRAIN_FILE)):,}"
    )
    print(
        f"Fixed source subset: "
        f"{len(train_df):,}"
    )
    print(
        f"Census test rows: "
        f"{len(test_df):,}"
    )

    print(
        "\nThe same fixed Census subset is used for all 3 seeds."
    )
    print(
        "Only corruption randomization changes."
    )

    clean_df = run_clean_baseline(
        train_df,
        test_df,
    )

    single_df = run_single_experiments(
        train_df,
        test_df,
    )

    combined_df = run_combined_experiments(
        train_df,
        test_df,
    )

    print("\n" + "=" * 70)
    print("CENSUS SOURCE EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"\nClean baseline records:   {len(clean_df)}"
    )
    print(
        f"Single records:           {len(single_df)}"
    )
    print(
        f"Combined records:         {len(combined_df)}"
    )

    print("\nClean baseline:")
    print(
        clean_df[
            [
                "dataset",
                "model",
                "macro_f1",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(CLEAN_OUTPUT)
    print(SINGLE_OUTPUT)
    print(COMBINED_OUTPUT)

    print("\n" + "=" * 70)
    print("NEXT: RUN CENSUS-AUGMENTED SELECTOR")
    print("=" * 70)


if __name__ == "__main__":
    main()
