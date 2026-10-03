from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =========================================================
# PROJECT ROOT
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

# Make sure Python can import files from the project root
# even when this script is executed directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =========================================================
# IMPORT EXISTING FUNCTIONS
# =========================================================

from experiments.corruption_experiment import (
    get_models,
    train_evaluate,
    add_missing_values,
    add_duplicates,
    add_label_noise,
    add_outliers,
    add_imbalance,
)


# =========================================================
# PATHS
# =========================================================

RESULTS_DIR = (
    ROOT /
    "results" /
    "tables"
)

SPLIT_DIR = (
    ROOT /
    "data" /
    "processed" /
    "splits"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# OUTPUT FILES
# =========================================================

SINGLE_OUTPUT = (
    RESULTS_DIR /
    "multi_seed_single_results.csv"
)

COMBINED_OUTPUT = (
    RESULTS_DIR /
    "multi_seed_combined_results.csv"
)

SELECTOR_OUTPUT = (
    RESULTS_DIR /
    "multi_seed_selector_results.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR /
    "multi_seed_summary.csv"
)


# =========================================================
# DATASETS
# =========================================================

DATASETS = {
    "car_evaluation": "class",
    "maternal_health": "RiskLevel",
    "mobile_price": "price_range",
    "bank_marketing": "y"
}

# =========================================================
# RANDOM SEEDS
# =========================================================

SEEDS = [
    42,
    123,
    2026
]


# =========================================================
# COMBINED CONDITIONS
# =========================================================

CONDITIONS = {

    "missing_noise": [
        ("missing", 0.10),
        ("label_noise", 0.10)
    ],

    "missing_noise_duplicates": [
        ("missing", 0.10),
        ("label_noise", 0.10),
        ("duplicates", 0.10)
    ],

    "missing_imbalance": [
        ("missing", 0.10),
        ("imbalance", 5)
    ],

    "noise_imbalance": [
        ("label_noise", 0.10),
        ("imbalance", 5)
    ],

    "missing_noise_imbalance": [
        ("missing", 0.10),
        ("label_noise", 0.10),
        ("imbalance", 5)
    ],

    "missing_noise_outliers": [
        ("missing", 0.10),
        ("label_noise", 0.10),
        ("outliers", 0.10)
    ]
}


# =========================================================
# DIFFERENT SEED FOR EACH CORRUPTION TYPE
# =========================================================

SEED_OFFSETS = {

    "missing": 11,

    "label_noise": 22,

    "outliers": 33,

    "duplicates": 44,

    "imbalance": 55
}


# =========================================================
# APPLY ONE CORRUPTION
# =========================================================

def apply_corruption(
    df,
    target,
    corruption,
    severity,
    seed
):

    actual_seed = (
        seed +
        SEED_OFFSETS[corruption]
    )

    if corruption == "missing":

        return add_missing_values(
            df,
            target,
            severity,
            seed=actual_seed
        )

    elif corruption == "label_noise":

        return add_label_noise(
            df,
            target,
            severity,
            seed=actual_seed
        )

    elif corruption == "outliers":

        return add_outliers(
            df,
            target,
            severity,
            seed=actual_seed
        )

    elif corruption == "duplicates":

        return add_duplicates(
            df,
            severity,
            seed=actual_seed
        )

    elif corruption == "imbalance":

        return add_imbalance(
            df,
            target,
            severity,
            seed=actual_seed
        )

    return df.copy()


# =========================================================
# CREATE COMBINED CORRUPTION DATA
# =========================================================

def create_combined_data(
    train_df,
    target,
    components,
    seed
):

    result = train_df.copy()

    # -----------------------------------------------------
    # STEP 1: Apply imbalance first
    # -----------------------------------------------------

    for corruption, severity in components:

        if corruption == "imbalance":

            result = apply_corruption(
                result,
                target,
                corruption,
                severity,
                seed
            )

    # -----------------------------------------------------
    # STEP 2: Apply missing / noise / outliers
    # -----------------------------------------------------
    #
    # IMPORTANT:
    # Duplicates are NOT applied here.
    # They are applied only in STEP 3.
    #

    for corruption, severity in components:

        if corruption not in [
            "imbalance",
            "duplicates"
        ]:

            result = apply_corruption(
                result,
                target,
                corruption,
                severity,
                seed
            )

    # -----------------------------------------------------
    # STEP 3: Apply duplicates LAST
    # -----------------------------------------------------

    for corruption, severity in components:

        if corruption == "duplicates":

            result = apply_corruption(
                result,
                target,
                corruption,
                severity,
                seed
            )

    return result


# =========================================================
# SINGLE-CORRUPTION EXPERIMENT
# =========================================================

def run_single_experiment(
    dataset,
    target,
    train_df,
    test_df,
    seed
):

    rows = []

    conditions = [

        # Missing
        ("missing", 0.05),
        ("missing", 0.10),
        ("missing", 0.20),

        # Duplicates
        ("duplicates", 0.05),
        ("duplicates", 0.10),
        ("duplicates", 0.20),

        # Label noise
        ("label_noise", 0.05),
        ("label_noise", 0.10),
        ("label_noise", 0.20)
    ]

# -----------------------------------------------------
# -----------------------------------------------------
# Outliers are not artificially applied to Car Evaluation,
# but are applied to Bank Marketing, Maternal Health,
# and Mobile Price.
# -----------------------------------------------------

    if dataset != "car_evaluation":

        conditions.extend([

            # Outliers
            ("outliers", 0.05),
            ("outliers", 0.10),
            ("outliers", 0.20)
        ])


    # -----------------------------------------------------
    # Artificial imbalance is not applied to Car Evaluation
    # or Bank Marketing.
    # -----------------------------------------------------

    if dataset not in ["car_evaluation", "bank_marketing"]:

        conditions.extend([

            # Class imbalance
            ("imbalance", 2),
            ("imbalance", 5),
            ("imbalance", 10)
        ])

    # -----------------------------------------------------
    # Run each corruption condition
    # -----------------------------------------------------

    for corruption, severity in conditions:

        print(
            f"    Condition: "
            f"{corruption}_{severity}"
        )

        corrupted_train = apply_corruption(
            train_df,
            target,
            corruption,
            severity,
            seed
        )

        for model_name, model in get_models().items():

            result = train_evaluate(

                corrupted_train,

                test_df,

                target,

                model_name,

                model
            )

            rows.append({

                "seed":
                    seed,

                "dataset":
                    dataset,

                "model":
                    model_name,

                "corruption":
                    corruption,

                "severity":
                    severity,

                "macro_f1":
                    result["macro_f1"],

                "accuracy":
                    result["accuracy"],

                "balanced_accuracy":
                    result["balanced_accuracy"]
            })

    return pd.DataFrame(rows)


# =========================================================
# COMBINED-CORRUPTION EXPERIMENT
# =========================================================

def run_combined_experiment(
    dataset,
    target,
    train_df,
    test_df,
    seed
):

    rows = []

    # -----------------------------------------------------
    # Select applicable conditions
    # -----------------------------------------------------

    available_conditions = []

    for condition, components in CONDITIONS.items():

        # Car Evaluation:
        # no numerical outliers
        # no artificial imbalance

        if dataset == "car_evaluation":

            contains_unsupported = any(
                corruption in [
                    "outliers",
                    "imbalance"
                ]
                for corruption, _ in components
            )

            if contains_unsupported:
                continue

        # Bank Marketing:
        # outliers are allowed, but artificial imbalance is not.

        elif dataset == "bank_marketing":

            contains_unsupported = any(
                corruption == "imbalance"
                for corruption, _ in components
            )

            if contains_unsupported:
                continue

        available_conditions.append(
            (
                condition,
                components
            )
        )

    # -----------------------------------------------------
    # Run combined conditions
    # -----------------------------------------------------

    for condition, components in available_conditions:

        print(
            f"    Combined condition: "
            f"{condition}"
        )

        corrupted_train = create_combined_data(

            train_df,

            target,

            components,

            seed
        )

        print(
            f"    Training rows: "
            f"{len(corrupted_train)}"
        )

        for model_name, model in get_models().items():

            result = train_evaluate(

                corrupted_train,

                test_df,

                target,

                model_name,

                model
            )

            rows.append({

                "seed":
                    seed,

                "dataset":
                    dataset,

                "condition":
                    condition,

                "model":
                    model_name,

                "macro_f1":
                    result["macro_f1"],

                "accuracy":
                    result["accuracy"],

                "balanced_accuracy":
                    result["balanced_accuracy"]
            })

    return pd.DataFrame(rows)


# =========================================================
# ADD CLEAN BASELINE
# =========================================================

def add_clean_baseline(
    single_df
):

    baseline_file = (
        RESULTS_DIR /
        "baseline_results.csv"
    )

    if not baseline_file.exists():

        raise FileNotFoundError(
            f"Baseline file not found:\n"
            f"{baseline_file}"
        )

    baseline = pd.read_csv(
        baseline_file
    )

    baseline_small = baseline[
        [
            "dataset",
            "model",
            "macro_f1"
        ]
    ].rename(

        columns={
            "macro_f1":
                "clean_macro_f1"
        }
    )

    return single_df.merge(

        baseline_small,

        on=[
            "dataset",
            "model"
        ],

        how="left"
    )


# =========================================================
# CALCULATE RELATIVE CORRUPTION PENALTY
# =========================================================

def calculate_relative_drop(
    single_df
):

    df = single_df.copy()

    # Performance decrease
    df["absolute_drop"] = (

        df["clean_macro_f1"]

        -

        df["macro_f1"]
    )

    # Negative values mean the corrupted version happened
    # to perform slightly better.
    #
    # Such cases are treated as zero penalty.

    df["absolute_drop"] = (
        df["absolute_drop"]
        .clip(lower=0)
    )

    # Convert to relative degradation.

    df["relative_drop"] = np.where(

        df["clean_macro_f1"] > 0,

        (
            df["absolute_drop"]
            /
            df["clean_macro_f1"]
        ),

        0
    )

    return df


# =========================================================
# ADAPTIVE SELECTOR
# =========================================================

def calculate_selector(
    single_df,
    combined_df
):

    single_df = add_clean_baseline(
        single_df
    )

    single_df = calculate_relative_drop(
        single_df
    )

    rows = []

    # -----------------------------------------------------
    # Process each random seed
    # -----------------------------------------------------

    for seed in SEEDS:

        print("\n" + "-" * 70)
        print(
            f"SELECTOR FOR SEED: {seed}"
        )

        seed_single = single_df[
            single_df["seed"] == seed
        ]

        seed_combined = combined_df[
            combined_df["seed"] == seed
        ]

        # -------------------------------------------------
        # Process each dataset
        # -------------------------------------------------

        for dataset in DATASETS.keys():

            dataset_single = seed_single[
                seed_single["dataset"]
                == dataset
            ]

            dataset_combined = seed_combined[
                seed_combined["dataset"]
                == dataset
            ]

            if dataset_combined.empty:
                continue

            print(
                f"\nDataset: {dataset}"
            )

            # -------------------------------------------------
            # Process each combined condition
            # -------------------------------------------------

            for condition in dataset_combined[
                "condition"
            ].unique():

                actual = dataset_combined[
                    dataset_combined[
                        "condition"
                    ] == condition
                ]

                predictions = []

                # ---------------------------------------------
                # Estimate performance of every model
                # ---------------------------------------------

                for model in get_models().keys():

                    clean_row = dataset_single[
                        dataset_single["model"]
                        == model
                    ]

                    if clean_row.empty:
                        continue

                    clean_f1 = float(
                        clean_row.iloc[0][
                            "clean_macro_f1"
                        ]
                    )

                    estimated_f1 = clean_f1

                    # -----------------------------------------
                    # Get components of this condition
                    # -----------------------------------------

                    components = CONDITIONS[
                        condition
                    ]

                    for corruption, severity in components:

                        penalty_row = dataset_single[

                            (
                                dataset_single[
                                    "model"
                                ]
                                == model
                            )

                            &

                            (
                                dataset_single[
                                    "corruption"
                                ]
                                == corruption
                            )

                            &

                            (
                                dataset_single[
                                    "severity"
                                ]
                                == severity
                            )
                        ]

                        if penalty_row.empty:
                            continue

                        penalty = float(
                            penalty_row.iloc[0][
                                "relative_drop"
                            ]
                        )

                        estimated_f1 *= max(
                            0.0,
                            1.0 - penalty
                        )

                    predictions.append({

                        "model":
                            model,

                        "estimated_f1":
                            estimated_f1
                    })

                prediction_df = pd.DataFrame(
                    predictions
                )

                if prediction_df.empty:
                    continue

                # ---------------------------------------------
                # Adaptive model selection
                # ---------------------------------------------

                selected_row = prediction_df.loc[

                    prediction_df[
                        "estimated_f1"
                    ].idxmax()
                ]

                selected_model = (
                    selected_row["model"]
                )

                predicted_f1 = float(
                    selected_row["estimated_f1"]
                )

                # ---------------------------------------------
                # Actual oracle model
                # ---------------------------------------------

                oracle_row = actual.loc[
                    actual["macro_f1"].idxmax()
                ]

                oracle_model = (
                    oracle_row["model"]
                )

                oracle_f1 = float(
                    oracle_row["macro_f1"]
                )

                # ---------------------------------------------
                # Actual performance of selected model
                # ---------------------------------------------

                selected_actual = actual[
                    actual["model"]
                    == selected_model
                ]

                if selected_actual.empty:
                    continue

                selected_actual_f1 = float(
                    selected_actual.iloc[0][
                        "macro_f1"
                    ]
                )

                # ---------------------------------------------
                # Store result
                # ---------------------------------------------

                rows.append({

                    "seed":
                        seed,

                    "dataset":
                        dataset,

                    "condition":
                        condition,

                    "selected_model":
                        selected_model,

                    "predicted_macro_f1":
                        predicted_f1,

                    "actual_selected_macro_f1":
                        selected_actual_f1,

                    "oracle_model":
                        oracle_model,

                    "oracle_macro_f1":
                        oracle_f1,

                    "regret":
                        (
                            oracle_f1
                            -
                            selected_actual_f1
                        ),

                    "correct_selection":
                        (
                            selected_model
                            ==
                            oracle_model
                        )
                })

                print(
                    f"  {condition} "
                    f"→ {selected_model} "
                    f"({'CORRECT' if selected_model == oracle_model else 'WRONG'})"
                )

    return pd.DataFrame(rows)


# =========================================================
# CREATE SUMMARY
# =========================================================

def create_summary(
    selector_results
):

    rows = []

    # -----------------------------------------------------
    # Summary per seed
    # -----------------------------------------------------

    for seed in SEEDS:

        subset = selector_results[
            selector_results["seed"] == seed
        ]

        if subset.empty:
            continue

        rows.append({

            "seed":
                seed,

            "conditions_tested":
                len(subset),

            "correct_selections":
                int(
                    subset[
                        "correct_selection"
                    ].sum()
                ),

            "selection_accuracy":
                subset[
                    "correct_selection"
                ].mean(),

            "mean_regret":
                subset[
                    "regret"
                ].mean()
        })

    # -----------------------------------------------------
    # Overall
    # -----------------------------------------------------

    if not selector_results.empty:

        rows.append({

            "seed":
                "OVERALL",

            "conditions_tested":
                len(selector_results),

            "correct_selections":
                int(
                    selector_results[
                        "correct_selection"
                    ].sum()
                ),

            "selection_accuracy":
                selector_results[
                    "correct_selection"
                ].mean(),

            "mean_regret":
                selector_results[
                    "regret"
                ].mean()
        })

    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("MULTI-SEED ADAPTIVE SELECTOR VALIDATION")
    print("=" * 70)

    print(
        f"\nSeeds: {SEEDS}"
    )

    print(
        "\nClean train/test splits will remain fixed."
    )

    print(
        "Only the corruption randomization changes."
    )

    # =====================================================
    # RUN ALL SEEDS
    # =====================================================

    all_single = []

    all_combined = []

    for seed in SEEDS:

        print("\n" + "=" * 70)
        print(
            f"STARTING SEED {seed}"
        )
        print("=" * 70)

        for dataset, target in DATASETS.items():

            print("\n" + "-" * 70)

            print(
                f"Dataset: {dataset}"
            )

            train_file = (
                SPLIT_DIR /
                f"{dataset}_train.csv"
            )

            test_file = (
                SPLIT_DIR /
                f"{dataset}_test.csv"
            )

            if not train_file.exists():

                raise FileNotFoundError(
                    f"Train split not found:\n"
                    f"{train_file}"
                )

            if not test_file.exists():

                raise FileNotFoundError(
                    f"Test split not found:\n"
                    f"{test_file}"
                )

            train_df = pd.read_csv(
                train_file
            )

            test_df = pd.read_csv(
                test_file
            )

            print(
                f"Train rows: {len(train_df)}"
            )

            print(
                f"Test rows:  {len(test_df)}"
            )

            # -------------------------------------------------
            # Single corruption
            # -------------------------------------------------

            print(
                "\nRunning single-corruption experiments..."
            )

            single = run_single_experiment(

                dataset,

                target,

                train_df,

                test_df,

                seed
            )

            all_single.append(
                single
            )

            # -------------------------------------------------
            # Combined corruption
            # -------------------------------------------------

            print(
                "\nRunning combined-corruption experiments..."
            )

            combined = run_combined_experiment(

                dataset,

                target,

                train_df,

                test_df,

                seed
            )

            all_combined.append(
                combined
            )

    # =====================================================
    # COMBINE ALL RESULTS
    # =====================================================

    single_df = pd.concat(

        all_single,

        ignore_index=True
    )

    combined_df = pd.concat(

        all_combined,

        ignore_index=True
    )

    # =====================================================
    # SAVE RAW RESULTS
    # =====================================================

    single_df.to_csv(

        SINGLE_OUTPUT,

        index=False
    )

    combined_df.to_csv(

        COMBINED_OUTPUT,

        index=False
    )

    print("\n" + "=" * 70)

    print(
        "RAW MULTI-SEED EXPERIMENTS COMPLETE"
    )

    print("=" * 70)

    print(
        f"\nSingle records: "
        f"{len(single_df)}"
    )

    print(
        f"Combined records: "
        f"{len(combined_df)}"
    )

    # =====================================================
    # RUN ADAPTIVE SELECTOR
    # =====================================================

    print("\n" + "=" * 70)

    print(
        "RUNNING ADAPTIVE SELECTOR"
    )

    print("=" * 70)

    selector_results = calculate_selector(

        single_df,

        combined_df
    )

    # =====================================================
    # SAVE SELECTOR RESULTS
    # =====================================================

    selector_results.to_csv(

        SELECTOR_OUTPUT,

        index=False
    )

    # =====================================================
    # CREATE SUMMARY
    # =====================================================

    summary = create_summary(

        selector_results
    )

    summary.to_csv(

        SUMMARY_OUTPUT,

        index=False
    )

    # =====================================================
    # DISPLAY FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 70)

    print(
        "MULTI-SEED SUMMARY"
    )

    print("=" * 70)

    if summary.empty:

        print(
            "\nNo selector results were generated."
        )

    else:

        print(
            summary.to_string(
                index=False
            )
        )

    # =====================================================
    # DISPLAY FILES
    # =====================================================

    print("\nFiles saved:")

    print(
        SINGLE_OUTPUT
    )

    print(
        COMBINED_OUTPUT
    )

    print(
        SELECTOR_OUTPUT
    )

    print(
        SUMMARY_OUTPUT
    )

    print("\n" + "=" * 70)

    print(
        "MULTI-SEED VALIDATION COMPLETE"
    )

    print("=" * 70)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()