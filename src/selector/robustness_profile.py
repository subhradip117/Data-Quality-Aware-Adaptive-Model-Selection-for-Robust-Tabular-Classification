from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "results" / "tables"


# =========================================================
# LOAD RESULTS
# =========================================================

def load_results():

    file = RESULTS_DIR / "corruption_results.csv"

    if not file.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{file}"
        )

    return pd.read_csv(file)


# =========================================================
# CALCULATE ROBUSTNESS
# =========================================================

def calculate_robustness(df):

    # Lower performance drop = better robustness.
    #
    # We use Macro-F1 because it is more informative than
    # plain accuracy when class distributions are unequal.

    df["robustness_score"] = (
        1 - df["macro_f1_drop"]
    )

    return df


# =========================================================
# MODEL PROFILE
# =========================================================

def create_model_profile(df):

    profile = (
        df.groupby(
            [
                "dataset",
                "model",
                "corruption"
            ]
        )
        .agg(
            mean_macro_f1=(
                "macro_f1",
                "mean"
            ),

            mean_macro_f1_drop=(
                "macro_f1_drop",
                "mean"
            ),

            worst_macro_f1=(
                "macro_f1",
                "min"
            ),

            worst_macro_f1_drop=(
                "macro_f1_drop",
                "max"
            ),

            mean_accuracy_drop=(
                "accuracy_drop",
                "mean"
            )
        )
        .reset_index()
    )

    profile["robustness_score"] = (
    1 - profile["mean_macro_f1_drop"].clip(lower=0)
).clip(upper=1)
    return profile


# =========================================================
# DATASET-WIDE PROFILE
# =========================================================

def create_dataset_profile(df):

    profile = (
        df.groupby(
            [
                "dataset",
                "model"
            ]
        )
        .agg(
            mean_macro_f1=(
                "macro_f1",
                "mean"
            ),

            mean_macro_f1_drop=(
                "macro_f1_drop",
                "mean"
            ),

            worst_macro_f1=(
                "macro_f1",
                "min"
            ),

            worst_macro_f1_drop=(
                "macro_f1_drop",
                "max"
            ),

            mean_accuracy_drop=(
                "accuracy_drop",
                "mean"
            )
        )
        .reset_index()
    )

    dataset_profile = profile.copy()

    dataset_profile["robustness_score"] = (
    1 - dataset_profile["mean_macro_f1_drop"].clip(lower=0)
).clip(upper=1)

    return dataset_profile
# =========================================================
# CONDITION WINNER
# =========================================================

def create_condition_profile(df):

    rows = []

    for (dataset, condition), group in df.groupby(
        ["dataset", "condition"]
    ):

        best_row = group.loc[
            group["macro_f1"].idxmax()
        ]

        rows.append({
            "dataset": dataset,
            "condition": condition,
            "model": best_row["model"],
            "macro_f1": best_row["macro_f1"],
            "balanced_accuracy": best_row["balanced_accuracy"]
        })

    return pd.DataFrame(rows)

# =========================================================
# SAVE
# =========================================================

def main():

    print("=" * 70)
    print("BUILDING MODEL ROBUSTNESS PROFILE")
    print("=" * 70)

    df = load_results()

    print(
        f"\nLoaded {len(df)} experiment records."
    )

    # -----------------------------------------------------
    # Calculate robustness
    # -----------------------------------------------------

    df = calculate_robustness(df)

    # -----------------------------------------------------
    # Model × corruption profile
    # -----------------------------------------------------

    model_profile = create_model_profile(df)

    model_profile_file = (
        RESULTS_DIR /
        "model_corruption_profile.csv"
    )

    model_profile.to_csv(
        model_profile_file,
        index=False
    )

    # -----------------------------------------------------
    # Dataset × model profile
    # -----------------------------------------------------

    dataset_profile = create_dataset_profile(df)

    dataset_profile_file = (
        RESULTS_DIR /
        "dataset_model_robustness_profile.csv"
    )

    dataset_profile.to_csv(
        dataset_profile_file,
        index=False
    )

    # -----------------------------------------------------
    # Best model for each measured condition
    # -----------------------------------------------------

    condition_profile = create_condition_profile(
        df
    )

    condition_profile_file = (
        RESULTS_DIR /
        "condition_model_profile.csv"
    )

    condition_profile.to_csv(
        condition_profile_file,
        index=False
    )

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    print("\nModel × Corruption Profile")
    print("-" * 70)

    print(
        model_profile.to_string(
            index=False
        )
    )

    print("\nDataset × Model Profile")
    print("-" * 70)

    print(
        dataset_profile.to_string(
            index=False
        )
    )

    print("\nFiles created:")
    print(model_profile_file)
    print(dataset_profile_file)
    print(condition_profile_file)

    print("\n" + "=" * 70)
    print("ROBUSTNESS PROFILE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()