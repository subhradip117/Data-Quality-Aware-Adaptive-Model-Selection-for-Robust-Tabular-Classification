from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

PROCESSED_DIR = Path("data/processed")
CORRUPTED_DIR = Path("data/corrupted")

IMBALANCE_RATIOS = [2, 5, 10]


# ============================================================
# DATASET CONFIGURATION
# ============================================================

# Car Evaluation is already strongly imbalanced (about 18.6:1),
# so we keep its original distribution as a natural condition.
DATASETS = {
    "maternal_health": "RiskLevel",
    "mobile_price": "price_range",
}


# ============================================================
# LOAD
# ============================================================

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a processed dataset."""

    path = PROCESSED_DIR / f"{dataset_name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    return pd.read_csv(path)


# ============================================================
# CREATE CONTROLLED MULTICLASS IMBALANCE
# ============================================================

def inject_imbalance(
    df: pd.DataFrame,
    target_column: str,
    ratio: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Create a controlled multiclass imbalance condition.

    The largest class is kept unchanged.

    Every minority class is downsampled to approximately:

        largest_class_count / ratio

    This makes the largest-to-smallest-class ratio close to
    the requested value.

    Example:
        Largest class = 500
        ratio = 5
        minority classes are reduced to about 100 each.

    No labels are changed. Rows are only removed by stratified
    downsampling.
    """

    if ratio <= 1:
        raise ValueError("ratio must be greater than 1.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    rng = np.random.default_rng(seed)

    class_counts = df[target_column].value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "At least two classes are required."
        )

    majority_class = class_counts.index[0]
    majority_count = int(class_counts.iloc[0])

    target_minority_count = max(
        1,
        int(majority_count / ratio)
    )

    sampled_parts = []

    for label, count in class_counts.items():

        class_rows = df[df[target_column] == label]

        if label == majority_class:
            sampled = class_rows.copy()

        else:
            sample_size = min(
                int(count),
                target_minority_count,
            )

            selected_indices = rng.choice(
                class_rows.index.to_numpy(),
                size=sample_size,
                replace=False,
            )

            sampled = class_rows.loc[selected_indices]

        sampled_parts.append(sampled)

    corrupted = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    # Shuffle the final dataset.
    shuffled_indices = rng.permutation(len(corrupted))

    corrupted = corrupted.iloc[
        shuffled_indices
    ].reset_index(drop=True)

    return corrupted


# ============================================================
# SAVE
# ============================================================

def save_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    ratio: int,
) -> Path:

    output_dir = CORRUPTED_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir /
        f"imbalance_{ratio:02d}to1.csv"
    )

    df.to_csv(output_path, index=False)

    return output_path


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(
    dataset_name: str,
    target_column: str,
) -> None:

    df = load_dataset(dataset_name)

    original_counts = df[target_column].value_counts()

    print("\n" + "-" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Original rows: {len(df)}")
    print("\nOriginal class distribution:")

    for label, count in original_counts.items():
        print(f"  {label}: {count}")

    for ratio in IMBALANCE_RATIOS:

        corrupted = inject_imbalance(
            df=df,
            target_column=target_column,
            ratio=ratio,
        )

        new_counts = corrupted[target_column].value_counts()

        actual_ratio = (
            new_counts.max() / new_counts.min()
        )

        output_path = save_dataset(
            corrupted,
            dataset_name,
            ratio,
        )

        print(
            f"\nTarget ratio: {ratio}:1"
        )
        print(
            f"New rows: {len(corrupted)}"
        )

        print("New class distribution:")

        for label, count in new_counts.items():
            print(f"  {label}: {count}")

        print(
            f"Actual max/min ratio: "
            f"{actual_ratio:.2f}:1"
        )

        print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("GENERATING CONTROLLED CLASS-IMBALANCE DATASETS")
    print("=" * 70)

    for dataset_name, target_column in DATASETS.items():

        try:
            process_dataset(
                dataset_name=dataset_name,
                target_column=target_column,
            )

        except Exception as error:
            print(f"\nERROR in {dataset_name}: {error}")

    print("\n" + "=" * 70)
    print("CLASS-IMBALANCE GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
