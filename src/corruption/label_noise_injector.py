from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

PROCESSED_DIR = Path("data/processed")
CORRUPTED_DIR = Path("data/corrupted")

NOISE_RATES = [0.05, 0.10, 0.20]


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "car_evaluation": "class",
    "maternal_health": "RiskLevel",
    "mobile_price": "price_range",
}


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load one processed dataset."""

    path = PROCESSED_DIR / f"{dataset_name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    return pd.read_csv(path)


# ============================================================
# LABEL NOISE INJECTOR
# ============================================================

def inject_label_noise(
    df: pd.DataFrame,
    target_column: str,
    rate: float,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Randomly change a percentage of target labels.

    The feature columns are never changed.

    For multiclass datasets, a selected label is changed to
    a different class chosen randomly from the other classes.

    For binary datasets, the label is flipped to the other class.
    """

    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    corrupted = df.copy()

    unique_labels = list(corrupted[target_column].dropna().unique())

    if len(unique_labels) < 2:
        raise ValueError(
            "Label noise requires at least two target classes."
        )

    number_to_corrupt = int(len(corrupted) * rate)

    if number_to_corrupt == 0:
        return corrupted

    rng = np.random.default_rng(seed)

    # Use a reproducible subset so 5%, 10%, and 20% are nested.
    selected_indices = rng.permutation(len(corrupted))[
        :number_to_corrupt
    ]

    for row_index in selected_indices:

        current_label = corrupted.iloc[
            row_index
        ][target_column]

        possible_labels = [
            label
            for label in unique_labels
            if label != current_label
        ]

        new_label = rng.choice(possible_labels)

        corrupted.iat[
            row_index,
            corrupted.columns.get_loc(target_column),
        ] = new_label

    return corrupted


# ============================================================
# SAVE
# ============================================================

def save_corrupted_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    rate: float,
) -> Path:

    percentage = int(rate * 100)

    output_dir = CORRUPTED_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir /
        f"label_noise_{percentage:02d}.csv"
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

    original_labels = df[target_column].copy()

    print("\n" + "-" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Rows: {len(df)}")
    print(f"Target: {target_column}")
    print(f"Classes: {df[target_column].nunique()}")

    for rate in NOISE_RATES:

        corrupted = inject_label_noise(
            df=df,
            target_column=target_column,
            rate=rate,
        )

        changed_labels = int(
            (corrupted[target_column] != original_labels).sum()
        )

        actual_rate = (
            changed_labels / len(df)
            if len(df) > 0
            else 0
        )

        output_path = save_corrupted_dataset(
            corrupted,
            dataset_name,
            rate,
        )

        print(
            f"{int(rate * 100):>3}% label noise → "
            f"changed {changed_labels} labels → "
            f"actual {actual_rate * 100:.2f}% → "
            f"{output_path}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("GENERATING LABEL-NOISE DATASETS")
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
    print("LABEL-NOISE GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
