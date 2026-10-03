from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

PROCESSED_DIR = Path("data/processed")
CORRUPTED_DIR = Path("data/corrupted")

CORRUPTION_RATES = [0.05, 0.10, 0.20]


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
# INJECT MISSING VALUES
# ============================================================

def inject_missing_values(
    df: pd.DataFrame,
    target_column: str,
    rate: float,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Add missing values to feature cells only.

    The target column is never modified.
    """

    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    corrupted = df.copy()

    feature_columns = [
        column for column in df.columns
        if column != target_column
    ]

    total_cells = len(corrupted) * len(feature_columns)
    cells_to_corrupt = int(total_cells * rate)

    rng = np.random.default_rng(seed)

    # Use one reproducible permutation so higher rates contain
    # the lower-rate corrupted cells as well.
    positions = rng.permutation(total_cells)[:cells_to_corrupt]

    feature_count = len(feature_columns)

    for position in positions:
        row_index = position // feature_count
        column_index = position % feature_count

        column_name = feature_columns[column_index]

        corrupted.iat[row_index, corrupted.columns.get_loc(column_name)] = np.nan

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

    output_path = output_dir / f"missing_{percentage:02d}.csv"

    df.to_csv(output_path, index=False)

    return output_path


# ============================================================
# GENERATE ALL MISSING-VALUE VERSIONS
# ============================================================

def process_dataset(
    dataset_name: str,
    target_column: str,
) -> None:

    df = load_dataset(dataset_name)

    print("\n" + "-" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"Target: {target_column}")

    for rate in CORRUPTION_RATES:

        corrupted = inject_missing_values(
            df=df,
            target_column=target_column,
            rate=rate,
        )

        feature_columns = [
            column for column in df.columns
            if column != target_column
        ]

        total_feature_cells = len(df) * len(feature_columns)
        missing_cells = int(
            corrupted[feature_columns].isna().sum().sum()
        )

        actual_rate = (
            missing_cells / total_feature_cells
            if total_feature_cells
            else 0
        )

        target_missing = int(
            corrupted[target_column].isna().sum()
        )

        output_path = save_corrupted_dataset(
            corrupted,
            dataset_name,
            rate,
        )

        print(
            f"{int(rate * 100):>3}% missing → "
            f"{missing_cells} cells → "
            f"actual {actual_rate * 100:.2f}% → "
            f"target missing {target_missing} → "
            f"{output_path}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("GENERATING MISSING-VALUE DATASETS")
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
    print("MISSING-VALUE GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
