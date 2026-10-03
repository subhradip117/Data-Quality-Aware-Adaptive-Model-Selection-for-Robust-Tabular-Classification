from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

PROCESSED_DIR = Path("data/processed")
CORRUPTED_DIR = Path("data/corrupted")


# ============================================================
# MISSING VALUE INJECTOR
# ============================================================

def inject_missing_values(
    df: pd.DataFrame,
    rate: float,
    target_column: str,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Randomly replace a percentage of feature values with NaN.

    The target column is NEVER modified.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataset.
    rate : float
        Fraction of feature cells to replace.
        Example: 0.05 means 5%.
    target_column : str
        Column containing the prediction target.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Corrupted copy of the dataset.
    """

    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    corrupted_df = df.copy()

    feature_columns = [
        column for column in df.columns
        if column != target_column
    ]

    if not feature_columns:
        raise ValueError("No feature columns available.")

    rng = np.random.default_rng(seed)

    total_feature_cells = len(df) * len(feature_columns)
    number_to_corrupt = int(total_feature_cells * rate)

    if number_to_corrupt == 0:
        return corrupted_df

    # Flatten all feature-cell positions.
    positions = rng.choice(
        total_feature_cells,
        size=number_to_corrupt,
        replace=False,
    )

    number_of_features = len(feature_columns)

    for position in positions:
        row_index = position // number_of_features
        column_index = position % number_of_features
        column_name = feature_columns[column_index]

        corrupted_df.at[
            corrupted_df.index[row_index],
            column_name
        ] = np.nan

    return corrupted_df


# ============================================================
# LOAD DATASET
# ============================================================

def load_processed_dataset(
    dataset_name: str,
) -> pd.DataFrame:
    """Load a processed dataset."""

    path = PROCESSED_DIR / f"{dataset_name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    return pd.read_csv(path)


# ============================================================
# SAVE CORRUPTED DATASET
# ============================================================

def save_corrupted_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    rate: float,
) -> Path:
    """Save a corrupted dataset."""

    percentage = int(rate * 100)

    output_dir = CORRUPTED_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir /
        f"missing_{percentage:02d}.csv"
    )

    df.to_csv(output_path, index=False)

    return output_path


# ============================================================
# MAIN TEST
# ============================================================

def main() -> None:

    dataset_name = "mobile_price"
    target_column = "price_range"

    corruption_rate = 0.05

    print("=" * 70)
    print("MISSING VALUE INJECTION TEST")
    print("=" * 70)

    df = load_processed_dataset(dataset_name)

    print(f"Dataset          : {dataset_name}")
    print(f"Original rows    : {len(df)}")
    print(f"Original columns : {len(df.columns)}")
    print(
        f"Original missing : "
        f"{df.isna().sum().sum()}"
    )

    corrupted_df = inject_missing_values(
        df=df,
        rate=corruption_rate,
        target_column=target_column,
    )

    missing_cells = int(
        corrupted_df.isna().sum().sum()
    )

    feature_count = len(corrupted_df.columns) - 1
    total_feature_cells = len(corrupted_df) * feature_count

    actual_rate = (
        missing_cells / total_feature_cells
        if total_feature_cells > 0
        else 0
    )

    print(
        f"\nTarget protected : {target_column}"
    )
    print(
        f"Requested rate   : {corruption_rate * 100:.2f}%"
    )
    print(
        f"Actual rate      : {actual_rate * 100:.2f}%"
    )
    print(
        f"Missing cells    : {missing_cells}"
    )

    # Make sure target was not corrupted.
    target_missing = corrupted_df[target_column].isna().sum()

    print(
        f"Target missing   : {target_missing}"
    )

    output_path = save_corrupted_dataset(
        corrupted_df,
        dataset_name,
        corruption_rate,
    )

    print(f"\nSaved to         : {output_path}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
