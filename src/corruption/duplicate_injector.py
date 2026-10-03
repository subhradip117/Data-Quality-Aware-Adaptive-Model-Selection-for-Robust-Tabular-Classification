from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

PROCESSED_DIR = Path("data/processed")
CORRUPTED_DIR = Path("data/corrupted")

DUPLICATE_RATES = [0.05, 0.10, 0.20]


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "car_evaluation": "class",
    "maternal_health": "RiskLevel",
    "mobile_price": "price_range",
}


# ============================================================
# DUPLICATE INJECTOR
# ============================================================

def inject_duplicates(
    df: pd.DataFrame,
    rate: float,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Add duplicate rows to the dataset.

    The original rows are kept unchanged.
    The target column is copied together with the row.

    Example:
        0.10 means add duplicates equal to 10%
        of the original number of rows.
    """

    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1.")

    if len(df) == 0:
        return df.copy()

    rng = np.random.default_rng(seed)

    number_of_duplicates = int(len(df) * rate)

    duplicate_indices = rng.choice(
        len(df),
        size=number_of_duplicates,
        replace=True,
    )

    duplicate_rows = df.iloc[duplicate_indices].copy()

    corrupted_df = pd.concat(
        [df.copy(), duplicate_rows],
        ignore_index=True,
    )

    return corrupted_df


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(dataset_name: str) -> pd.DataFrame:

    path = PROCESSED_DIR / f"{dataset_name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    return pd.read_csv(path)


# ============================================================
# SAVE DATASET
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
        f"duplicates_{percentage:02d}.csv"
    )

    df.to_csv(output_path, index=False)

    return output_path


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(
    dataset_name: str,
) -> None:

    df = load_dataset(dataset_name)

    original_rows = len(df)

    print("\n" + "-" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Original rows: {original_rows}")

    for rate in DUPLICATE_RATES:

        corrupted = inject_duplicates(
            df=df,
            rate=rate,
        )

        added_rows = len(corrupted) - original_rows

        duplicate_rows = int(
            corrupted.duplicated().sum()
        )

        output_path = save_corrupted_dataset(
            corrupted,
            dataset_name,
            rate,
        )

        print(
            f"{int(rate * 100):>3}% duplicates → "
            f"added {added_rows} rows → "
            f"total {len(corrupted)} rows → "
            f"detected duplicates {duplicate_rows} → "
            f"{output_path}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("GENERATING DUPLICATE DATASETS")
    print("=" * 70)

    for dataset_name in DATASETS:

        try:
            process_dataset(dataset_name)

        except Exception as error:
            print(f"\nERROR in {dataset_name}: {error}")

    print("\n" + "=" * 70)
    print("DUPLICATE GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()