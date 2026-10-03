from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

PROCESSED_DIR = Path("data/processed")
CORRUPTED_DIR = Path("data/corrupted")

OUTLIER_RATES = [0.05, 0.10, 0.20]


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "maternal_health": "RiskLevel",
    "mobile_price": "price_range",
}


# ============================================================
# OUTLIER INJECTOR
# ============================================================

def inject_outliers(
    df: pd.DataFrame,
    target_column: str,
    rate: float,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Add artificial outliers to numerical feature values.

    The target column is never modified.
    The corruption rate means the percentage of rows that
    receive one artificial outlier.
    """

    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    corrupted = df.copy()

    numerical_columns = [
        column
        for column in df.select_dtypes(include=np.number).columns
        if column != target_column
    ]

    # Do not alter binary 0/1 columns.
    eligible_columns = [
        column
        for column in numerical_columns
        if df[column].nunique(dropna=True) > 2
    ]

    if not eligible_columns:
        raise ValueError(
            "No suitable numerical feature columns found."
        )

    number_to_corrupt = int(len(df) * rate)

    if number_to_corrupt == 0:
        return corrupted

    rng = np.random.default_rng(seed)

    selected_rows = rng.choice(
        len(df),
        size=number_to_corrupt,
        replace=False,
    )

    for row_index in selected_rows:

        column = rng.choice(eligible_columns)

        series = df[column]

        median = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            std = series.std()

            if pd.isna(std) or std == 0:
                std = max(abs(median) * 0.1, 1.0)

            magnitude = 6 * std
        else:
            magnitude = 6 * iqr

        # Randomly create a high or low outlier.
        if rng.random() < 0.5:
            new_value = median + magnitude
        else:
            new_value = median - magnitude

        # Preserve the original column data type.
        if pd.api.types.is_integer_dtype(df[column]):
            new_value = int(round(new_value))

        corrupted.iat[
            row_index,
            corrupted.columns.get_loc(column),
        ] = new_value

    return corrupted


# ============================================================
# LOAD
# ============================================================

def load_dataset(dataset_name: str) -> pd.DataFrame:

    path = PROCESSED_DIR / f"{dataset_name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    return pd.read_csv(path)


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
        f"outliers_{percentage:02d}.csv"
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

    eligible_columns = [
        column
        for column in df.select_dtypes(include=np.number).columns
        if column != target_column
        and df[column].nunique(dropna=True) > 2
    ]

    print("\n" + "-" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Original rows: {len(df)}")
    print(
        f"Eligible numerical features: "
        f"{len(eligible_columns)}"
    )

    for rate in OUTLIER_RATES:

        corrupted = inject_outliers(
            df=df,
            target_column=target_column,
            rate=rate,
        )

        changed_rows = int(
            (corrupted != df).any(axis=1).sum()
        )

        target_changed = int(
            (corrupted[target_column] != df[target_column]).sum()
        )

        output_path = save_corrupted_dataset(
            corrupted,
            dataset_name,
            rate,
        )

        print(
            f"{int(rate * 100):>3}% outliers → "
            f"changed rows {changed_rows} → "
            f"target changed {target_changed} → "
            f"{output_path}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("GENERATING OUTLIER DATASETS")
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
    print("OUTLIER GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
