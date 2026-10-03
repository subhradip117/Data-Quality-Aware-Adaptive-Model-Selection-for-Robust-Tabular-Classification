from pathlib import Path

import pandas as pd


# ============================================================
# DATASET PATHS
# ============================================================

DATASETS = {
    "car_evaluation": {
        "path": Path("data/raw/car_evaluation/car_evaluation.csv"),
        "target": "unacc",
    },
    "maternal_health": {
        "path": Path(
            "data/raw/maternal_health/Maternal Health Risk Data Set.csv"
        ),
        "target": "RiskLevel",
    },
    "mobile_price": {
        "path": Path("data/raw/mobile_price/train.csv"),
        "target": "price_range",
    },
}


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


# ============================================================
# SHOW DATASET INFORMATION
# ============================================================

def summarize_dataset(name: str, path: Path, target: str) -> None:
    """Print useful information about one dataset."""

    print("\n" + "=" * 70)
    print(f"DATASET: {name}")
    print("=" * 70)

    try:
        df = load_dataset(path)
    except Exception as error:
        print(f"ERROR: {error}")
        return

    print(f"Rows              : {df.shape[0]}")
    print(f"Columns           : {df.shape[1]}")
    print(f"Target column     : {target}")

    if target not in df.columns:
        print(f"ERROR: Target '{target}' was not found.")
        return

    X = df.drop(columns=[target])
    y = df[target]

    print(f"Feature columns   : {X.shape[1]}")
    print(f"Target data type  : {y.dtype}")
    print(f"Number of classes : {y.nunique()}")

    print("\nClass distribution:")
    print(y.value_counts())

    print("\nClass percentages:")
    percentages = y.value_counts(normalize=True) * 100
    print(percentages.round(2))

    print("\nFeature data types:")
    print(X.dtypes.value_counts())

    print("\nFirst 5 rows:")
    print(df.head())


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    for name, information in DATASETS.items():

        summarize_dataset(
            name=name,
            path=information["path"],
            target=information["target"],
        )

    print("\n" + "=" * 70)
    print("SUMMARY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()