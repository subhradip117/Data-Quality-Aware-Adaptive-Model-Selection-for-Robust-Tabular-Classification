from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_CONFIG = {
    "car_evaluation": {
        "path": RAW_DIR / "car_evaluation" / "car_evaluation.csv",
        "target": "class",
    },
    "maternal_health": {
        "path": RAW_DIR / "maternal_health" / "Maternal Health Risk Data Set.csv",
        "target": "RiskLevel",
    },
    "mobile_price": {
        "path": RAW_DIR / "mobile_price" / "train.csv",
        "target": "price_range",
    },
}


# ============================================================
# CAR EVALUATION
# ============================================================

def load_car_evaluation(path: Path) -> pd.DataFrame:
    """
    Load Car Evaluation data.

    The downloaded CSV does not contain a header row, so we provide
    the correct column names manually.
    """

    columns = [
        "buying",
        "maint",
        "doors",
        "persons",
        "lug_boot",
        "safety",
        "class",
    ]

    df = pd.read_csv(path, header=None, names=columns)

    return df


# ============================================================
# MATERNAL HEALTH
# ============================================================

def load_maternal_health(path: Path) -> pd.DataFrame:
    """Load Maternal Health Risk dataset."""

    return pd.read_csv(path)


# ============================================================
# MOBILE PRICE
# ============================================================

def load_mobile_price(path: Path) -> pd.DataFrame:
    """Load Mobile Price training dataset."""

    return pd.read_csv(path)


# ============================================================
# GENERAL LOADER
# ============================================================

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load one dataset using its configured loader."""

    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available datasets: {list(DATASET_CONFIG.keys())}"
        )

    path = DATASET_CONFIG[dataset_name]["path"]

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{path.resolve()}"
        )

    if dataset_name == "car_evaluation":
        return load_car_evaluation(path)

    if dataset_name == "maternal_health":
        return load_maternal_health(path)

    if dataset_name == "mobile_price":
        return load_mobile_price(path)

    raise ValueError(f"No loader defined for {dataset_name}")


# ============================================================
# STANDARDIZE TARGET
# ============================================================

def split_features_target(
    df: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) and target (y)."""

    target = DATASET_CONFIG[dataset_name]["target"]

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in {dataset_name}."
        )

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    return X, y


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Save standardized dataset as CSV."""

    output_path = PROCESSED_DIR / f"{dataset_name}.csv"

    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("DATASET LOADER TEST")
    print("=" * 70)

    for dataset_name in DATASET_CONFIG:

        print(f"\n[{dataset_name}]")

        try:
            df = load_dataset(dataset_name)
            X, y = split_features_target(df, dataset_name)

            print(f"Rows              : {len(df)}")
            print(f"Features          : {X.shape[1]}")
            print(f"Target            : {y.name}")
            print(f"Classes           : {y.nunique()}")
            print(f"Missing cells     : {df.isna().sum().sum()}")
            print(f"Duplicate rows    : {df.duplicated().sum()}")

            print("\nClass distribution:")
            print(y.value_counts())

            save_processed_dataset(df, dataset_name)

        except Exception as error:
            print(f"ERROR: {error}")

    print("\n" + "=" * 70)
    print("LOADER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
