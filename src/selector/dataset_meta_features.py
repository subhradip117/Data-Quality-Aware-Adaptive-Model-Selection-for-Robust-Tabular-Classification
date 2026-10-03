from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"

QUALITY_FILE = (
    ROOT
    / "results"
    / "tables"
    / "data_quality_summary.csv"
)

OUTPUT_FILE = (
    ROOT
    / "results"
    / "tables"
    / "dataset_meta_features.csv"
)


# =========================================================
# DATASETS
# =========================================================
DATASETS = {
    "car_evaluation": {
        "file": PROCESSED_DIR / "car_evaluation.csv",
        "target": "class",
    },

    "maternal_health": {
        "file": PROCESSED_DIR / "maternal_health.csv",
        "target": "RiskLevel",
    },

    "mobile_price": {
        "file": PROCESSED_DIR / "mobile_price.csv",
        "target": "price_range",
    },

    "bank_marketing": {
        "file": PROCESSED_DIR / "bank_marketing.csv",
        "target": "y",
    },
}


# =========================================================
# CALCULATE META-FEATURES
# =========================================================

def calculate_meta_features(
    dataset_name,
    file_path,
    target
):

    df = pd.read_csv(file_path)

    # Remove accidental index columns
    unnamed_columns = [
        col
        for col in df.columns
        if col.lower().startswith("unnamed:")
    ]

    if unnamed_columns:
        df = df.drop(
            columns=unnamed_columns
        )

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # -----------------------------------------------------
    # Basic dataset size
    # -----------------------------------------------------

    n_rows = len(df)

    n_features = len(X.columns)

    # -----------------------------------------------------
    # Data types
    # -----------------------------------------------------

    numeric_columns = X.select_dtypes(
        include=[
            "int64",
            "int32",
            "float64",
            "float32"
        ]
    ).columns

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "string",
            "category",
            "bool"
        ]
    ).columns

    n_numeric = len(
        numeric_columns
    )

    n_categorical = len(
        categorical_columns
    )

    categorical_ratio = (
        n_categorical / n_features
        if n_features > 0
        else 0
    )

    numeric_ratio = (
        n_numeric / n_features
        if n_features > 0
        else 0
    )

    # -----------------------------------------------------
    # Target characteristics
    # -----------------------------------------------------

    n_classes = y.nunique()

    class_counts = (
        y.value_counts()
    )

    if len(class_counts) > 1:

        imbalance_ratio = (
            class_counts.max()
            /
            class_counts.min()
        )

    else:

        imbalance_ratio = 1.0

    # -----------------------------------------------------
    # Class entropy
    # -----------------------------------------------------

    probabilities = (
        class_counts
        / class_counts.sum()
    )

    class_entropy = -np.sum(
        probabilities
        * np.log2(probabilities)
    )

    # -----------------------------------------------------
    # Average numerical cardinality
    # -----------------------------------------------------

    if n_numeric > 0:

        numeric_cardinalities = [
            X[col].nunique()
            for col in numeric_columns
        ]

        avg_numeric_cardinality = (
            np.mean(
                numeric_cardinalities
            )
        )

    else:

        avg_numeric_cardinality = 0.0

    # -----------------------------------------------------
    # Average categorical cardinality
    # -----------------------------------------------------

    if n_categorical > 0:

        categorical_cardinalities = [
            X[col].nunique()
            for col in categorical_columns
        ]

        avg_categorical_cardinality = (
            np.mean(
                categorical_cardinalities
            )
        )

    else:

        avg_categorical_cardinality = 0.0

    # -----------------------------------------------------
    # Quality information
    # -----------------------------------------------------

    missing_rate = 0.0
    duplicate_rate = 0.0
    outlier_rate = 0.0

    if QUALITY_FILE.exists():

        quality_df = pd.read_csv(
            QUALITY_FILE
        )

        matching = quality_df[
            quality_df["dataset"]
            == dataset_name
        ]

        if not matching.empty:

            row = matching.iloc[0]

            if "missing_rate_percent" in row:
                missing_rate = (
                    float(row["missing_rate_percent"]) / 100.0
                )

            if "duplicate_rate_percent" in row:
                duplicate_rate = (
                    float(row["duplicate_rate_percent"]) / 100.0
                )

            if "outlier_row_rate_percent" in row:
                outlier_rate = (
                    float(row["outlier_row_rate_percent"]) / 100.0
                )

    # -----------------------------------------------------
    # Return profile
    # -----------------------------------------------------

    return {

        "dataset":
            dataset_name,

        "n_rows":
            n_rows,

        "n_features":
            n_features,

        "n_numeric":
            n_numeric,

        "n_categorical":
            n_categorical,

        "numeric_ratio":
            numeric_ratio,

        "categorical_ratio":
            categorical_ratio,

        "n_classes":
            n_classes,

        "imbalance_ratio":
            imbalance_ratio,

        "class_entropy":
            class_entropy,

        "avg_numeric_cardinality":
            avg_numeric_cardinality,

        "avg_categorical_cardinality":
            avg_categorical_cardinality,

        "missing_rate":
            missing_rate,

        "duplicate_rate":
            duplicate_rate,

        "outlier_rate":
            outlier_rate
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("BUILDING DATASET META-FEATURE PROFILE")
    print("=" * 70)

    profiles = []

    for dataset_name, config in DATASETS.items():

        print(
            f"\nAnalyzing: {dataset_name}"
        )

        profile = calculate_meta_features(
            dataset_name,
            config["file"],
            config["target"]
        )

        profiles.append(
            profile
        )

    result = pd.DataFrame(
        profiles
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("DATASET META-FEATURES")
    print("=" * 70)

    print(
        result.to_string(
            index=False
        )
    )

    print("\nSaved to:")

    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("META-FEATURE PROFILE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()