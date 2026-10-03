from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROCESSED_DIR = Path("data/processed")


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "car_evaluation": "class",
    "maternal_health": "RiskLevel",
    "mobile_price": "price_range",
}


# ============================================================
# ANALYZE CLASS DISTRIBUTION
# ============================================================

def analyze_dataset(
    dataset_name: str,
    target_column: str,
) -> None:
    """Print the class distribution and safe imbalance targets."""

    path = PROCESSED_DIR / f"{dataset_name}.csv"

    if not path.exists():
        print(f"\nERROR: Dataset not found: {path.resolve()}")
        return

    df = pd.read_csv(path)

    if target_column not in df.columns:
        print(
            f"\nERROR: Target '{target_column}' "
            f"not found in {dataset_name}."
        )
        return

    counts = df[target_column].value_counts()
    total = len(df)

    largest = int(counts.max())
    smallest = int(counts.min())

    imbalance_ratio = largest / smallest

    print("\n" + "-" * 70)
    print(f"DATASET: {dataset_name}")
    print("-" * 70)

    print(f"Rows: {total}")
    print(f"Classes: {len(counts)}")
    print(f"Target: {target_column}")

    print("\nCurrent class distribution:")
    for label, count in counts.items():
        percentage = (count / total) * 100
        print(
            f"  {label}: {count} rows ({percentage:.2f}%)"
        )

    print(
        f"\nCurrent max/min imbalance ratio: "
        f"{imbalance_ratio:.4f}"
    )

    print("\nSuggested experimental imbalance levels:")

    # The plan is based on the smallest class.
    # We will later downsample larger classes so that the
    # minority class is preserved as much as possible.
    for ratio in [2, 5, 10]:
        print(
            f"  1:{ratio} target ratio "
            f"(minority:majority)"
        )

    if imbalance_ratio >= 5:
        print(
            "\nNote: This dataset is already strongly imbalanced. "
            "We should use its original distribution as a natural "
            "imbalance condition instead of forcing another 5%/10% "
            "corruption level."
        )
    elif imbalance_ratio > 1:
        print(
            "\nNote: This dataset already has some natural imbalance. "
            "We will preserve the original condition as a baseline."
        )
    else:
        print(
            "\nNote: This dataset is approximately balanced. "
            "It is suitable for controlled imbalance experiments."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 70)

    for dataset_name, target_column in DATASETS.items():
        analyze_dataset(
            dataset_name=dataset_name,
            target_column=target_column,
        )

    print("\n" + "=" * 70)
    print("IMBALANCE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
