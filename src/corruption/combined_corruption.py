from pathlib import Path

import pandas as pd

from src.corruption.missing_injector import inject_missing_values
from src.corruption.duplicate_injector import inject_duplicates
from src.corruption.label_noise_injector import inject_label_noise
from src.corruption.outlier_injector import inject_outliers
from src.corruption.imbalance_injector import inject_imbalance


# ============================================================
# SETTINGS
# ============================================================

PROCESSED_DIR = Path("data/processed")
CORRUPTED_DIR = Path("data/corrupted")

SEEDS = {
    "missing": 42,
    "noise": 142,
    "outliers": 242,
    "duplicates": 342,
    "imbalance": 442,
}
MISSING_RATE = 0.10
DUPLICATE_RATE = 0.10
NOISE_RATE = 0.10
OUTLIER_RATE = 0.10
IMBALANCE_RATIO = 5


# ============================================================
# COMBINED EXPERIMENTS
# ============================================================

# Each combination is written explicitly so the experiment is
# reproducible and easy to describe in the research paper.
COMBINATIONS = {
    "missing_noise": ["missing", "noise"],
    "missing_imbalance": ["missing", "imbalance"],
    "noise_imbalance": ["noise", "imbalance"],
    "missing_noise_imbalance": [
        "missing",
        "noise",
        "imbalance",
    ],
    "missing_noise_outliers": [
        "missing",
        "noise",
        "outliers",
    ],
    "missing_duplicates_noise": [
        "missing",
        "duplicates",
        "noise",
    ],
}


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "car_evaluation": {
        "target": "class",
        "supports_outliers": False,
        "supports_controlled_imbalance": False,
    },
    "maternal_health": {
        "target": "RiskLevel",
        "supports_outliers": True,
        "supports_controlled_imbalance": True,
    },
    "mobile_price": {
        "target": "price_range",
        "supports_outliers": True,
        "supports_controlled_imbalance": True,
    },
}


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
# APPLY ONE CORRUPTION
# ============================================================

def apply_corruption(
    df: pd.DataFrame,
    corruption: str,
    target_column: str,
    dataset_config: dict,
) -> pd.DataFrame:

    if corruption == "missing":
        return inject_missing_values(
            df=df,
            rate=MISSING_RATE,
            target_column=target_column,
            seed=SEEDS["missing"],
        )

    if corruption == "noise":
        return inject_label_noise(
            df=df,
            target_column=target_column,
            rate=NOISE_RATE,
            seed=SEEDS["noise"],
        )

    if corruption == "outliers":
        if not dataset_config["supports_outliers"]:
            return df

        return inject_outliers(
            df=df,
            target_column=target_column,
            rate=OUTLIER_RATE,
            seed=SEEDS["outliers"],
        )

    if corruption == "duplicates":
        return inject_duplicates(
            df=df,
            rate=DUPLICATE_RATE,
            seed=SEEDS["duplicates"],
        )

    if corruption == "imbalance":
        if not dataset_config["supports_controlled_imbalance"]:
            return df

        return inject_imbalance(
            df=df,
            target_column=target_column,
            ratio=IMBALANCE_RATIO,
            seed=SEEDS["imbalance"],
        )

    raise ValueError(f"Unknown corruption: {corruption}")


# ============================================================
# CREATE ONE COMBINATION
# ============================================================

def create_combination(
    df: pd.DataFrame,
    target_column: str,
    combination: list[str],
    dataset_config: dict,
) -> pd.DataFrame:
    """
    Apply combined corruption in a fixed order.

    Row-level operations are performed first:
        imbalance -> missing/noise/outliers -> duplicates

    This keeps the experiment reproducible and prevents later
    row additions from changing the intended imbalance sampling.
    """

    row_level = ["imbalance"]
    cell_or_label_level = ["missing", "noise", "outliers"]
    duplicate_level = ["duplicates"]

    result = df.copy()

    # 1. Class imbalance first
    if "imbalance" in combination:
        result = apply_corruption(
            result,
            "imbalance",
            target_column,
            dataset_config,
        )

    # 2. Feature/label corruption
    for corruption in cell_or_label_level:
        if corruption in combination:
            result = apply_corruption(
                result,
                corruption,
                target_column,
                dataset_config,
            )

    # 3. Duplicate rows last
    if "duplicates" in combination:
        result = apply_corruption(
            result,
            "duplicates",
            target_column,
            dataset_config,
        )

    return result


# ============================================================
# SAVE
# ============================================================

def save_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    combination_name: str,
) -> Path:

    output_dir = CORRUPTED_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir /
        f"combined_{combination_name}.csv"
    )

    df.to_csv(output_path, index=False)

    return output_path


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    dataset_name: str,
    dataset_config: dict,
) -> None:

    df = load_dataset(dataset_name)
    target_column = dataset_config["target"]

    print("\n" + "-" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Original rows: {len(df)}")

    for combination_name, combination in COMBINATIONS.items():

        # Skip combinations containing unsupported outliers
        # or controlled imbalance for this dataset.
        if (
            "outliers" in combination
            and not dataset_config["supports_outliers"]
        ):
            print(
                f"SKIP {combination_name}: "
                "outliers not applicable."
            )
            continue

        if (
            "imbalance" in combination
            and not dataset_config["supports_controlled_imbalance"]
        ):
            print(
                f"SKIP {combination_name}: "
                "controlled imbalance not applied."
            )
            continue

        corrupted = create_combination(
            df=df,
            target_column=target_column,
            combination=combination,
            dataset_config=dataset_config,
        )

        output_path = save_dataset(
            corrupted,
            dataset_name,
            combination_name,
        )

        print(
            f"{combination_name:<28} "
            f"→ {len(corrupted):>5} rows "
            f"→ {output_path}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("GENERATING COMBINED CORRUPTION DATASETS")
    print("=" * 70)

    for dataset_name, config in DATASETS.items():

        try:
            process_dataset(
                dataset_name=dataset_name,
                dataset_config=config,
            )

        except Exception as error:
            print(
                f"\nERROR in {dataset_name}: {error}"
            )

    print("\n" + "=" * 70)
    print("COMBINED CORRUPTION GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
