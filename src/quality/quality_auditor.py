from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "car_evaluation": {
        "target": "class",
    },

    "maternal_health": {
        "target": "RiskLevel",
    },

    "mobile_price": {
        "target": "price_range",
    },

    "bank_marketing": {
        "target": "y",
    },
}


# ============================================================
# OUTLIER DETECTION
# ============================================================

def calculate_iqr_outlier_rate(df: pd.DataFrame) -> float:
    """
    Estimate the percentage of rows that contain at least one
    IQR-based numerical outlier.

    A row is counted once even if it contains outliers in multiple
    columns.
    """

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        return 0.0

    outlier_mask = pd.DataFrame(
        False,
        index=df.index,
        columns=numeric_df.columns,
    )

    for column in numeric_df.columns:
        q1 = numeric_df[column].quantile(0.25)
        q3 = numeric_df[column].quantile(0.75)

        iqr = q3 - q1

        # Constant columns cannot meaningfully have IQR outliers.
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask[column] = (
            (numeric_df[column] < lower)
            | (numeric_df[column] > upper)
        )

    row_has_outlier = outlier_mask.any(axis=1)

    return float(row_has_outlier.mean() * 100)


# ============================================================
# LABEL CONFLICT DETECTION
# ============================================================

def calculate_label_conflict_rate(
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """
    Estimate label conflicts among identical feature rows.

    If identical feature rows have more than one target label,
    those rows are counted as potential label conflicts.

    Important:
    This is only a quality warning, not proof that a label is wrong.
    """

    if X.empty:
        return 0.0

    temp = X.copy()
    temp["__target__"] = y.to_numpy()

    grouped = temp.groupby(
        list(X.columns),
        dropna=False,
        sort=False,
    )["__target__"].nunique()

    conflict_groups = grouped[grouped > 1]

    if conflict_groups.empty:
        return 0.0

    conflict_row_count = 0

    for group_values in conflict_groups.index:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        mask = np.ones(len(temp), dtype=bool)

        for column, value in zip(X.columns, group_values):
            if pd.isna(value):
                mask &= temp[column].isna()
            else:
                mask &= temp[column].eq(value)

        conflict_row_count += int(mask.sum())

    return float((conflict_row_count / len(X)) * 100)


# ============================================================
# CLASS IMBALANCE
# ============================================================

def calculate_class_imbalance(y: pd.Series) -> dict:
    """Calculate class counts, percentages, and imbalance ratio."""

    counts = y.value_counts(dropna=False)
    percentages = (counts / len(y)) * 100

    if len(counts) > 1:
        imbalance_ratio = float(counts.max() / counts.min())
    else:
        imbalance_ratio = 1.0

    return {
        "class_counts": {
            str(label): int(count)
            for label, count in counts.items()
        },
        "class_percentages": {
            str(label): round(float(percent), 2)
            for label, percent in percentages.items()
        },
        "imbalance_ratio": round(imbalance_ratio, 4),
    }


# ============================================================
# AUDIT ONE DATASET
# ============================================================

def audit_dataset(dataset_name: str, target: str) -> dict:
    """Perform the complete quality audit for one dataset."""

    path = PROCESSED_DIR / f"{dataset_name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path.resolve()}"
        )

    df = pd.read_csv(path)

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in {dataset_name}."
        )

    X = df.drop(columns=[target])
    y = df[target]

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())

    duplicate_rows = int(df.duplicated().sum())
    duplicate_rate = (duplicate_rows / len(df)) * 100

    missing_rate = (
        (missing_cells / total_cells) * 100
        if total_cells > 0
        else 0.0
    )

    class_info = calculate_class_imbalance(y)

    quality_report = {
        "dataset": dataset_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "features": int(X.shape[1]),
        "target": target,
        "numerical_features": int(
            X.select_dtypes(include=np.number).shape[1]
        ),
        "categorical_features": int(
            X.select_dtypes(exclude=np.number).shape[1]
        ),
        "missing_cells": missing_cells,
        "missing_rate_percent": round(missing_rate, 4),
        "duplicate_rows": duplicate_rows,
        "duplicate_rate_percent": round(duplicate_rate, 4),
        "outlier_row_rate_percent": round(
            calculate_iqr_outlier_rate(X), 4
        ),
        "potential_label_conflict_rate_percent": round(
            calculate_label_conflict_rate(X, y), 4
        ),
        "number_of_classes": int(y.nunique()),
        "class_counts": class_info["class_counts"],
        "class_percentages": class_info["class_percentages"],
        "imbalance_ratio": class_info["imbalance_ratio"],
    }

    return quality_report


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    all_reports = []

    print("=" * 70)
    print("DATA QUALITY AUDIT")
    print("=" * 70)

    for dataset_name, config in DATASETS.items():

        print(f"\nAuditing: {dataset_name}")

        try:
            report = audit_dataset(
                dataset_name,
                config["target"],
            )

            all_reports.append(report)

            print(f"Rows                  : {report['rows']}")
            print(f"Features              : {report['features']}")
            print(
                f"Missing rate          : "
                f"{report['missing_rate_percent']}%"
            )
            print(
                f"Duplicate rate        : "
                f"{report['duplicate_rate_percent']}%"
            )
            print(
                f"Outlier row rate      : "
                f"{report['outlier_row_rate_percent']}%"
            )
            print(
                f"Potential label issue : "
                f"{report['potential_label_conflict_rate_percent']}%"
            )
            print(
                f"Imbalance ratio       : "
                f"{report['imbalance_ratio']}"
            )

        except Exception as error:
            print(f"ERROR: {error}")

    # Save JSON report
    json_path = RESULTS_DIR / "data_quality_audit.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(all_reports, file, indent=4)

    # Save CSV summary
    summary_df = pd.DataFrame(all_reports)
    csv_path = RESULTS_DIR / "data_quality_summary.csv"

    # Class dictionaries make the summary hard to read in CSV.
    # They remain in the JSON report, while the CSV keeps the main metrics.
    summary_columns = [
        "dataset",
        "rows",
        "columns",
        "features",
        "target",
        "numerical_features",
        "categorical_features",
        "missing_cells",
        "missing_rate_percent",
        "duplicate_rows",
        "duplicate_rate_percent",
        "outlier_row_rate_percent",
        "potential_label_conflict_rate_percent",
        "number_of_classes",
        "imbalance_ratio",
    ]

    summary_df[summary_columns].to_csv(
        csv_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    print(f"JSON report : {json_path}")
    print(f"CSV summary : {csv_path}")


if __name__ == "__main__":
    main()
