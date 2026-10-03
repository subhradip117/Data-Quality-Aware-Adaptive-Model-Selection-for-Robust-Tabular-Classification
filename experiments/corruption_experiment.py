from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = ROOT / "data" / "processed" / "splits"
RESULTS_DIR = ROOT / "results" / "tables"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# DATASETS
# =========================================================

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


# =========================================================
# MODELS
# =========================================================

def get_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "SVM": SVC(
            kernel="rbf",
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss"
        )
    }


# =========================================================
# PREPROCESSING
# =========================================================

def create_preprocessor(X):

    numeric_columns = X.select_dtypes(
        include=[
            "int64",
            "int32",
            "float64",
            "float32"
        ]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "string",
            "category",
            "bool"
        ]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    return ColumnTransformer(transformers)


# =========================================================
# CORRUPTION 1
# MISSING VALUES
# =========================================================

def add_missing_values(
    df,
    target,
    rate,
    seed=42
):

    result = df.copy()

    feature_columns = [
        c for c in result.columns
        if c != target
    ]

    total_cells = (
        len(result) * len(feature_columns)
    )

    number_missing = int(
        round(total_cells * rate)
    )

    rng = np.random.default_rng(seed)

    selected_cells = rng.choice(
        total_cells,
        size=number_missing,
        replace=False
    )

    for cell in selected_cells:

        row_index = cell // len(feature_columns)
        col_index = cell % len(feature_columns)

        column = feature_columns[col_index]

        result.iat[
            row_index,
            result.columns.get_loc(column)
        ] = np.nan

    return result


# =========================================================
# CORRUPTION 2
# DUPLICATE ROWS
# =========================================================

def add_duplicates(
    df,
    rate,
    seed=342
):

    result = df.copy()

    number_duplicates = int(
        round(len(result) * rate)
    )

    rng = np.random.default_rng(seed)

    selected_indices = rng.choice(
        len(result),
        size=number_duplicates,
        replace=False
    )

    duplicate_rows = result.iloc[
        selected_indices
    ].copy()

    result = pd.concat(
        [
            result,
            duplicate_rows
        ],
        ignore_index=True
    )

    return result


# =========================================================
# CORRUPTION 3
# LABEL NOISE
# =========================================================

def add_label_noise(
    df,
    target,
    rate,
    seed=142
):

    result = df.copy()

    number_changes = int(
        round(len(result) * rate)
    )

    rng = np.random.default_rng(seed)

    classes = result[target].dropna().unique()

    selected_indices = rng.choice(
        len(result),
        size=number_changes,
        replace=False
    )

    for index in selected_indices:

        current_label = result.at[
            index,
            target
        ]

        possible_labels = [
            label
            for label in classes
            if label != current_label
        ]

        new_label = rng.choice(
            possible_labels
        )

        result.at[
            index,
            target
        ] = new_label

    return result


# =========================================================
# CORRUPTION 4
# OUTLIERS
# =========================================================

def add_outliers(
    df,
    target,
    rate,
    seed=242
):

    result = df.copy()

    numeric_columns = result.drop(
        columns=[target]
    ).select_dtypes(
        include=[
            "int64",
            "int32",
            "float64",
            "float32"
        ]
    ).columns.tolist()

    # Remove binary columns because they are usually
    # categorical indicators rather than continuous variables.
    eligible_columns = []

    for column in numeric_columns:

        unique_values = result[column].dropna().nunique()

        if unique_values > 2:
            eligible_columns.append(column)

    if not eligible_columns:
        return result

    number_rows = int(
        round(len(result) * rate)
    )

    rng = np.random.default_rng(seed)

    selected_rows = rng.choice(
        len(result),
        size=number_rows,
        replace=False
    )

    for row_index in selected_rows:

        column = rng.choice(
            eligible_columns
        )

        series = result[column]

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0 or pd.isna(iqr):
            continue

        direction = rng.choice(
            [-1, 1]
        )

        original_value = series.iloc[row_index]

        if direction == 1:
            new_value = q3 + 3 * iqr
        else:
            new_value = q1 - 3 * iqr

        # Preserve integer dtype when necessary
        if pd.api.types.is_integer_dtype(
            series.dtype
        ):
            new_value = int(round(new_value))

        result.iat[
            row_index,
            result.columns.get_loc(column)
        ] = new_value

    return result


# =========================================================
# CORRUPTION 5
# CLASS IMBALANCE
# =========================================================

def add_imbalance(
    df,
    target,
    ratio,
    seed=442
):

    result = df.copy()

    class_counts = (
        result[target]
        .value_counts()
    )

    # Already extremely imbalanced or only one class
    if len(class_counts) < 2:
        return result

    majority_class = class_counts.idxmax()

    majority_count = class_counts.max()

    target_minority_count = int(
        majority_count / ratio
    )

    rng = np.random.default_rng(seed)

    parts = []

    for class_value in class_counts.index:

        class_rows = result[
            result[target] == class_value
        ]

        if class_value == majority_class:

            selected = class_rows

        else:

            keep_count = min(
                len(class_rows),
                target_minority_count
            )

            selected_indices = rng.choice(
                len(class_rows),
                size=keep_count,
                replace=False
            )

            selected = class_rows.iloc[
                selected_indices
            ]

        parts.append(selected)

    result = pd.concat(
        parts,
        ignore_index=True
    )

    # Shuffle final training set
    result = result.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    return result


# =========================================================
# TRAIN + TEST
# =========================================================

def train_evaluate(
    train_df,
    test_df,
    target,
    model_name,
    model
):

    X_train = train_df.drop(
        columns=[target]
    )

    y_train = train_df[target]

    X_test = test_df.drop(
        columns=[target]
    )

    y_test = test_df[target]

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(
        y_train
    )

    y_test_encoded = label_encoder.transform(
        y_test
    )

    preprocessor = create_preprocessor(
        X_train
    )

    pipeline = Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    start = time.time()

    pipeline.fit(
        X_train,
        y_train_encoded
    )

    training_time = time.time() - start

    predictions = pipeline.predict(
        X_test
    )

    return {
        "accuracy": accuracy_score(
            y_test_encoded,
            predictions
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_test_encoded,
            predictions
        ),

        "macro_f1": f1_score(
            y_test_encoded,
            predictions,
            average="macro"
        ),

        "training_time_seconds":
            training_time
    }


# =========================================================
# AVAILABLE CORRUPTIONS FOR EACH DATASET
# =========================================================

def get_conditions(dataset_name):

    conditions = [
        ("missing", 0.05),
        ("missing", 0.10),
        ("missing", 0.20),

        ("duplicates", 0.05),
        ("duplicates", 0.10),
        ("duplicates", 0.20),

        ("label_noise", 0.05),
        ("label_noise", 0.10),
        ("label_noise", 0.20),
    ]

    # Car Evaluation contains categorical features,
    # so numerical outlier corruption is not applicable.
    if dataset_name != "car_evaluation":

        conditions.extend([
            ("outliers", 0.05),
            ("outliers", 0.10),
            ("outliers", 0.20),
        ])

    # Car Evaluation already has strong natural imbalance.
    # We do not artificially downsample it.
    if dataset_name != "car_evaluation":

        conditions.extend([
            ("imbalance", 2),
            ("imbalance", 5),
            ("imbalance", 10),
        ])

    return conditions


# =========================================================
# APPLY CORRUPTION
# =========================================================

def corrupt_training_data(
    train_df,
    target,
    corruption,
    severity
):

    if corruption == "missing":

        return add_missing_values(
            train_df,
            target,
            severity,
            seed=42
        )

    elif corruption == "duplicates":

        return add_duplicates(
            train_df,
            severity,
            seed=342
        )

    elif corruption == "label_noise":

        return add_label_noise(
            train_df,
            target,
            severity,
            seed=142
        )

    elif corruption == "outliers":

        return add_outliers(
            train_df,
            target,
            severity,
            seed=242
        )

    elif corruption == "imbalance":

        return add_imbalance(
            train_df,
            target,
            severity,
            seed=442
        )

    return train_df.copy()


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SINGLE-CORRUPTION ROBUSTNESS EXPERIMENT")
    print("=" * 70)

    all_results = []

    for dataset_name, config in DATASETS.items():

        target = config["target"]

        train_file = (
            SPLIT_DIR /
            f"{dataset_name}_train.csv"
        )

        test_file = (
            SPLIT_DIR /
            f"{dataset_name}_test.csv"
        )

        train_df = pd.read_csv(
            train_file
        )

        test_df = pd.read_csv(
            test_file
        )

        print("\n" + "-" * 70)
        print(f"Dataset: {dataset_name}")
        print(f"Clean train rows: {len(train_df)}")
        print(f"Clean test rows:  {len(test_df)}")

        conditions = get_conditions(
            dataset_name
        )

        for corruption, severity in conditions:

            corrupted_train = corrupt_training_data(
                train_df,
                target,
                corruption,
                severity
            )

            if corruption == "imbalance":

                condition_name = (
                    f"imbalance_{severity}to1"
                )

            else:

                percentage = int(
                    severity * 100
                )

                condition_name = (
                    f"{corruption}_{percentage}"
                )

            print(
                f"\nCondition: {condition_name}"
            )

            print(
                f"Training rows after corruption: "
                f"{len(corrupted_train)}"
            )

            models = get_models()

            for model_name, model in models.items():

                print(
                    f"  Training {model_name}..."
                )

                try:

                    result = train_evaluate(
                        corrupted_train,
                        test_df,
                        target,
                        model_name,
                        model
                    )

                    result["dataset"] = (
                        dataset_name
                    )
                    result["model"] = model_name

                    result["corruption"] = (
                        corruption
                    )

                    result["severity"] = (
                        severity
                    )

                    result["condition"] = (
                        condition_name
                    )

                    result["train_rows"] = (
                        len(corrupted_train)
                    )

                    all_results.append(
                        result
                    )

                    print(
                        f"    Accuracy: "
                        f"{result['accuracy']:.4f} | "
                        f"Macro-F1: "
                        f"{result['macro_f1']:.4f}"
                    )

                except Exception as e:

                    print(
                        f"    ERROR: {e}"
                    )

    # =====================================================
    # SAVE
    # =====================================================

    results_df = pd.DataFrame(
        all_results
    )

    # Load clean baseline
    baseline_file = (
        RESULTS_DIR /
        "baseline_results.csv"
    )

    baseline_df = pd.read_csv(
        baseline_file
    )

    baseline_small = baseline_df[
        [
            "dataset",
            "model",
            "accuracy",
            "balanced_accuracy",
            "macro_f1"
        ]
    ].rename(
        columns={
            "accuracy": "clean_accuracy",
            "balanced_accuracy":
                "clean_balanced_accuracy",
            "macro_f1": "clean_macro_f1"
        }
    )

    results_df = results_df.merge(
        baseline_small,
        on=["dataset", "model"],
        how="left"
    )

    # =====================================================
    # PERFORMANCE DROP
    # =====================================================

    results_df["accuracy_drop"] = (
        results_df["clean_accuracy"]
        - results_df["accuracy"]
    )

    results_df["balanced_accuracy_drop"] = (
        results_df["clean_balanced_accuracy"]
        - results_df["balanced_accuracy"]
    )

    results_df["macro_f1_drop"] = (
        results_df["clean_macro_f1"]
        - results_df["macro_f1"]
    )

    output_file = (
        RESULTS_DIR /
        "corruption_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("CORRUPTION EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:\n{output_file}"
    )

    print(
        f"\nTotal experiments: "
        f"{len(results_df)}"
    )


if __name__ == "__main__":
    main()