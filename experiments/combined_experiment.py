from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
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
        "target": "class"
    },

    "maternal_health": {
        "target": "RiskLevel"
    },

    "mobile_price": {
        "target": "price_range"
    },

    "bank_marketing": {
        "target": "y"
    }
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
# PREPROCESSOR
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

    transformers = []

    if numeric_columns:

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

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    if categorical_columns:

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

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    return ColumnTransformer(transformers)


# =========================================================
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
        len(result) *
        len(feature_columns)
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

        row_index = (
            cell //
            len(feature_columns)
        )

        col_index = (
            cell %
            len(feature_columns)
        )

        column = feature_columns[col_index]

        result.iat[
            row_index,
            result.columns.get_loc(column)
        ] = np.nan

    return result


# =========================================================
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

    classes = (
        result[target]
        .dropna()
        .unique()
    )

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
# OUTLIERS
# =========================================================

def add_outliers(
    df,
    target,
    rate,
    seed=242
):

    result = df.copy()

    numeric_columns = (
        result
        .drop(columns=[target])
        .select_dtypes(
            include=[
                "int64",
                "int32",
                "float64",
                "float32"
            ]
        )
        .columns
        .tolist()
    )

    eligible_columns = []

    for column in numeric_columns:

        unique_values = (
            result[column]
            .dropna()
            .nunique()
        )

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

        direction = rng.choice([-1, 1])

        if direction == 1:
            new_value = q3 + 3 * iqr
        else:
            new_value = q1 - 3 * iqr

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
# DUPLICATES
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
# IMBALANCE
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

    majority_class = (
        class_counts.idxmax()
    )

    majority_count = (
        class_counts.max()
    )

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

            indices = rng.choice(
                len(class_rows),
                size=keep_count,
                replace=False
            )

            selected = class_rows.iloc[
                indices
            ]

        parts.append(selected)

    result = pd.concat(
        parts,
        ignore_index=True
    )

    result = result.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    return result


# =========================================================
# DEFINE COMBINATIONS
# =========================================================

def get_combinations(dataset_name):

    combinations = [
        (
            "missing_noise",
            ["missing", "noise"]
        ),

        (
            "missing_imbalance",
            ["missing", "imbalance"]
        ),

        (
            "noise_imbalance",
            ["noise", "imbalance"]
        ),

        (
            "missing_noise_imbalance",
            [
                "missing",
                "noise",
                "imbalance"
            ]
        ),

        (
            "missing_noise_duplicates",
            [
                "missing",
                "noise",
                "duplicates"
            ]
        )
    ]

    # Outliers are not applicable to Car Evaluation
    if dataset_name != "car_evaluation":

        combinations.append(
            (
                "missing_noise_outliers",
                [
                    "missing",
                    "noise",
                    "outliers"
                ]
            )
        )

    # Artificial imbalance is not applied to datasets
    # that already have substantial natural imbalance.
    if dataset_name in [
        "car_evaluation",
        "bank_marketing"
    ]:

        combinations = [
            item
            for item in combinations
            if "imbalance" not in item[1]
        ]

    return combinations


# =========================================================
# APPLY A COMBINATION
# =========================================================

def apply_combination(
    train_df,
    target,
    combination
):

    result = train_df.copy()

    # -----------------------------------------------------
    # IMPORTANT:
    # imbalance first
    # then feature/label corruption
    # duplicates last
    # -----------------------------------------------------

    if "imbalance" in combination:

        result = add_imbalance(
            result,
            target,
            ratio=5,
            seed=442
        )

    if "missing" in combination:

        result = add_missing_values(
            result,
            target,
            rate=0.10,
            seed=42
        )

    if "noise" in combination:

        result = add_label_noise(
            result,
            target,
            rate=0.10,
            seed=142
        )

    if "outliers" in combination:

        result = add_outliers(
            result,
            target,
            rate=0.10,
            seed=242
        )

    if "duplicates" in combination:

        result = add_duplicates(
            result,
            rate=0.10,
            seed=342
        )

    return result


# =========================================================
# TRAIN + EVALUATE
# =========================================================

def train_evaluate(
    train_df,
    test_df,
    target,
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

    encoder = LabelEncoder()

    y_train_encoded = (
        encoder.fit_transform(y_train)
    )

    y_test_encoded = (
        encoder.transform(y_test)
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

    training_time = (
        time.time() - start
    )

    predictions = pipeline.predict(
        X_test
    )

    return {
        "accuracy": accuracy_score(
            y_test_encoded,
            predictions
        ),

        "balanced_accuracy":
            balanced_accuracy_score(
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
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("COMBINED-CORRUPTION ROBUSTNESS EXPERIMENT")
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
        print(
            f"Dataset: {dataset_name}"
        )

        combinations = get_combinations(
            dataset_name
        )

        for combination_name, combination in combinations:

            print(
                f"\nCombination: "
                f"{combination_name}"
            )

            corrupted_train = apply_combination(
                train_df,
                target,
                combination
            )

            print(
                f"Training rows: "
                f"{len(corrupted_train)}"
            )

            models = get_models()

            for model_name, model in models.items():

                print(
                    f"  Training "
                    f"{model_name}..."
                )

                try:

                    result = train_evaluate(
                        corrupted_train,
                        test_df,
                        target,
                        model
                    )

                    result["dataset"] = (
                        dataset_name
                    )

                    result["model"] = (
                        model_name
                    )

                    result["condition"] = (
                        combination_name
                    )

                    result["corruption_combination"] = (
                        "+".join(combination)
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
    # SAVE RAW RESULTS
    # =====================================================

    results_df = pd.DataFrame(
        all_results
    )

    # =====================================================
    # LOAD CLEAN BASELINE
    # =====================================================

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
            "accuracy":
                "clean_accuracy",

            "balanced_accuracy":
                "clean_balanced_accuracy",

            "macro_f1":
                "clean_macro_f1"
        }
    )

    results_df = results_df.merge(
        baseline_small,
        on=[
            "dataset",
            "model"
        ],
        how="left"
    )

    # =====================================================
    # PERFORMANCE DROP
    # =====================================================

    results_df["accuracy_drop"] = (
        results_df["clean_accuracy"]
        - results_df["accuracy"]
    )

    results_df[
        "balanced_accuracy_drop"
    ] = (
        results_df["clean_balanced_accuracy"]
        - results_df["balanced_accuracy"]
    )

    results_df["macro_f1_drop"] = (
        results_df["clean_macro_f1"]
        - results_df["macro_f1"]
    )

    # =====================================================
    # SAVE
    # =====================================================

    output_file = (
        RESULTS_DIR /
        "combined_corruption_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print(
        "COMBINED EXPERIMENT COMPLETE"
    )
    print("=" * 70)

    print(
        f"\nTotal experiments: "
        f"{len(results_df)}"
    )

    print(
        f"\nResults saved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()