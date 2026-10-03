from pathlib import Path
import time

import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
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
    precision_score,
    recall_score,
)


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_DIR = PROCESSED_DIR / "splits"
RESULTS_DIR = ROOT / "results" / "tables"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# DATASETS
# ---------------------------------------------------------

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
}


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

def create_preprocessor(X):

    numeric_columns = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
    include=["object", "string", "category", "bool"]
).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    return preprocessor


# ---------------------------------------------------------
# CREATE FIXED SPLIT
# ---------------------------------------------------------

def create_split(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    # Use feature values as groups.
    # This prevents identical feature rows from appearing
    # in both train and test.
    groups = X.astype(str).agg("||".join, axis=1)

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, test_df


# ---------------------------------------------------------
# TRAIN + EVALUATE
# ---------------------------------------------------------

def train_and_evaluate(train_df, test_df, target, model_name, model):

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Convert class names into numbers
    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    preprocessor = create_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    start_time = time.time()

    pipeline.fit(
        X_train,
        y_train_encoded
    )

    training_time = time.time() - start_time

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test_encoded,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test_encoded,
        predictions
    )

    macro_f1 = f1_score(
        y_test_encoded,
        predictions,
        average="macro"
    )

    macro_precision = precision_score(
        y_test_encoded,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test_encoded,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "training_time_seconds": training_time
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("CLEAN DATA BASELINE EXPERIMENT")
    print("=" * 70)

    all_results = []

    for dataset_name, config in DATASETS.items():

        print("\n" + "-" * 70)
        print(f"Dataset: {dataset_name}")

        df = pd.read_csv(config["file"])
        target = config["target"]

        print(f"Original rows: {len(df)}")
        print(f"Features: {len(df.columns) - 1}")

        # Remove accidental index columns if present
        unnamed_columns = [
            col for col in df.columns
            if col.lower().startswith("unnamed:")
        ]

        if unnamed_columns:
            df = df.drop(columns=unnamed_columns)

        # -------------------------------------------------
        # CREATE FIXED TRAIN / TEST SPLIT
        # -------------------------------------------------

        train_df, test_df = create_split(
            df,
            target
        )

        train_file = SPLIT_DIR / f"{dataset_name}_train.csv"
        test_file = SPLIT_DIR / f"{dataset_name}_test.csv"

        train_df.to_csv(
            train_file,
            index=False
        )

        test_df.to_csv(
            test_file,
            index=False
        )

        print(f"Train rows: {len(train_df)}")
        print(f"Test rows:  {len(test_df)}")

        print(
            f"Saved: {train_file}"
        )

        print(
            f"Saved: {test_file}"
        )

        # -------------------------------------------------
        # TRAIN 5 BASELINE MODELS
        # -------------------------------------------------

        models = get_models()

        for model_name, model in models.items():

            print(f"\nTraining: {model_name}")

            try:

                result = train_and_evaluate(
                    train_df,
                    test_df,
                    target,
                    model_name,
                    model
                )

                result["dataset"] = dataset_name

                all_results.append(result)

                print(
                    f"Accuracy:          {result['accuracy']:.4f}"
                )

                print(
                    f"Balanced Accuracy: {result['balanced_accuracy']:.4f}"
                )

                print(
                    f"Macro F1:          {result['macro_f1']:.4f}"
                )

            except Exception as e:

                print(
                    f"ERROR in {model_name}: {e}"
                )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    results_df = pd.DataFrame(all_results)

    # Put dataset first
    columns = [
        "dataset",
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "training_time_seconds"
    ]

    results_df = results_df[columns]

    output_file = RESULTS_DIR / "baseline_results.csv"

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("BASELINE EXPERIMENT COMPLETE")
    print("=" * 70)

    print("\nResults:")
    print(results_df.to_string(index=False))

    print(
        f"\nResults saved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()