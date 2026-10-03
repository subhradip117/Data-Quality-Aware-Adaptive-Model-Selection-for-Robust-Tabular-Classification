from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MODEL_DIR = RESULTS_DIR / "model_outputs"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "SVM": SVC(kernel="rbf", random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        ),
    }


def create_preprocessor(X):
    numeric = X.select_dtypes(
        include=["int64", "int32", "int16", "float64", "float32", "float16"]
    ).columns.tolist()

    categorical = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns found.")

    return ColumnTransformer(transformers=transformers)


def build_pipeline(X, model):
    return Pipeline(
        [
            ("preprocessing", create_preprocessor(X)),
            ("model", model),
        ]
    )


def audit_dataset(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    n_rows = len(df)
    n_features = X.shape[1]

    missing_cells = int(X.isna().sum().sum())
    total_cells = n_rows * n_features
    missing_rate = missing_cells / total_cells if total_cells else 0.0

    duplicate_rows = int(df.duplicated().sum())
    duplicate_rate = duplicate_rows / n_rows if n_rows else 0.0

    numeric_columns = X.select_dtypes(
        include=["int64", "int32", "int16", "float64", "float32", "float16"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    outlier_mask = pd.Series(False, index=df.index)

    for col in numeric_columns:
        s = pd.to_numeric(X[col], errors="coerce")
        if s.notna().sum() < 4:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask |= (s < lower) | (s > upper)

    outlier_rows = int(outlier_mask.sum())
    outlier_rate = outlier_rows / n_rows if n_rows else 0.0

    class_counts = y.value_counts(dropna=False)
    n_classes = int(y.nunique(dropna=False))

    if len(class_counts) > 1:
        imbalance_ratio = float(class_counts.max() / class_counts.min())
    else:
        imbalance_ratio = 1.0

    p = class_counts / class_counts.sum()
    entropy = float(-np.sum(p * np.log2(p)))

    conflict_rate = 0.0
    try:
        conflicts = (
            df.groupby(list(X.columns), dropna=False)[target]
            .transform("nunique")
        )
        conflict_rate = (
            float((conflicts > 1).sum()) / n_rows if n_rows else 0.0
        )
    except Exception:
        pass

    return {
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "n_numeric": int(len(numeric_columns)),
        "n_categorical": int(len(categorical_columns)),
        "numeric_ratio": float(len(numeric_columns) / n_features) if n_features else 0.0,
        "categorical_ratio": float(len(categorical_columns) / n_features) if n_features else 0.0,
        "n_classes": n_classes,
        "imbalance_ratio": imbalance_ratio,
        "class_entropy": entropy,
        "missing_cells": missing_cells,
        "missing_rate": float(missing_rate),
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": float(duplicate_rate),
        "outlier_rows": outlier_rows,
        "outlier_rate": float(outlier_rate),
        "potential_label_conflict_rate": conflict_rate,
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
    }


def detect_impurities(audit):
    impurities = []

    if audit["missing_rate"] > 0:
        impurities.append(("missing", float(min(0.20, max(0.01, audit["missing_rate"])))))

    if audit["duplicate_rate"] > 0:
        impurities.append(("duplicates", float(min(0.20, max(0.01, audit["duplicate_rate"])))))

    if audit["outlier_rate"] > 0 and audit["n_numeric"] > 0:
        impurities.append(("outliers", float(min(0.20, max(0.01, audit["outlier_rate"])))))

    if audit["potential_label_conflict_rate"] > 0:
        impurities.append(
            (
                "label_noise",
                float(min(0.20, max(0.01, audit["potential_label_conflict_rate"]))),
            )
        )

    # Natural imbalance is treated as a condition only for ratios up to 10:1.
    if 1.5 < audit["imbalance_ratio"] <= 10:
        rounded = int(round(audit["imbalance_ratio"]))
        if rounded >= 2:
            impurities.append(("imbalance", float(min(10, max(2, rounded)))))

    return impurities


def corrupt_missing(X, rate, seed):
    out = X.copy()
    cols = out.columns.tolist()
    if not cols or rate <= 0:
        return out

    total = len(out) * len(cols)
    count = min(total, int(round(total * rate)))
    if count <= 0:
        return out

    rng = np.random.default_rng(seed)
    chosen = rng.choice(total, size=count, replace=False)

    for pos in chosen:
        r = pos // len(cols)
        c = pos % len(cols)
        out.iat[r, out.columns.get_loc(cols[c])] = np.nan

    return out


def corrupt_duplicates(X, y, rate, seed):
    if rate <= 0:
        return X.reset_index(drop=True), y.reset_index(drop=True)

    n = min(len(X), int(round(len(X) * rate)))
    if n <= 0:
        return X.reset_index(drop=True), y.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)

    x2 = pd.concat([X.reset_index(drop=True), X.iloc[idx]], ignore_index=True)
    y2 = pd.concat([y.reset_index(drop=True), y.iloc[idx]], ignore_index=True)
    return x2, y2


def corrupt_label_noise(y, rate, seed):
    out = y.reset_index(drop=True).copy()
    n = min(len(out), int(round(len(out) * rate)))
    classes = out.dropna().unique().tolist()

    if n <= 0 or len(classes) < 2:
        return out

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(out), size=n, replace=False)

    for i in idx:
        current = out.iloc[i]
        choices = [c for c in classes if c != current]
        if choices:
            out.iloc[i] = rng.choice(choices)

    return out


def corrupt_outliers(X, rate, seed):
    out = X.copy()
    numeric = out.select_dtypes(
        include=["int64", "int32", "int16", "float64", "float32", "float16"]
    ).columns.tolist()

    numeric = [c for c in numeric if out[c].nunique(dropna=True) > 2]

    n = min(len(out), int(round(len(out) * rate)))
    if not numeric or n <= 0:
        return out

    rng = np.random.default_rng(seed)

    for r in rng.choice(len(out), size=n, replace=False):
        c = rng.choice(numeric)
        s = out[c]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        value = q3 + 3 * iqr if rng.integers(0, 2) else q1 - 3 * iqr

        if pd.api.types.is_integer_dtype(s.dtype):
            value = int(round(value))

        out.iat[r, out.columns.get_loc(c)] = value

    return out


def corrupt_imbalance(X, y, ratio, seed):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    counts = y.value_counts()
    if len(counts) < 2 or ratio <= 1:
        return X, y

    majority = counts.idxmax()
    majority_count = int(counts.max())
    target_minority = max(1, int(majority_count / ratio))

    rng = np.random.default_rng(seed)
    keep = []

    for cls in counts.index:
        idx = np.flatnonzero(y.values == cls)

        if cls == majority:
            chosen = idx
        else:
            chosen = rng.choice(
                idx,
                size=min(len(idx), target_minority),
                replace=False,
            )

        keep.extend(chosen.tolist())

    rng.shuffle(keep)

    return (
        X.iloc[keep].reset_index(drop=True),
        y.iloc[keep].reset_index(drop=True),
    )


def apply_corruption(X, y, corruption, severity, seed):
    if corruption == "missing":
        return corrupt_missing(X, severity, seed), y.reset_index(drop=True)

    if corruption == "duplicates":
        return corrupt_duplicates(X, y, severity, seed)

    if corruption == "label_noise":
        return X.reset_index(drop=True), corrupt_label_noise(y, severity, seed)

    if corruption == "outliers":
        return corrupt_outliers(X, severity, seed), y.reset_index(drop=True)

    if corruption == "imbalance":
        return corrupt_imbalance(X, y, severity, seed)

    return X.reset_index(drop=True), y.reset_index(drop=True)


def cv_score(X, y, model, corruption=None, severity=None, seed=42, folds=3):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    min_class = int(y.value_counts().min())
    n_splits = min(folds, min_class)

    if n_splits < 2:
        raise ValueError("Not enough samples per class for stratified cross-validation.")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    scores = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y), start=1
    ):
        Xtr = X.iloc[train_idx].reset_index(drop=True)
        ytr = y.iloc[train_idx].reset_index(drop=True)
        Xv = X.iloc[val_idx].reset_index(drop=True)
        yv = y.iloc[val_idx].reset_index(drop=True)

        if corruption is not None:
            Xtr, ytr = apply_corruption(
                Xtr,
                ytr,
                corruption,
                severity,
                seed + fold,
            )

        pipe = build_pipeline(Xtr, model)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xv)

        scores.append(
            f1_score(
                yv,
                pred,
                average="macro",
            )
        )

    return float(np.mean(scores))


def select_model(X_train, y_train, impurities, seed=42):
    rows = []

    for model_name, model in get_models().items():

        clean_f1 = cv_score(
            X_train,
            y_train,
            model,
            seed=seed,
            folds=3,
        )

        estimated = clean_f1

        row = {
            "model": model_name,
            "clean_cv_macro_f1": clean_f1,
        }

        for corruption, severity in impurities:
            corrupted_f1 = cv_score(
                X_train,
                y_train,
                model,
                corruption=corruption,
                severity=severity,
                seed=seed + 100,
                folds=3,
            )

            relative_drop = 0.0

            if clean_f1 > 0:
                relative_drop = max(
                    0.0,
                    (clean_f1 - corrupted_f1) / clean_f1,
                )

            relative_drop = min(1.0, relative_drop)
            estimated *= (1.0 - relative_drop)

            row[f"{corruption}_probe_f1"] = corrupted_f1
            row[f"{corruption}_relative_drop"] = relative_drop

        row["estimated_combined_macro_f1"] = estimated
        rows.append(row)

    profile = pd.DataFrame(rows)
    selected = profile.loc[
        profile["estimated_combined_macro_f1"].idxmax()
    ]

    return selected["model"], profile


def run_pipeline(
    df,
    target,
    prediction_df=None,
    dataset_name="custom_dataset",
    test_size=0.20,
    seed=42,
):
    print("=" * 70)
    print("DATA QUALITY AWARE ADAPTIVE MODEL SELECTION")
    print("=" * 70)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    df = df.copy()

    unnamed = [
        c for c in df.columns
        if c.lower().startswith("unnamed:")
    ]
    if unnamed:
        df = df.drop(columns=unnamed)

    empty_cols = [
        c for c in df.columns
        if c != target and df[c].isna().all()
    ]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    df = df[df[target].notna()].reset_index(drop=True)

    if df[target].nunique() < 2:
        raise ValueError("The target must contain at least 2 classes.")

    print("\n1. DATA QUALITY AUDIT")
    audit = audit_dataset(df, target)

    for key in [
        "n_rows",
        "n_features",
        "n_numeric",
        "n_categorical",
        "n_classes",
        "missing_cells",
        "duplicate_rows",
        "outlier_rows",
    ]:
        print(f"{key}: {audit[key]}")

    print(f"missing_rate: {audit['missing_rate']:.4%}")
    print(f"duplicate_rate: {audit['duplicate_rate']:.4%}")
    print(f"outlier_rate: {audit['outlier_rate']:.4%}")
    print(f"imbalance_ratio: {audit['imbalance_ratio']:.4f}")
    print(
        "potential_label_conflict_rate: "
        f"{audit['potential_label_conflict_rate']:.4%}"
    )

    impurities = detect_impurities(audit)

    print("\n2. DETECTED IMPURITY PROFILE")
    if impurities:
        for corruption, severity in impurities:
            print(f"{corruption}: severity={severity:.4f}")
    else:
        print("No measurable impurity detected.")

    X = df.drop(columns=[target])
    y_raw = df[target]

    label_encoder = LabelEncoder()
    y = pd.Series(
        label_encoder.fit_transform(y_raw),
        name=target,
    )

    from sklearn.model_selection import train_test_split

    min_class = int(y.value_counts().min())
    if min_class < 2:
        raise ValueError("Every class needs at least 2 samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("\n3. MODEL CALIBRATION")
    start = time.time()

    selected_model_name, profile = select_model(
        X_train,
        y_train,
        impurities,
        seed=seed,
    )

    calibration_time = time.time() - start

    print("\n" + profile[
        ["model", "clean_cv_macro_f1", "estimated_combined_macro_f1"]
    ].to_string(index=False))

    print(f"\nSelected model: {selected_model_name}")
    print(f"Calibration time: {calibration_time:.2f}s")

    print("\n4. FINAL MODEL TRAINING")

    final_model = get_models()[selected_model_name]
    final_pipeline = build_pipeline(
        X_train,
        final_model,
    )

    start = time.time()
    final_pipeline.fit(X_train, y_train)
    training_time = time.time() - start

    predictions = final_pipeline.predict(X_test)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro")
        ),
        "macro_precision": float(
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }

    print("\n5. FINAL CLEAN TEST RESULTS")
    print("-" * 60)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print(f"training_time_seconds: {training_time:.4f}")

    # -----------------------------------------------------
    # Prediction on separate unseen CSV
    # -----------------------------------------------------

    if prediction_df is not None:

        prediction_features = prediction_df.drop(
            columns=[target],
            errors="ignore",
        )

        prediction_numeric = final_pipeline.predict(
            prediction_features
        )

        prediction_labels = label_encoder.inverse_transform(
            prediction_numeric.astype(int)
        )

        prediction_output = prediction_df.copy()
        prediction_output["prediction"] = prediction_labels

        prediction_path = (
            MODEL_DIR /
            f"{dataset_name}_predictions.csv"
        )

        prediction_output.to_csv(
            prediction_path,
            index=False,
        )

        print(
            f"\nPredictions saved to:\n{prediction_path}"
        )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model_path = (
        MODEL_DIR /
        f"{dataset_name}_selected_model.joblib"
    )

    joblib.dump(
        {
            "pipeline": final_pipeline,
            "label_encoder": label_encoder,
            "target": target,
            "selected_model": selected_model_name,
        },
        model_path,
    )

    # -----------------------------------------------------
    # Save profiles
    # -----------------------------------------------------

    profile_path = (
        TABLES_DIR /
        f"{dataset_name}_final_model_profile.csv"
    )

    profile.to_csv(
        profile_path,
        index=False,
    )

    audit_path = (
        TABLES_DIR /
        f"{dataset_name}_final_quality_audit.json"
    )

    with open(
        audit_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                **audit,
                "dataset": dataset_name,
                "target": target,
                "detected_impurities": [
                    {
                        "corruption": c,
                        "severity": s,
                    }
                    for c, s in impurities
                ],
            },
            f,
            indent=4,
        )

    report_path = (
        MODEL_DIR /
        f"{dataset_name}_final_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "dataset": dataset_name,
                "target": target,
                "selected_model": selected_model_name,
                "estimated_combined_macro_f1": float(
                    profile.loc[
                        profile["model"] == selected_model_name,
                        "estimated_combined_macro_f1",
                    ].iloc[0]
                ),
                "final_test_metrics": metrics,
                "calibration_time_seconds": calibration_time,
                "training_time_seconds": training_time,
                "quality_audit": audit,
                "detected_impurities": [
                    {
                        "corruption": c,
                        "severity": s,
                    }
                    for c, s in impurities
                ],
            },
            f,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("FINAL PIPELINE COMPLETE")
    print("=" * 70)

    print(f"\nSelected model: {selected_model_name}")
    print(f"Final Macro-F1: {metrics['macro_f1']:.4f}")

    print("\nFiles created:")
    print(model_path)
    print(profile_path)
    print(audit_path)
    print(report_path)

    return {
        "selected_model": selected_model_name,
        "metrics": metrics,
        "audit": audit,
        "profile": profile,
        "model_path": model_path,
        "report_path": report_path,
    }
