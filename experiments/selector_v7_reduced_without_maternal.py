
from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# V7-REDUCED
# Census-augmented selector WITHOUT Maternal Health
# =========================================================
#
# Main held-out evaluation datasets:
#   Car Evaluation
#   Mobile Price
#   Bank Marketing
#
# Source/calibration datasets:
#   The same 3 evaluation datasets
#   + Census KDD
#
# Maternal Health is completely excluded:
#   - not a source dataset
#   - not a held-out evaluation dataset
#
# Primary metric:
#   condition-level model-selection accuracy
#
# Secondary:
#   mean regret
#
# This is a new evaluation scope and must be reported
# explicitly as such. It is NOT directly comparable to
# the 4-dataset 14/17 headline without stating the
# denominator change.
# =========================================================


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables"
OUT_DIR = RESULTS / "robustness_consensus_v7_reduced"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------

SINGLE_FILE = RESULTS / "multi_seed_single_results.csv"
COMBINED_FILE = RESULTS / "multi_seed_combined_results.csv"
BASELINE_FILE = RESULTS / "baseline_results.csv"

CENSUS_SINGLE_FILE = (
    RESULTS
    / "census_source_experiments"
    / "census_source_single_results.csv"
)

CENSUS_COMBINED_FILE = (
    RESULTS
    / "census_source_experiments"
    / "census_source_combined_results.csv"
)

CENSUS_BASELINE_FILE = (
    RESULTS
    / "census_source_experiments"
    / "census_source_clean_baseline.csv"
)


# ---------------------------------------------------------
# EVALUATION / SOURCE DATASETS
# ---------------------------------------------------------

EVAL_DATASETS = [
    "car_evaluation",
    "mobile_price",
    "bank_marketing",
]

SOURCE_DATASETS = EVAL_DATASETS + [
    "census_income_kdd"
]


MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "XGBoost",
]


# ---------------------------------------------------------
# CONDITIONS
# ---------------------------------------------------------

COMPONENTS = {
    "missing_noise": [
        ("missing", 0.10),
        ("label_noise", 0.10),
    ],

    "missing_noise_duplicates": [
        ("missing", 0.10),
        ("label_noise", 0.10),
        ("duplicates", 0.10),
    ],

    "missing_imbalance": [
        ("missing", 0.10),
        ("imbalance", 5.0),
    ],

    "noise_imbalance": [
        ("label_noise", 0.10),
        ("imbalance", 5.0),
    ],

    "missing_noise_imbalance": [
        ("missing", 0.10),
        ("label_noise", 0.10),
        ("imbalance", 5.0),
    ],

    "missing_noise_outliers": [
        ("missing", 0.10),
        ("label_noise", 0.10),
        ("outliers", 0.10),
    ],
}


FORMULAS = [
    "multiplicative",
    "geometric_mean",
    "arithmetic_mean",
    "harmonic_mean",
    "minimum",
    "additive_drop",
    "mean_f1",
]


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

def load_data():

    required = [
        SINGLE_FILE,
        COMBINED_FILE,
        BASELINE_FILE,
        CENSUS_SINGLE_FILE,
        CENSUS_COMBINED_FILE,
        CENSUS_BASELINE_FILE,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    single = pd.read_csv(SINGLE_FILE)
    combined = pd.read_csv(COMBINED_FILE)
    baseline = pd.read_csv(BASELINE_FILE)

    census_single = pd.read_csv(
        CENSUS_SINGLE_FILE
    )
    census_combined = pd.read_csv(
        CENSUS_COMBINED_FILE
    )
    census_baseline = pd.read_csv(
        CENSUS_BASELINE_FILE
    )

    # Remove Maternal Health completely.
    single = single[
        single["dataset"].isin(EVAL_DATASETS)
    ].copy()

    combined = combined[
        combined["dataset"].isin(EVAL_DATASETS)
    ].copy()

    baseline = baseline[
        baseline["dataset"].isin(EVAL_DATASETS)
    ].copy()

    # Census source-only.
    single = pd.concat(
        [single, census_single],
        ignore_index=True,
    )

    combined = pd.concat(
        [combined, census_combined],
        ignore_index=True,
    )

    baseline = pd.concat(
        [baseline, census_baseline],
        ignore_index=True,
    )

    return single, combined, baseline


# ---------------------------------------------------------
# LOOKUPS
# ---------------------------------------------------------

def build_clean_lookup(baseline):

    return {
        (
            row.dataset,
            row.model,
        ): float(row.macro_f1)
        for row in baseline.itertuples(
            index=False
        )
    }


def build_single_lookup(single):

    return {
        (
            row.dataset,
            int(row.seed),
            row.model,
            row.corruption,
            float(row.severity),
        ): float(row.macro_f1)
        for row in single.itertuples(
            index=False
        )
    }


# ---------------------------------------------------------
# FORMULAS
# ---------------------------------------------------------

def predict_formula(
    clean_f1,
    component_f1s,
    formula,
):

    values = np.asarray(
        component_f1s,
        dtype=float,
    )

    values = np.maximum(
        values,
        0.0,
    )

    if not len(values):
        return float(clean_f1)

    if formula == "multiplicative":

        if clean_f1 <= 0:
            return 0.0

        retention = np.clip(
            values / clean_f1,
            0.0,
            1.0,
        )

        return float(
            clean_f1 * np.prod(retention)
        )

    if formula == "geometric_mean":

        return float(
            np.exp(
                np.mean(
                    np.log(
                        np.maximum(
                            values,
                            1e-12,
                        )
                    )
                )
            )
        )

    if formula in (
        "arithmetic_mean",
        "mean_f1",
    ):
        return float(
            np.mean(values)
        )

    if formula == "harmonic_mean":

        return float(
            len(values)
            / np.sum(
                1.0
                / np.maximum(
                    values,
                    1e-12,
                )
            )
        )

    if formula == "minimum":

        return float(
            np.min(values)
        )

    if formula == "additive_drop":

        drops = np.maximum(
            clean_f1 - values,
            0.0,
        )

        return float(
            max(
                0.0,
                clean_f1 - np.sum(drops),
            )
        )

    raise ValueError(
        f"Unknown formula: {formula}"
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

def predict_model(
    dataset,
    seed,
    model,
    condition,
    formula,
    weight,
    clean_lookup,
    single_lookup,
):

    clean_f1 = clean_lookup[
        (dataset, model)
    ]

    components = []

    for corruption, severity in COMPONENTS[
        condition
    ]:

        value = single_lookup.get(
            (
                dataset,
                int(seed),
                model,
                corruption,
                float(severity),
            ),
            clean_f1,
        )

        components.append(
            float(value)
        )

    aggregate = predict_formula(
        clean_f1,
        components,
        formula,
    )

    return float(
        weight * aggregate
        + (1.0 - weight) * clean_f1
    )


# ---------------------------------------------------------
# EVALUATE DATASET
# ---------------------------------------------------------

def evaluate_dataset(
    dataset,
    combined,
    clean_lookup,
    single_lookup,
    formula,
    weight,
):

    rows = []

    target = combined[
        combined["dataset"] == dataset
    ]

    if target.empty:
        return pd.DataFrame()

    for condition in sorted(
        target["condition"].unique()
    ):

        condition_df = target[
            target["condition"]
            == condition
        ]

        predictions = {}
        actuals = {}

        for model in MODELS:

            pred_values = []
            actual_values = []

            for seed in sorted(
                condition_df["seed"].unique()
            ):

                pred_values.append(
                    predict_model(
                        dataset,
                        int(seed),
                        model,
                        condition,
                        formula,
                        weight,
                        clean_lookup,
                        single_lookup,
                    )
                )

                row = condition_df[
                    (
                        condition_df["seed"]
                        == seed
                    )
                    &
                    (
                        condition_df["model"]
                        == model
                    )
                ]

                actual_values.append(
                    float(
                        row.iloc[0]["macro_f1"]
                    )
                )

            predictions[model] = float(
                np.mean(pred_values)
            )

            actuals[model] = float(
                np.mean(actual_values)
            )

        selected = max(
            predictions,
            key=predictions.get,
        )

        oracle = max(
            actuals,
            key=actuals.get,
        )

        rows.append({
            "dataset": dataset,
            "condition": condition,
            "formula": formula,
            "weight": weight,
            "selected_model": selected,
            "oracle_model": oracle,
            "selected_predicted_mean_f1":
                predictions[selected],
            "selected_actual_mean_f1":
                actuals[selected],
            "oracle_mean_f1":
                actuals[oracle],
            "regret":
                actuals[oracle]
                - actuals[selected],
            "correct_selection":
                selected == oracle,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# SOURCE-BASED STRATEGY SELECTION
# ---------------------------------------------------------

def choose_strategy(
    held_out,
    combined,
    clean_lookup,
    single_lookup,
):

    sources = [
        d
        for d in SOURCE_DATASETS
        if d != held_out
    ]

    candidates = []

    for formula in FORMULAS:

        for weight in np.arange(
            0.0,
            1.01,
            0.05,
        ):

            source_results = []

            for source in sources:

                result = evaluate_dataset(
                    dataset=source,
                    combined=combined,
                    clean_lookup=clean_lookup,
                    single_lookup=single_lookup,
                    formula=formula,
                    weight=float(weight),
                )

                if not result.empty:
                    source_results.append(
                        result
                    )

            if not source_results:
                continue

            source_df = pd.concat(
                source_results,
                ignore_index=True,
            )

            candidates.append({
                "formula": formula,
                "weight": float(weight),
                "source_cases": len(
                    source_df
                ),
                "source_correct": int(
                    source_df[
                        "correct_selection"
                    ].sum()
                ),
                "source_accuracy":
                    float(
                        source_df[
                            "correct_selection"
                        ].mean()
                    ),
                "source_mean_regret":
                    float(
                        source_df["regret"].mean()
                    ),
            })

    candidate_df = pd.DataFrame(
        candidates
    )

    candidate_df = candidate_df.sort_values(
        [
            "source_accuracy",
            "source_mean_regret",
            "weight",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    best = candidate_df.iloc[0]

    return (
        str(best["formula"]),
        float(best["weight"]),
        candidate_df,
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "V7-REDUCED: CENSUS-AUGMENTED SELECTOR"
    )
    print(
        "Maternal Health EXCLUDED"
    )
    print("=" * 70)

    single, combined, baseline = load_data()

    clean_lookup = build_clean_lookup(
        baseline
    )

    single_lookup = build_single_lookup(
        single
    )

    print(
        f"\nEvaluation datasets: "
        f"{EVAL_DATASETS}"
    )

    print(
        f"Source datasets: "
        f"{SOURCE_DATASETS}"
    )

    print(
        f"Single records: "
        f"{len(single)}"
    )

    print(
        f"Combined records: "
        f"{len(combined)}"
    )

    all_results = []
    strategy_rows = []

    for held_out in EVAL_DATASETS:

        formula, weight, candidates = (
            choose_strategy(
                held_out=held_out,
                combined=combined,
                clean_lookup=clean_lookup,
                single_lookup=single_lookup,
            )
        )

        result = evaluate_dataset(
            dataset=held_out,
            combined=combined,
            clean_lookup=clean_lookup,
            single_lookup=single_lookup,
            formula=formula,
            weight=weight,
        )

        all_results.append(result)

        strategy_rows.append({
            "held_out_dataset":
                held_out,
            "selected_formula":
                formula,
            "selected_weight":
                weight,
            "source_selection_accuracy":
                float(
                    candidates.iloc[0][
                        "source_accuracy"
                    ]
                ),
            "source_mean_regret":
                float(
                    candidates.iloc[0][
                        "source_mean_regret"
                    ]
                ),
        })

        print("\n" + "-" * 70)
        print(
            f"Held-out dataset: "
            f"{held_out}"
        )
        print(
            f"Formula: {formula}"
        )
        print(
            f"Weight: {weight:.2f}"
        )
        print(
            f"Selection accuracy: "
            f"{result['correct_selection'].mean():.4%}"
        )
        print(
            f"Mean regret: "
            f"{result['regret'].mean():.6f}"
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    strategies = pd.DataFrame(
        strategy_rows
    )

    correct = int(
        results[
            "correct_selection"
        ].sum()
    )

    cases = len(results)

    summary = pd.DataFrame([{
        "version":
            "V7_reduced_without_maternal",
        "datasets":
            len(EVAL_DATASETS),
        "cases":
            cases,
        "correct":
            correct,
        "selection_accuracy":
            correct / cases,
        "mean_regret":
            float(
                results["regret"].mean()
            ),
    }])

    results.to_csv(
        OUT_DIR / "results.csv",
        index=False,
    )

    strategies.to_csv(
        OUT_DIR / "strategies.csv",
        index=False,
    )

    summary.to_csv(
        OUT_DIR / "summary.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print("\nStrategies:")
    print(
        strategies.to_string(
            index=False
        )
    )

    print("\nCondition-level results:")
    print(
        results[
            [
                "dataset",
                "condition",
                "selected_model",
                "oracle_model",
                "correct_selection",
                "regret",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(OUT_DIR / "summary.csv")
    print(OUT_DIR / "results.csv")
    print(OUT_DIR / "strategies.csv")

    print("\n" + "=" * 70)
    print("V7-REDUCED COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
