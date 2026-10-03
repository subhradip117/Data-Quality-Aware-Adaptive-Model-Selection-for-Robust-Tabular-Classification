# Data-Quality-Aware Adaptive Model Selection for Robust Tabular Classification

A Python-based machine learning project that studies how different classifiers behave when tabular data contains quality problems such as missing values, duplicates, label noise, outliers, and class imbalance.

The main idea is simple: **instead of always using one fixed ML model, use measured robustness to data corruption to select a more suitable model for the current corruption profile.**

## Project Overview

The project evaluates five classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- XGBoost

The workflow is:

```text
Dataset
   ↓
Data Quality Audit
   ↓
Single-Corruption Experiments
   ↓
Model Robustness Profile
   ↓
Adaptive Model Selector
   ↓
Combined-Corruption Evaluation
   ↓
Best Model Selection
```

## Main Idea

For each dataset, we first measure how candidate models perform when individual data problems are introduced.

The selector then uses those robustness measurements to estimate which model is most suitable when multiple corruption types occur together.

The main evaluation uses:

- **Car Evaluation**
- **Mobile Price**
- **Bank Marketing**

**Census KDD** is used as an additional source/calibration dataset.

## Data Corruptions Studied

The project includes controlled experiments involving:

- Missing values
- Duplicate records
- Label noise
- Outliers
- Class imbalance

It also evaluates combinations such as:

- Missing + label noise
- Missing + label noise + duplicates
- Missing + imbalance
- Label noise + imbalance
- Missing + label noise + imbalance
- Missing + label noise + outliers

## Main Result

On the primary three-dataset benchmark:

**10 out of 11 conditions were selected correctly.**

### Model-selection accuracy

**90.91% (10/11)**

Important: this is **model-selection accuracy**, not normal classification accuracy. It measures how often the selector correctly identifies the best-performing classifier for a tested corruption condition.

The primary benchmark uses three evaluation datasets, while Census KDD is used only as an additional source/calibration dataset and is not counted as a held-out evaluation dataset.

## Additional Evaluation

An earlier broader evaluation including the Maternal Health dataset produced:

**14/17 = 82.35% model-selection accuracy**

Maternal Health is therefore retained as an additional evaluation/sensitivity result rather than part of the primary V7 benchmark.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib

## Repository Structure

```text
data_quality_ml/
│
├── data/
│   └── README.md
│
├── src/
│   ├── data/
│   ├── quality/
│   ├── corruption/
│   ├── models/
│   ├── selector/
│   └── pipeline/
│
├── experiments/
│   ├── baseline_experiment.py
│   ├── corruption_experiment.py
│   ├── combined_experiment.py
│   ├── multi_seed_validation.py
│   ├── census_source_experiments.py
│   └── selector_v7_reduced_without_maternal.py
│
├── results/
│   ├── tables/
│   └── figures/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

Create and activate a virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Run the main selector experiment:

```bash
python -m experiments.selector_v7_reduced_without_maternal
```

## Reproducibility

The experiments use multiple random seeds for corruption generation and evaluation.

The repository should contain the code and final results needed to reproduce the reported experiments. Dataset files may need to be obtained separately depending on their source and redistribution terms.

## Research Note

This repository contains the final primary experiment as well as supporting experiments used during development. Earlier exploratory experiments are retained separately as research history.

## Authors

Add your team members here.

## License

Add the license you want to use for this project.
