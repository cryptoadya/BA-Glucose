# Short-Term Blood Glucose Prediction Benchmark

Research code for a bachelor thesis on short-term blood glucose prediction using traditional machine learning and compact deep learning models.

This repository contains the implementation of a reproducible benchmark on the HUPA-UCM Diabetes Dataset. The goal of the project is not to introduce a new neural network architecture, but to compare representative model families under a consistent, leakage-aware experimental protocol.

## Project overview

This project evaluates baseline models, traditional machine learning models, and deep learning models for forecasting future blood glucose values from multimodal patient time series.

The benchmark focuses on:

* short-term blood glucose forecasting
* prediction horizons of 30 and 60 minutes
* 120 minutes as an additional robustness check
* patient-wise chronological train/validation/test splitting
* explicit protection against sliding-window overlap leakage
* train-only feature scaling
* numerical and clinical evaluation metrics

The project was developed as part of a bachelor thesis in Informatik in Kultur und Gesundheit.

## Repository note

This repository contains research code developed for a bachelor thesis. The structure follows the experimental workflow of the thesis rather than a production software architecture.

Scripts are organized around benchmark stages such as baseline experiments, traditional ML experiments, deep learning experiments, patient-level evaluation, prediction export, and figure generation.

The main goal of the repository is reproducibility and transparency of the benchmark protocol, not deployment readiness.

## Why this project matters

Short-term blood glucose prediction is a relevant healthcare time-series forecasting problem. Glucose values can change rapidly due to insulin, meals, physical activity, and other physiological factors.

In this setting, model comparison is sensitive to preprocessing choices, feature definitions, splitting strategy, and leakage control. Sliding-window time-series workflows can easily produce overly optimistic results if train, validation, and test windows overlap in their underlying raw time points.

This repository therefore emphasizes the benchmark protocol as much as the models themselves.

## Dataset

The benchmark uses the HUPA-UCM Diabetes Dataset, a multimodal real-world dataset from 25 people with type 1 diabetes mellitus.

Dataset source:

* Dataset: **HUPA-UCM Diabetes Dataset**
* Dataset DOI: `10.17632/3hbcscwz44.1`
* Data article: Hidalgo et al., "HUPA-UCM diabetes dataset", Data in Brief 55 (2024), 110559
* Article DOI: `10.1016/j.dib.2024.110559`

The dataset contains patient-wise CSV files with signals such as:

* continuous glucose monitoring data: `glucose`
* basal insulin: `basal_rate`
* bolus insulin: `bolus_volume_delivered`
* carbohydrate intake: `carb_input`
* heart rate: `heart_rate`
* steps: `steps`
* calories: `calories`
* sleep-related variables in the original dataset

Dataset files are included in this repository.

## Task formulation

The task is formulated as supervised time-series regression.

Input:

* sliding window over the previous 60 minutes
* 5-minute sampling interval
* `history_steps = 12`

Target:

* future glucose value at `t + H`

Prediction horizons:

* PH = 30 minutes
* PH = 60 minutes
* PH = 120 minutes as a robustness check

Deep learning models receive the input as sequences with shape `(N, T, F)`. Traditional machine learning models receive the same window information in flattened tabular form.

## Feature sets

Core benchmark feature sets:

| Feature set | Variables                                                       |
| ----------- | --------------------------------------------------------------- |
| E0          | `glucose`                                                       |
| E1          | `glucose`, `basal_rate`, `bolus_volume_delivered`, `carb_input` |
| E2          | E1 plus `heart_rate`, `steps`                                   |

Exploratory feature sets:

| Feature set | Variables                                |
| ----------- | ---------------------------------------- |
| E3          | E2 plus `calories`                       |
| E4          | `glucose`, `insulin_total`, `carb_input` |

E0–E2 are used for the main benchmark. E3 and E4 are treated as exploratory extensions.

## Models

Baselines:

* Persistence
* Linear Trend

Traditional machine learning:

* Ridge Regression
* Random Forest
* XGBoost

Deep learning:

* CNN
* LSTM
* CRNN

The models are intentionally representative rather than heavily optimized. The focus is on a controlled comparison under a shared preprocessing and evaluation protocol.

## Benchmark protocol

All models are evaluated under the same benchmark protocol:

1. patient-wise preprocessing
2. harmonization to a 5-minute grid
3. sliding-window generation
4. patient-wise chronological train/validation/test split: 70/15/15
5. Exclusion Zone / Temporal Buffer at split boundaries
6. train-only scaling
7. model training
8. evaluation on the pooled test set
9. additional patient-level macro-average evaluation

The Exclusion Zone is used to prevent overlap leakage between adjacent splits in the sliding-window setup.

The temporal buffer is defined as:

```text
buffer_steps = (history_steps - 1) + horizon_steps
```

For `history_steps = 12`:

| Horizon | Buffer steps |
| ------- | -----------: |
| PH=30   |           17 |
| PH=60   |           23 |
| PH=120  |           35 |

Scaling parameters are fitted only on the training split and then applied to validation and test data.

## Evaluation metrics

Reported metrics include:

* RMSE
* MAE
* R²
* Clarke Error Grid Analysis
* Clarke Zone A+B
* micro-average on the pooled test set
* macro-average across patients
* bootstrap confidence intervals on patient level

The clinical interpretation is based on Clarke Error Grid zones. Zone A+B is reported as a compact clinical complement to numerical error metrics.

## Key results

Best main setup:

| Horizon | Best model / feature set |        RMSE | Clarke A+B |
| ------- | ------------------------ | ----------: | ---------: |
| PH=30   | LSTM / E1                | 14.89 mg/dL |     98.34% |
| PH=60   | LSTM / E1                | 27.74 mg/dL |     93.54% |

Main interpretation:

* Deep learning achieved the best RMSE in the main setup.
* Classical ML models remained competitive.
* Best-of differences between classical ML and deep learning were below 1 mg/dL RMSE.
* Therapy-event features showed the most consistent added value.
* Wearable signals did not provide consistent additional gains under the conservative missingness handling used here.

The reported results should be interpreted as within-subject temporal generalization: models are tested on later time segments of the same patients, not on entirely unseen patients.

## Repository structure

```text
.
├── config.py                         # Global paths, constants, and hyperparameters
├── pyproject.toml                    # Python project metadata and dependencies
├── data/
│   └── preprocessed/                 # Location for downloaded patient CSV files
├── results/
│   ├── csv/                          # Aggregate benchmark result tables
│   ├── figures/                      # Generated aggregate visualizations
│   └── predictions/                  # Optional local prediction exports
├── scripts/
│   ├── run_baseline_experiments.py   # Persistence and linear trend baselines
│   ├── run_tml_experiments.py        # Ridge, Random Forest, XGBoost benchmark
│   ├── run_dl_experiments.py         # CNN, LSTM, CRNN benchmark
│   ├── run_e3_e4_experiments.py      # Exploratory E3/E4 experiments
│   ├── run_per_patient_evaluation.py # Patient-level evaluation
│   ├── generate_predictions.py       # Prediction export and Clarke figures
│   └── generate_figures.py           # Thesis/benchmark figures
└── src/
    ├── data_prep.py                  # Preprocessing, feature sets, window generation
    ├── training.py                   # Splitting, scaling, training utilities
    ├── ml_models.py                  # Traditional ML model builders
    ├── models.py                     # Deep learning model builders
    ├── metrics.py                    # Regression and Clarke EGA metrics
    └── visualization.py              # Plotting utilities
```

## Installation

Requirements:

* Python 3.10 or newer

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -e .
```

Optional development tools:

```bash
pip install -e ".[dev]"
```

## How to reproduce

1. Create and activate a virtual environment.
2. Install the project dependencies.
3. Run the benchmark scripts.

Main benchmark:

```bash
python scripts/run_baseline_experiments.py
python scripts/run_tml_experiments.py
python scripts/run_dl_experiments.py
```

Exploratory E3/E4 experiments:

```bash
python scripts/run_e3_e4_experiments.py
```

Patient-level evaluation and figure generation:

```bash
python scripts/run_per_patient_evaluation.py
python scripts/generate_predictions.py
python scripts/generate_figures.py
```

Aggregate outputs are written to:

```text
results/csv/
results/figures/
```

Optional prediction-level exports may be generated under:

```text
results/predictions/
```

These prediction-level exports are intended for local analysis and are not meant to be committed as public artifacts.

## Limitations

* The benchmark is based on one public dataset and should not be treated as evidence of general clinical performance.
* The results describe within-subject temporal generalization, not generalization to unseen patients.
* Missingness handling and feature availability can affect the usefulness of wearable signals.
* Models are evaluated offline under a fixed experimental protocol.
* The project does not implement real-time monitoring, alerting, clinical validation, or deployment infrastructure.
* Results depend on the exact preprocessing, split protocol, random seed, and library versions.
* Deep learning results may show small nondeterministic variation depending on TensorFlow, hardware, and backend configuration.

## Medical disclaimer

This repository is a research and educational project. It is not intended for clinical use, diagnosis, treatment decisions, insulin dosing, real-time medical decision support, or replacement of professional medical advice.

Do not use this code or its predictions to make medical decisions.

## Citation / attribution

If you use this repository, please cite or acknowledge the original dataset:

Dataset: **HUPA-UCM Diabetes Dataset**, Mendeley Data, Version 1. DOI: `10.17632/3hbcscwz44.1`.

Data article: Hidalgo et al., "HUPA-UCM diabetes dataset", Data in Brief 55 (2024), 110559. DOI: `10.1016/j.dib.2024.110559`.

This repository implements an independent benchmark study built on that dataset and should be attributed separately when used as a code or experimental reference.
