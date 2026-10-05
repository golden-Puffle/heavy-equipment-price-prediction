# Heavy Equipment Selling Price Prediction

This project was developed as part of the **Machine Learning Practice (MLP)** project component of the **IIT Madras Diploma in Data Science (Data Science and Applications)** program.

## Problem Statement

The goal is to predict the selling price (`TargetValue`) of heavy equipment based on transaction records, equipment specifications, usage history, and sale details, using the Kaggle competition dataset for the *Heavy Equipment Selling Price Prediction Challenge*. The task is framed as a regression problem, evaluated using **Root Mean Squared Log Error (RMSLE)**.

## Dataset

- `train.csv` — labeled training transactions
- `test.csv` — unlabeled transactions for final prediction
- `sample_submission.csv` — submission format reference

The dataset includes numeric fields (e.g. manufacture year, operational hours), categorical fields (e.g. region, cabin type, product specifications), and a transaction date.

## Approach

### 1. Exploratory Data Analysis
- Examined the target distribution (right-skewed) and confirmed that a `log1p` transform normalizes it, aligning the training objective with the RMSLE evaluation metric.
- Identified missing-value patterns across columns.
- Investigated relationships between individual features (manufacture year, operational hours, region, utilization tier, cabin type) and the target.
- Found and handled placeholder/sentinel values in `ManufactureYear` (non-real years used as missing-value codes).

### 2. Feature Engineering
- `AssetAge` — age of equipment at time of sale
- `HoursPerYear` — usage intensity
- `DescriptorLength` — proxy for specification richness from a free-text field
- Missingness indicator flags for columns where absence itself may be informative
- Outlier capping on operational hours at the 99th percentile (computed on training data only)

### 3. Preprocessing
- Train/validation split performed before any imputation or encoding to prevent data leakage
- Median imputation for numeric columns, mode imputation for low-cardinality categorical columns
- One-hot encoding for low-cardinality categorical features
- K-fold smoothed target encoding for high-cardinality categorical features
- A separate preprocessing path using native categorical handling for LightGBM, which generally outperforms manual encoding for tree-based boosting models

### 4. Pipelines
A custom `KFoldTargetEncoder` transformer (sklearn-compatible, implementing `fit`/`transform`) was built to allow target encoding to be used inside `sklearn.Pipeline` objects alongside `ColumnTransformer`, ensuring preprocessing steps are fit only on training data and applied consistently to validation and test data.

### 5. Model Comparison
Three models from different algorithmic families were trained and compared:

| Model | RMSLE |
|---|---|
| Random Forest | 0.2235 |
| LightGBM (untuned) | 0.2120 |
| MLPRegressor | 0.2796 |
| LightGBM (RandomizedSearchCV tuned) | 0.2002 |
| LightGBM (manually tuned) | 0.1993 |

### 6. Hyperparameter Tuning
- `RandomizedSearchCV` was used to systematically search the LightGBM hyperparameter space (20 sampled combinations, 3-fold cross-validation).
- Manual refinement around the search's best region was used to further reduce RMSLE, given the limitations of a coarse, randomly-sampled search grid.
- Early stopping was used throughout to prevent overfitting and determine the optimal number of boosting rounds.

### 7. Evaluation
Train vs. validation RMSLE was compared to check for overfitting. Feature importance was examined to confirm that engineered features contributed meaningfully to the final model.

## Project Structure

```
.
├── heavy_equipment_price_prediction.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook heavy_equipment_price_prediction.ipynb
```

Note: the raw competition data is not included in this repository. Download `train.csv`, `test.csv`, and `sample_submission.csv` from the competition page and place them in the path referenced at the top of the notebook.

## Author

Monisha
GitHub: [golden-Puffle](https://github.com/golden-Puffle)