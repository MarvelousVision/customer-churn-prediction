# Customer Churn Prediction for a Telecom Company

An end-to-end machine learning project for predicting telecom customer churn with **business-driven model selection**, **decision-threshold tuning**, and a **FastAPI inference service**.

## Project Overview

The goal of this project is to identify telecom customers who are likely to churn and support retention decisions.

Instead of selecting a model only by accuracy, the project uses a custom **business profit metric** that reflects the different costs of prediction errors.

Missing a customer who actually churns is assumed to be substantially more expensive than contacting a loyal customer unnecessarily. Because of this, the project prioritizes **recall and business value** when selecting the decision threshold.

The final model is **CatBoost** with a decision threshold of **0.20**.

---

## Business Problem

For churn prevention, the model makes two important types of errors:

* **False Positive:** a loyal customer is incorrectly identified as being at risk and receives an unnecessary retention offer.
* **False Negative:** a customer who will churn is missed and no retention action is taken.

The project uses the following illustrative business assumptions:

| Outcome        | Business value |
| -------------- | -------------: |
| True Positive  |           +100 |
| False Positive |            -20 |
| False Negative |           -200 |
| True Negative  |              0 |

The corresponding profit function is:

```text
Profit = TP × 100 - FP × 20 - FN × 200
```

Because a false negative is much more expensive than a false positive, the optimal threshold is substantially lower than the default `0.50`.

These values are illustrative assumptions used to demonstrate business-oriented model selection.

---

## Dataset

The dataset contains approximately **100,000 telecom customers**.

The target variable is:

```text
Churn
```

with:

```text
No  -> 0
Yes -> 1
```

The churn rate is approximately **33%**.

Main input features include:

* Age
* Gender
* Tenure
* MonthlyCharges
* TotalCharges
* Contract
* PaymentMethod

Customer identifiers are excluded from model training.

---

## Evaluation Strategy

The data is divided into three stratified subsets:

* **70% training**
* **15% validation**
* **15% test**

The validation set is used for:

* comparing models;
* tuning the decision threshold;
* selecting the final configuration.

The test set is used for final evaluation after model and threshold selection.

Cross-validation is performed on the training data using **ROC-AUC**.

This separation helps reduce data leakage between model development and final evaluation.

---

## Feature Engineering

The project includes custom feature engineering implemented as part of the ML pipeline.

Engineered features include:

* `high_risk_contract`
* `avg_monthly_charge`
* `charge_to_tenure_ratio`
* `contract_tenure`
* `moths`
* `old_month_to_month_risk`

For example, `contract_tenure` combines customer tenure with month-to-month contract information to capture additional churn-risk patterns.

Feature engineering is included directly in the pipeline so the same transformations are applied during both training and inference.

---

## Preprocessing

Numerical features are processed using:

* `SimpleImputer(strategy="median")`
* `StandardScaler`

Categorical features are processed using:

* `SimpleImputer(strategy="most_frequent")`
* `OneHotEncoder(handle_unknown="ignore")`

All transformations are combined with `ColumnTransformer` and included in the final scikit-learn pipeline.

This makes preprocessing reproducible and reduces the risk of training/inference inconsistencies.

---

## Models Compared

Four models were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* CatBoost

Each model was evaluated using:

* cross-validated ROC-AUC;
* validation accuracy;
* precision;
* recall;
* business profit;
* decision-threshold tuning.

### Model Comparison

| Model               | CV ROC-AUC | Best validation threshold | Validation profit |
| ------------------- | ---------: | ------------------------: | ----------------: |
| Logistic Regression |     0.7830 |                      0.15 |           343,960 |
| Decision Tree       |     0.7917 |                      0.10 |           346,420 |
| Random Forest       |     0.8027 |                      0.17 |       **370,360** |
| CatBoost            | **0.8044** |                      0.20 |           370,300 |

Random Forest achieved the highest validation profit, while CatBoost achieved the highest cross-validated ROC-AUC.

The difference in validation profit between the two models was only **60 units**, making their business performance effectively tied under the assumed cost function.

CatBoost was therefore selected as the final model because it provided the strongest overall ranking performance while maintaining essentially the same business result.

---

## Final Model

The final configuration is:

```text
Model: CatBoost
Decision threshold: 0.20
CV ROC-AUC: 0.8044
```

CatBoost parameters:

```yaml
iterations: 60
depth: 8
learning_rate: 0.03
min_data_in_leaf: 72
```

### Validation Results

At threshold `0.20`:

| Metric    |  Result |
| --------- | ------: |
| Accuracy  |  0.5770 |
| Precision |  0.4393 |
| Recall    |  1.0000 |
| Profit    | 370,300 |

---

## Final Test Results

The selected CatBoost model and threshold were evaluated on the test split.

| Metric    |      Result |
| --------- | ----------: |
| Accuracy  |  **0.5745** |
| Precision |  **0.4378** |
| Recall    |  **0.9998** |
| Profit    | **369,160** |

The model achieves extremely high recall because the assumed business cost of missing a churn customer is much higher than the cost of contacting a loyal customer.

As a result, the selected threshold intentionally favors identifying more potentially risky customers at the expense of lower precision.

---

## Why Not Use the Default Threshold?

Using `0.50` would produce higher accuracy but significantly worse business results.

For example, for CatBoost:

```text
Threshold 0.20
Accuracy: 0.5770
Recall:   1.0000
Profit:   370,300
```

while at the default threshold:

```text
Threshold 0.50
Accuracy: 0.7672
Recall:   0.5638
Profit:  -179,960
```

This demonstrates an important conclusion of the project:

> The threshold that maximizes classification accuracy is not necessarily the threshold that maximizes business value.

For this use case, missing churn customers is expensive enough that a much lower threshold is preferred.

---

## ML Pipeline

The project uses a single pipeline containing:

```text
Raw customer data
        ↓
Custom feature engineering
        ↓
Numerical / categorical preprocessing
        ↓
CatBoost classifier
        ↓
Churn probability
        ↓
Business threshold
        ↓
Final churn prediction
```

The fitted pipeline is saved using `joblib`.

```text
artifacts/churn_model.joblib
```

Model metadata and evaluation results are stored separately in:

```text
artifacts/model_info.json
```

---

## Experiment Tracking

Model experiments are stored in:

```text
artifacts/experiment_log.csv
```

The log contains:

* model type;
* feature version;
* evaluation split;
* decision threshold;
* CV ROC-AUC;
* accuracy;
* precision;
* recall;
* business profit;
* model parameters.

This makes model-selection decisions reproducible and keeps validation experiments separate from final test evaluation.

---

## FastAPI Service

The trained model is exposed through a FastAPI inference service.

### Endpoints

* `GET /` — API status
* `GET /health` — health check
* `GET /model-info` — final model metadata
* `POST /predict` — churn prediction

### Example Prediction Request

```json
{
  "Age": 44,
  "Gender": "Male",
  "Tenure": 12,
  "MonthlyCharges": 109.67,
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "TotalCharges": 1316.04
}
```

The service applies the same feature engineering and preprocessing pipeline used during model training before producing the prediction.

---

## Training

Run model training from the project root:

```bash
python -m src.train
```

The training pipeline:

1. loads the dataset;
2. separates features and target;
3. creates train, validation, and test splits;
4. builds the preprocessing and ML pipeline;
5. evaluates the model using cross-validation;
6. evaluates the selected threshold;
7. saves the trained model artifact;
8. reports final evaluation metrics.

---

## Project Structure

```text
customer-churn-prediction/
│
├── artifacts/
│   ├── churn_model.joblib
│   ├── experiment_log.csv
│   └── model_info.json
│
├── catboost_info/
│
├── configs/
│   └── config.yaml
│
├── data/
│   └── mi_sss.csv
│
├── notebooks/
│
├── src/
│   ├── api/
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Key Takeaways

The main conclusions from the project are:

* Model selection should reflect the real business objective rather than rely only on accuracy.
* Decision-threshold tuning can have a major impact on business outcomes.
* ROC-AUC and business profit measure different aspects of model quality.
* The model with the highest generic ML metric is not automatically the model with the highest business value.
* Random Forest and CatBoost achieved almost identical validation profit, despite different ROC-AUC values.
* A reproducible feature-engineering and preprocessing pipeline helps keep training and inference consistent.
* Separating training, validation, and test data makes model evaluation more reliable.

---

## Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* CatBoost
* FastAPI
* Pydantic
* joblib
* Git / GitHub

---

## Final Result

**Final model:** CatBoost
**Decision threshold:** `0.20`
**CV ROC-AUC:** `0.8044`
**Test recall:** `0.9998`
**Test profit:** `369,160`

The project demonstrates an end-to-end ML workflow where the final prediction strategy is selected according to **business cost rather than accuracy alone**.
