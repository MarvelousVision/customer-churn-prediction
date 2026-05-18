# Customer Churn Prediction for a Telecom Company

A production-oriented machine learning project for predicting telecom customer churn with a profit-driven decision threshold and a deployed FastAPI service.

## Project Summary

The goal of this project was to identify customers at risk of churn and optimize business profit rather than maximize a generic metric like accuracy.

The final solution used a **CatBoost** model with a classification threshold of **0.3**, selected based on business impact and recall-oriented evaluation.

## Business Problem

In this task, missing a true churn customer was more expensive than contacting a loyal customer unnecessarily.  
Because of that, the project focused on **recall** and **profit-based decision-making** instead of accuracy alone.

## Approach

The project includes a full ML workflow:

- custom feature engineering
- preprocessing for numerical and categorical features
- missing value handling with `SimpleImputer`
- categorical encoding with `OneHotEncoder`
- feature transformation with `ColumnTransformer`
- controlled model comparison
- threshold tuning
- saved model artifacts
- FastAPI deployment

The pipeline also helped prevent data leakage and made training and inference reproducible.

## Models Compared

The following models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- CatBoost

Experiments were tracked and compared using cross-validation metrics, recall, and business profit.

## Final Result

The final production choice was **CatBoost** with a threshold of **0.3**.

Key results:

- **Best CV mean:** `0.7617`
- **Best profit:** `488,560`
- **Priority metric:** Recall
- **Important conclusion:** the best CV score and the best business result are not always the same thing

One of the most useful engineered features was **contract_tenure**, while some additional engineered features did not improve the final result enough to keep.

## API (FastAPI)

The trained model is exposed through a FastAPI service.

### Endpoints

- `GET /` — API status
- `GET /health` — health check
- `GET /model-info` — model metadata
- `POST /predict` — churn prediction

### Example request

```json
{
  "Age": 44,
  "Gender": "Male",
  "Tenure": 12,
  "MonthlyCharges": 109.67,
  "Contract": "Month-to-month",
  "PaymentMethod": "Mailed check",
  "TotalCharges": 1500.5
}
```
### Example response
```json
{
  "prediction": 1,
  "label": "churn",
  "probability": 0.6090453355565285,
  "threshold": 0.3
}
```
### Repository Structure

src/        source code for training, inference, and API
configs/    configuration files
artifacts/  saved model, metadata, and experiment results
data/       dataset files

## How to run 

1. Clone the repository:
```bash
git clone https://github.com/MarvelousVision/customer-churn-prediction.git
cd customer-churn-prediction
```
### Install dependencies:
```bash
pip install -r requirements.txt
```
### Run the API:
```bash
uvicorn src.api.main:app --reload
```
### Open in browser:
http://127.0.0.1:8000/docs

## Russian summary
Разработана ML-система для прогнозирования оттока клиентов телеком-компании с фокусом на бизнес-метрику — прибыль.
В проекте использовались feature engineering, полный pipeline предобработки, сравнение нескольких моделей, подбор порога классификации и FastAPI для инференса.

Лучшая финальная модель — CatBoost с порогом 0.3, выбранная на основе recall и итоговой прибыли, а не только стандартных ML-метрик.