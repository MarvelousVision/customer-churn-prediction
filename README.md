# Customer Churn Prediction for a Telecom Company

An end-to-end machine learning project for predicting telecom customer churn with a profit-driven classification threshold and a FastAPI inference service.

## Project Summary

The goal of this project was to identify customers at risk of churn and optimize the business outcome rather than maximize a generic metric such as accuracy.

The final selected model was **CatBoost** with a classification threshold of **0.3**.

## Business Problem

Missing a customer who is actually going to churn was considered more expensive than unnecessarily contacting a loyal customer.

Because of this, the project prioritized:

* high recall for churn customers;
* threshold optimization;
* a custom business profit metric.

This allowed the final decision threshold to be selected based on business impact rather than using the default threshold of 0.5.

## Approach

The project includes a complete machine learning workflow:

* exploratory data analysis;
* custom feature engineering;
* preprocessing of numerical and categorical features;
* missing-value handling with `SimpleImputer`;
* categorical encoding with `OneHotEncoder`;
* preprocessing with `ColumnTransformer`;
* comparison of several classification models;
* cross-validation;
* classification-threshold tuning;
* business profit evaluation;
* model artifact saving;
* FastAPI inference service.

Preprocessing was kept inside the ML pipeline so that transformations were fitted only on training data and applied consistently during validation, testing, and inference.

## Models Compared

The following models were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* CatBoost

Models were compared using cross-validation performance together with recall and the custom business profit metric.

## Final Result

The final model was **CatBoost** with a threshold of **0.3**.

Key results:

* **CV mean:** `0.7617`
* **Accuracy:** `0.5782`
* **Precision:** `0.4398`
* **Recall:** `0.9970`
* **Business profit:** `488,560`

The project demonstrates an important practical result: the threshold or model with the strongest standard ML metric is not necessarily the one that produces the best business outcome.

One of the useful engineered features was `contract_tenure`, while several additional engineered features were excluded because they did not provide meaningful improvement.

## API

The trained model is exposed through a FastAPI service.

### Endpoints

* `GET /` — API status
* `GET /health` — health check
* `GET /model-info` — model metadata
* `POST /predict` — churn prediction

### Example Request

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

### Example Response

```json
{
  "prediction": 1,
  "label": "churn",
  "probability": 0.6090453355565285,
  "threshold": 0.3
}
```

## Repository Structure

```text
customer-churn-prediction/
├── src/          # training, model logic, preprocessing, and API
├── configs/      # project configuration
├── artifacts/    # trained model, metadata, and experiment results
├── data/         # dataset files
└── requirements.txt
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/MarvelousVision/customer-churn-prediction.git
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn src.api.main:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Russian Summary

Разработана ML-система для прогнозирования оттока клиентов телеком-компании с фокусом на бизнес-метрику — прибыль.

В проекте реализованы feature engineering, pipeline предобработки данных, сравнение нескольких моделей, подбор порога классификации и FastAPI-сервис для инференса.

Финальная модель — CatBoost с порогом 0.3. Модель и порог выбирались с учетом recall и итоговой бизнес-прибыли, а не только стандартных ML-метрик.
