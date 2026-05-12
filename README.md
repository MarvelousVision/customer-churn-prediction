# Customer Churn Prediction for a Telecom Company
 A production-ready churn prediction system with a full ML pipeline and deployed FastAPI service.

## Problem
The goal of this project was to build a churn prediction model that optimizes business profit by identifying customers at risk of churn.
## Data and Pipeline
I built a full machine learning pipeline with:
- custom feature engineering
- separate preprocessing for numerical and categorical features
- `SimpleImputer`
- `OneHotEncoder`
- `ColumnTransformer`

This pipeline also helped prevent data leakage and made the workflow reproducible.

## Models Compared
I compared four models:
- Logistic Regression
- Decision Tree
- Random Forest
- CatBoost

## Business Metric and Threshold
Recall was the key metric in this project because missing a churn customer was more expensive than sending an unnecessary offer.
That is why I selected a relatively low classification threshold of **0.3**, which gave high recall and the strongest business result.

## Final Result
The final production choice was **CatBoost** because it achieved:
- the best cross-validation score: **0.7617**
- the highest profit: **488,180**

It also produced the strongest balance between model quality and business value.

## API (FastAPI)  
The trained model is deployed as a FastAPI service.

### Endpoints

- `GET /` — API status
- `GET /health` — health check
- `GET /model-info` — model metadata
- `POST /predict` — get churn prediction

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
  "probability": 0.59,
  "threshold": 0.3
}
```
## How to run 

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd pet2
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
Разработала модель прогнозирования оттока клиентов для телеком-компании с оптимизацией под бизнес-метрику — прибыль. Ключевая особенность задачи заключалась в том, что пропуск уходящего клиента обходится значительно дороже, чем ложноположительное срабатывание, поэтому важнейшей метрикой стал Recall, а также был подобран оптимальный порог классификации. В проекте использовались полный ML-пайплайн, feature engineering, предобработка данных, кросс-валидация и сравнение нескольких моделей: Logistic Regression, Random Forest и CatBoost. После настройки гиперпараметров лучшей моделью стал CatBoost, который показал лучший CV score и максимальную прибыль. Проект ориентирован не только на качество модели, но и на принятие решений с точки зрения бизнеса.