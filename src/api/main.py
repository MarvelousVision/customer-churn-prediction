from fastapi import FastAPI, HTTPException
import pandas as pd
from src.api.schemas import UserData, ModelInfoResponse
from src.api.model_loader import model, model_info, THRESHOLD

app = FastAPI(title="Churn Prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    return model_info


@app.post("/predict")
def predict(user_data: UserData):
    try:
        data = user_data.model_dump()
        df = pd.DataFrame([data])

        probability = model.predict_proba(df)[0, 1]

        prediction_value = 1 if probability >= THRESHOLD else 0
        label = "churn" if prediction_value == 1 else "no churn"

        return {
            "prediction": prediction_value,
            "label": label,
            "probability": float(probability),
            "threshold": THRESHOLD,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running", "docs": "/docs"}
