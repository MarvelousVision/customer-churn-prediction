from pathlib import Path

import joblib
from data import load_data, split_features_target, train_test_split_data, load_config
from model import (
    build_model_pipeline,
    evaluate_model,
    fit_and_score_model,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "artifacts" / "churn_model.joblib"


config = load_config()
df = load_data()

X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split_data(
    X, y, config["data"]["test_size"], config["data"]["random_state"]
)

m = build_model_pipeline(X_train)
base_result = fit_and_score_model(m, X_train, X_test, y_train)

cv = base_result["cv_mean"]
y_proba = base_result["y_proba"]

for threshold in [0.3]:
    result = evaluate_model(y_proba, y_test, threshold)
    print("Model:", config["model"]["type"])
    print(f"Threshold: {threshold}")
    print(f"CV mean: {cv:.4f}")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"Profit:    {result['profit']:,}")
    print("-" * 40)

joblib.dump(m, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")
