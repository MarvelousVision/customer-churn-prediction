from pathlib import Path
import joblib
from .data import load_data, split_features_target, train_val_test_split_data, load_config
from .model import (
    build_model_pipeline,
    evaluate_model,
    fit_and_score_model,
)
FINAL_THRESHOLD = 0.20
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "artifacts" / "churn_model.joblib"


config = load_config()
df = load_data()

X, y = split_features_target(df)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_data(
    X, y, config["data"]["random_state"]
)

m = build_model_pipeline(X_train)
base_result = fit_and_score_model(m, X_train, X_val, y_train)

cv = base_result["cv_mean"]
y_proba = base_result["y_proba"]

result = evaluate_model(y_proba, y_val, FINAL_THRESHOLD)
print("Model:", config["model"]["type"])
print(f"Threshold: {FINAL_THRESHOLD}")
print(f"CV mean: {cv:.4f}")
print(f"Accuracy:  {result['accuracy']:.4f}")
print(f"Precision: {result['precision']:.4f}")
print(f"Recall:    {result['recall']:.4f}")
print(f"Profit:    {result['profit']:,}")
print("-" * 40)

joblib.dump(m, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")
test_proba = m.predict_proba(X_test)[:, 1]

test_result = evaluate_model(
    test_proba,
    y_test,
    threshold=FINAL_THRESHOLD,
)

print("\nFINAL TEST")
print(f"Threshold: {FINAL_THRESHOLD}")
print(f"Accuracy:  {test_result['accuracy']:.4f}")
print(f"Precision: {test_result['precision']:.4f}")
print(f"Recall:    {test_result['recall']:.4f}")
print(f"Profit:    {test_result['profit']:,}")

