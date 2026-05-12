from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt
from data import load_data, split_features_target, train_test_split_data, load_config
from model import (
    build_model_pipeline,
    evaluate_model,
    fit_and_score_model,
)
import joblib
from datetime import datetime

config = load_config()
df = load_data()

X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split_data(
    X, y, config["data"]["test_size"], config["data"]["random_state"]
)


m = build_model_pipeline(X_train)
base_result = fit_and_score_model(m, X_train, X_test, y_train)
# joblib.dump(m1, "artifacts/churn_model.joblib")
# m = joblib.load("artifacts/churn_model.joblib")
model = m.named_steps["model"]
features_names = m.named_steps["prep"].get_feature_names_out()
importances = model.feature_importances_

i_df = pd.DataFrame({"feature": features_names, "importance": importances}).sort_values(
    "importance", ascending=False
)

print(i_df.head(15).to_string(index=False))


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
