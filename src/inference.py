import joblib
from data import load_data, split_features_target, train_test_split_data, load_config

config = load_config()
df = load_data()

X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split_data(
    X, y, config["data"]["test_size"], config["data"]["random_state"]
)

loaded_model = joblib.load("artifacts/churn_model.joblib")
sample = X_test.iloc[[0]]

print("Input customer data:")
print(sample.to_string())
print("Predicted class:", loaded_model.predict(sample)[0])
print("the churn probability", loaded_model.predict_proba(sample)[0][1])
