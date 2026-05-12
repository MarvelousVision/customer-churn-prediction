import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def load_config():
    with open("configs/config.yaml", "r") as file:
        return yaml.safe_load(file)


def load_data():
    return pd.read_csv("data/mi_sss.csv")


def split_features_target(df):
    config = load_config()
    t = config["data"]["target"]
    X = df.drop(columns=[t, "CustomerID"])
    y = df[t]

    if y.dtype == "object":
        y = y.map(config["data"]["target_mapping"])
        if y.isnull().any():
            raise ValueError(
                f"Target mapping failed. Missing values found in target column. "
                f"Total missing: {y.isnull().sum()}. "
                f"Unmapped values: {y[y.isnull()].index.tolist()[:10]}"
            )
    return X, y


def train_test_split_data(X, y, t_size, random_state):
    return train_test_split(
        X, y, test_size=t_size, random_state=random_state, stratify=y
    )
