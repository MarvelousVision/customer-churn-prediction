import yaml
from features import create_preprocessor
from features import FeatureEngineer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from scipy.stats import uniform, randint

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
)
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt
from data import load_data, split_features_target, train_test_split_data, load_config


def load_config():
    with open("configs/config.yaml", "r") as file:
        return yaml.safe_load(file)


def build_model_pipeline(X):
    config = load_config()
    preprocessor = create_preprocessor(X)

    model_type = config["model"]["type"]
    params = config[model_type]

    if model_type == "logistic_regression":
        model = LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            solver=params["solver"],
            random_state=config["data"]["random_state"],
            class_weight="balanced",
        )

    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=params["random_state"],
        )

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            class_weight="balanced",
        )

    elif model_type == "catboost":
        model = CatBoostClassifier(
            iterations=params["iterations"],
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            verbose=params["verbose"],
            min_data_in_leaf=params["min_data_in_leaf"],
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline(
        [
            ("feature_engineer", FeatureEngineer()),
            ("prep", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def evaluate_model(y_proba, y_test, threshold):
    config = load_config()

    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    gain = config["profit"]["gain"]
    cost = config["profit"]["cost"]
    loss = config["profit"]["loss"]
    profit = tp * gain - fp * cost - fn * loss

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "profit": int(profit),
    }


def fit_and_score_model(m, X_train, X_test, y_train):
    cv_scores = cross_val_score(m, X_train, y_train, cv=4)
    cv_mean = cv_scores.mean()

    m.fit(X_train, y_train)

    y_proba = m.predict_proba(X_test)[:, 1]
    return {"cv_mean": cv_mean, "y_proba": y_proba}
