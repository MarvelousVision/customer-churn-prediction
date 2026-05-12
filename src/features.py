import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


def load_config():
    with open("configs/config.yaml", "r") as file:
        return yaml.safe_load(file)


def create_preprocessor(X):
    config = load_config()
    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X)
    numeric_columns = X_train_fe.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_columns = X_train_fe.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_transform = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transform = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown=config["features"]["handle_unknown"],
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transform, numeric_columns),
            ("cat", categorical_transform, categorical_columns),
        ]
    )

    return preprocessor


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_charge_ = None
        self.age_bins_ = None
        self.tenure_bins_ = None
        self.charge_bins_ = None

    def fit(self, X, y=None):
        self.median_charge_ = X["MonthlyCharges"].median()
        self.tenure_bins_ = [0, 12, 24, 48, float("inf")]
        self.charge_bins_ = [0, 30, 60, 90, float("inf")]
        self.age_bins_ = [0, 30, 45, 60, float("inf")]
        return self

    def transform(self, X):
        X_copy = X.copy()
        if X_copy["TotalCharges"].dtype == "object":
            X_copy["TotalCharges"] = pd.to_numeric(
                X_copy["TotalCharges"], errors="coerce"
            )

        X_copy["TotalCharges"] = X_copy["TotalCharges"].fillna(
            X_copy["MonthlyCharges"] * (X_copy["Tenure"] + 1)
        )

        X_copy["high_risk_contract"] = (
            (X_copy["Contract"] == "Month-to-month")
            & (X_copy["MonthlyCharges"] > self.median_charge_)
        ).astype(int)

        X_copy["avg_monthly_charge"] = X_copy["TotalCharges"] / (X_copy["Tenure"] + 1)

        X_copy["charge_to_tenure_ratio"] = X_copy["MonthlyCharges"] / (
            X_copy["Tenure"] + 1
        )

        bins = self.age_bins_ if self.age_bins_ is not None else [0, 30, 50, 70, 100]

        """X_copy["age_group"] = pd.cut(
            X_copy["Age"],
            bins=bins,
            labels=["Young", "Middle", "Senior", "Elderly"],
            include_lowest=True,
        )"""
        X_copy["contract_tenure"] = X_copy["Tenure"] * (
            X_copy["Contract"] == "Month-to-month"
        )

        X_copy["moths"] = X_copy["TotalCharges"] / X_copy["MonthlyCharges"]
        X_copy["old_month_to_month_risk"] = (X_copy["Tenure"] > 24) & (
            X_copy["Contract"] == "Month-to-month"
        )
        """X_copy["electronic_check_risk"] = (
            X_copy["PaymentMethod"] == "Electronic check"
        ) & (X_copy["Tenure"] < 12)"""
        return X_copy
