from pydantic import BaseModel


class UserData(BaseModel):
    Age: int
    Gender: str
    Tenure: int
    MonthlyCharges: float
    Contract: str
    PaymentMethod: str
    TotalCharges: float


class ModelParameters(BaseModel):
    depth: int
    iterations: int
    learning_rate: float
    min_data_in_leaf: int


class TestMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    profit: int


class ModelInfoResponse(BaseModel):
    business_metric: str
    model_type: str
    threshold: float
    threshold_selection: str
    model_selection: str
    cv_metric: str
    cv_mean: float
    split: dict
    parameters: ModelParameters
    validation_metrics: TestMetrics
    test_metrics: TestMetrics
    created_at: str