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


class ModelInfoResponse(BaseModel):
    business_metric: str
    model_type: str
    threshold: float
    cv_mean: float
    profit: int
    parameters: ModelParameters
    test_metrics: TestMetrics
    created_at: str
