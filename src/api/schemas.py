from pydantic import BaseModel


class UserData(BaseModel):
    Age: int
    Gender: str
    Tenure: int
    MonthlyCharges: float
    Contract: str
    PaymentMethod: str
    TotalCharges: float
