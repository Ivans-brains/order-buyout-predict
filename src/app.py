import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Order Buyout Risk API")
bundle = joblib.load("models/buyout_bundle.joblib")
fe = bundle["fe"]
model = bundle["model"]
threshold = bundle["threshold"]

class OrderRequest(BaseModel):
    order_time: str
    category: str
    items_count: int
    order_amount: float
    user_order_count: int
    user_buyot_rate: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_buyout(order: OrderRequest):
    df = pd.DataFrame([order.model_dump()])
    df["order_time"] = pd.to_datetime(df['order_time'])

    X = fe.transform(df)

    prob = float(model.predict_proba(X)[:, 1][0])

    is_risky = prob >= threshold
    action = "warning_of_prepayment" if is_risky else "standart_checkout"

    return {
        "not_buyout_probability": round(prob, 4),
        "threshold": threshold,
        "is_risky": is_risky,
        "recommended_action": action
    }  