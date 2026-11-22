import pandas as pd

class OrderFeatureEngineering:
    def __init__(self):
        self.category_mean_amount_: dict[str, float] = {}
        self.category_to_id_: dict[str, int] ={
            "books": 0,
            "clothes": 1,
            "home": 2,
            "electronics": 3
        }

    def fit(self, df: pd.DataFrame):
        self.category_mean_amount_ = df.groupby('category')['order_amount'].mean().round(2).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        data['order_hour'] = data["order_time"].dt.hour
        data["day_of_week"] = data["order_time"].dt.dayofweek
        data["is_night"] = ((data["order_hour"] >= 22) | (data["order_hour"] <= 5)).astype(int)

        data["avg_amount"] = (data["order_amount"] / data["items_count"]).round(2)

        cat_mean = data["category"].map(self.category_mean_amount_)
        data["amount_to_cat_mean"] = data["order_amount"] / cat_mean

        data["category_id"] = data["category"].map(self.category_to_id_).fillna(-1).astype(int)

        cols_to_drop = ["user_id", "order_time", "category", "is_not_byout"]
        existing_drop = [c for c in cols_to_drop if c in data.columns]
        return data.drop(columns=existing_drop)

if __name__ == "__main__":
    from src.data_gen import generation_orders
    df = generation_orders(200)
    fe = OrderFeatureEngineering()
    X = fe.fit(df).transform(df)
    print(X.head())
    print("\nЗапомненные средние по категориям:", fe.category_mean_amount_)