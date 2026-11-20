import numpy as np
import pandas as pd


def generation_orders(n_rows: int =  5000, random_state: int=42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)


    n_users = max(1, n_rows // 4)
    user_id = rng.integers(1, n_users + 1, size=n_rows)

    user_base_rates = rng.uniform(0.2, 1.0, size=n_users + 1)
    user_bayout_rate = user_base_rates[user_id].round(2)
    user_order_count = rng.integers(0, 51, size=n_rows)

    start_date = pd.Timestamp('2026-08-01')
    random_minutes = rng.integers(0, 90*24*60, size=n_rows)
    order_time = start_date + pd.to_timedelta(random_minutes, unit='m')

    categories = ['electronics', 'clothes', 'books', 'home']
    category = rng.choice(categories, size=n_rows, p=[0.2, 0.35, 0.2, 0.25])
    items_count = rng.integers(1, 5, size=n_rows)

    base_price_map = {
        'books': 600,
        'clothes': 2500,
        'electronics': 9000,
        'home': 3000
    }
    base_price = pd.Series(category).map(base_price_map).to_numpy()
    price_noise = rng.uniform(0.7, 1.4, size=n_rows)
    order_amount = (base_price * items_count * price_noise).round(-1).astype(int)

    order_hour = order_time.hour
    is_night = ((order_hour >= 22) | (order_hour <= 5)).astype(int)
    is_expensive = (order_amount >= 10000).astype(int)
    is_clothes = (category == 'clothes').astype(int)

    hidden_noise = rng.normal(0, 0.5, size=n_rows)
    night_clothes = (is_night & is_clothes).astype(int)

    risk_score = (
        -3.6
        + 2.6 * (1-user_bayout_rate)
        + 0.7 * is_night
        + 0.4 * is_expensive
        + 0.5 * is_clothes
        + 1.0 * night_clothes
        + hidden_noise
    )
    prob = 1 / (1 + np.exp(-risk_score))
    is_not_byout = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "order_time": order_time,
            "category": category,
            "items_count": items_count,
            "order_amount": order_amount,
            "user_orders_count": user_order_count,
            "user_byout_rate": user_bayout_rate,
            "is_not_byout": is_not_byout
        }
    )
    

    return df.sort_values(by='order_time').reset_index(drop=True)

if __name__ == "__main__":
    df = generation_orders(1000)
    print(df.head())
    print('\nСредний чек по категориям:\n', df.groupby("category")["order_amount"].mean().round(0))
    print("\nДоля невыкупов:\n", round(df["is_not_byout"].mean(), 3))  