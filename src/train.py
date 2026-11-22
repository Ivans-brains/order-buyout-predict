from pathlib import Path
import joblib 
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.data_gen import generation_orders
from src.features import OrderFeatureEngineering

TARGET_COL = "is_not_byout"

def find_best_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        fn_cost: float = 300.0,
        fp_cost: float = 150.0
) -> tuple[float, float]:
    best_thresh = 0.5
    min_loss = float("inf")

    for thresh in np.arange(0.10, 0.91, 0.05):
        preds = (y_prob >= thresh).astype(int)
        fp = int(np.sum((y_true == 0) & (preds == 1)))
        fn = int(np.sum((y_true == 1) & (preds == 0)))
        total_loss = fp * fp_cost + fn * fn_cost
        if total_loss < min_loss:
            min_loss = total_loss
            best_thresh = float(round(thresh, 2))

    return best_thresh, float(min_loss)

def train() -> None:
    df = generation_orders(n_rows=5000, random_state=42)
    y = df[TARGET_COL].to_numpy()
    groups = df["user_id"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=5)
    oof_preds = np.zeros(len(df))
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(df, y, groups=groups), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        fe = OrderFeatureEngineering()
        X_train = fe.fit(train_df).transform(train_df)
        X_val = fe.transform(val_df)

        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y[train_idx])

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs

        auc = roc_auc_score(y[val_idx], val_probs)
        fold_aucs.append(auc)
        print(f"Fold {fold} ROC-AUC: {auc:.4f}")

    print(f"\nСредний ROC-AUC по 5 фолдам: {np.mean(fold_aucs):.4f}")

    best_thresh, min_loss = find_best_threshold(y, oof_preds)
    print(f"Оптимальный порог по деньгам: {best_thresh} (потери: {min_loss:.0f} руб.)")

    final_fe = OrderFeatureEngineering()
    X_full = final_fe.fit(df).transform(df)
    final_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        verbose=-1
    )
    final_model.fit(X_full, y)

    Path('models').mkdir(exist_ok=True)
    bundle = {"fe": final_fe, "model": final_model, "threshold": best_thresh}
    joblib.dump(bundle, "models/buyout_bundle.joblib")
    print('Модель успешно сохранена в models/buyout_bundle.joblib')

if __name__ == "__main__":
    train()
