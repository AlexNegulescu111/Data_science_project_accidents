from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

RANDOM_STATE = 42

# 1. Load + preprocess
df = pd.read_csv(ROOT / "data" / "accidente_processed.csv")
df = df[["nr_morti", "nr_raniti", "reluare", "judet", "drum", "data_ora"]].dropna()
df = df[~df["judet"].str.contains("nementionat", case=False)]

# 2. Feature engineering
df["data_ora"] = pd.to_datetime(df["data_ora"])
df["ora"] = df["data_ora"].dt.hour
df["zi"] = df["data_ora"].dt.dayofweek
df["este_weekend"] = df["zi"].isin([5, 6]).astype(int)

df["judet"] = LabelEncoder().fit_transform(df["judet"])
df["drum"] = LabelEncoder().fit_transform(df["drum"])

FEATURES = ["nr_morti", "nr_raniti", "judet", "drum", "ora", "zi", "este_weekend"]
X = df[FEATURES]
y = df["reluare"].astype(float)

# 3. Split: train (60%) / validation (20%) / test (20%).
# Optuna only ever sees the validation set; the test set is used exactly once.
X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.25, random_state=RANDOM_STATE)

# 4. Remove outliers (IQR) - bounds computed on the training data only
q1, q3 = y_train.quantile([0.25, 0.75])
iqr = q3 - q1
low, high = max(0, q1 - 1.5 * iqr), q3 + 1.5 * iqr
print(f"Keeping resumption times between {low:.0f} and {high:.0f} minutes")


def in_range(X_part, y_part):
    mask = (y_part >= low) & (y_part <= high)
    return X_part[mask], y_part[mask]


X_train, y_train = in_range(X_train, y_train)
X_val, y_val = in_range(X_val, y_val)
X_trval, y_trval = in_range(X_trval, y_trval)
X_test, y_test = in_range(X_test, y_test)


# 5. Optuna objective function (evaluated on the validation set)
def objective(trial):
    model = XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return root_mean_squared_error(y_val, model.predict(X_val))


# 6. Optuna study
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=30)

best_params = study.best_params
print("Best Parameters:", best_params)
print(f"Validation RMSE: {study.best_value:.2f} minutes")

# 7. Train final model with best params on train + validation
model = XGBRegressor(**best_params, random_state=RANDOM_STATE)
model.fit(X_trval, y_trval)

# 8. Final evaluation on the untouched test set, compared to naive baselines
y_pred = model.predict(X_test)
baselines = {
    "Baseline (mean)": np.full(len(y_test), y_trval.mean()),
    "Baseline (median)": np.full(len(y_test), y_trval.median()),
    "XGBoost": y_pred,
}
print(f"\nTest set: {len(y_test)} accidents")
for name, pred in baselines.items():
    mae = mean_absolute_error(y_test, pred)
    rmse = root_mean_squared_error(y_test, pred)
    print(f"{name:<18} MAE: {mae:6.2f} min | RMSE: {rmse:6.2f} min")
