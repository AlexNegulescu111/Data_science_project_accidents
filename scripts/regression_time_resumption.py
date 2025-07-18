from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import optuna

# 1. Load + preprocess
df = pd.read_csv(ROOT / "data" / "accidente_processed.csv")
df = df[["nr_morti", "nr_raniti", "reluare", "judet", "drum", "data_ora"]].dropna()
df = df[~df["judet"].str.contains("nementionat", case=False)]
df = df[~df["drum"].str.contains("nan", case=False)]

# 2. Remove outliers (IQR)
q1 = df["reluare"].quantile(0.25)
q3 = df["reluare"].quantile(0.75)
iqr = q3 - q1
df = df[(df["reluare"] >= q1 - 1.5 * iqr) & (df["reluare"] <= q3 + 1.5 * iqr)]

# 3. Feature engineering
df["data_ora"] = pd.to_datetime(df["data_ora"])
df["ora"] = df["data_ora"].dt.hour
df["zi"] = df["data_ora"].dt.dayofweek
df["este_weekend"] = df["zi"].isin([5, 6]).astype(int)

df["judet"] = LabelEncoder().fit_transform(df["judet"])
df["drum"] = LabelEncoder().fit_transform(df["drum"])

# 4. Split
X = df[["nr_morti", "nr_raniti", "judet", "drum", "ora", "zi", "este_weekend"]]
y = df["reluare"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 5. Optuna objective function
def objective(trial):
    model = XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return root_mean_squared_error(y_test, pred)  # RMSE

# 6. Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# 7. Train final model with best params
best_params = study.best_params
print("Best Parameters:", best_params)

model = XGBRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# 8. Final evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"MAE: {mae:.2f} minute")
print(f"RMSE: {rmse:.2f} minute")
