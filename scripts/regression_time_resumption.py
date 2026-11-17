"""Model 2: predict the traffic resumption time (XGBoost tuned with Optuna).

Outputs: metrics in reports/regression_metrics.json and figures in
visuals/regression/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from accidents.config import RANDOM_STATE, REPORTS_DIR
from accidents.data import load_processed, model_frame
from accidents.plotting import plt, save_figure

FEATURES = ["nr_morti", "nr_raniti", "judet", "drum", "ora", "zi", "este_weekend"]
SUBDIR = "regression"


def prepare():
    df = model_frame(load_processed(), ["nr_morti", "nr_raniti", "reluare", "data_ora"])
    df["ora"] = df["data_ora"].dt.hour
    df["zi"] = df["data_ora"].dt.dayofweek
    df["este_weekend"] = df["zi"].isin([5, 6]).astype(int)
    for col in ["judet", "drum"]:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df[FEATURES], df["reluare"].astype(float)


def main(n_trials=30):
    X, y = prepare()

    # train (60%) / validation (20%) / test (20%): Optuna only sees the
    # validation set, the test set is used exactly once.
    X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.25, random_state=RANDOM_STATE)

    # IQR outlier bounds computed on the training data only
    q1, q3 = y_train.quantile([0.25, 0.75])
    low, high = max(0.0, q1 - 1.5 * (q3 - q1)), q3 + 1.5 * (q3 - q1)
    print(f"Keeping resumption times between {low:.0f} and {high:.0f} minutes")

    def in_range(X_part, y_part):
        mask = (y_part >= low) & (y_part <= high)
        return X_part[mask], y_part[mask]

    X_train, y_train = in_range(X_train, y_train)
    X_val, y_val = in_range(X_val, y_val)
    X_trval, y_trval = in_range(X_trval, y_trval)
    X_test, y_test = in_range(X_test, y_test)

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

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    print("Best parameters:", study.best_params)

    model = XGBRegressor(**study.best_params, random_state=RANDOM_STATE)
    model.fit(X_trval, y_trval)
    y_pred = model.predict(X_test)

    predictions = {
        "baseline_mean": np.full(len(y_test), y_trval.mean()),
        "baseline_median": np.full(len(y_test), y_trval.median()),
        "xgboost": y_pred,
    }
    metrics = {
        "n_test": int(len(y_test)),
        "outlier_bounds_min": [round(low, 1), round(high, 1)],
        "validation_rmse": round(study.best_value, 2),
        "best_params": study.best_params,
    }
    for name, pred in predictions.items():
        metrics[f"{name}_mae"] = round(float(mean_absolute_error(y_test, pred)), 2)
        metrics[f"{name}_rmse"] = round(float(root_mean_squared_error(y_test, pred)), 2)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "regression_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # --- figures -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, s=14)
    ax.plot([low, high], [low, high], "k--", lw=1, label="perfect prediction")
    ax.axhline(y_trval.mean(), color="tab:red", lw=1, label="baseline (mean)")
    ax.set_xlabel("Actual resumption time (min)")
    ax.set_ylabel("Predicted resumption time (min)")
    ax.set_title(f"XGBoost on the test set - MAE {metrics['xgboost_mae']} min")
    ax.legend()
    save_figure("predicted_vs_actual", fig=fig, subdir=SUBDIR)

    importance = (np.array(model.feature_importances_), np.array(FEATURES))
    order = np.argsort(importance[0])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(importance[1][order], importance[0][order], color="steelblue")
    ax.set_title("XGBoost feature importance (gain)")
    save_figure("feature_importance", fig=fig, subdir=SUBDIR)


if __name__ == "__main__":
    main()
