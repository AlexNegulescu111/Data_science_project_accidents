"""Model 2: predict the traffic resumption time (XGBoost tuned with Optuna).

Two feature sets are tuned and evaluated with exactly the same procedure:
  - "phase2":        deaths, injured, county, road, hour, day, weekend
  - "circumstances": the above + road type, number of vehicles, accident vs.
                     other event, circumstance flags from the text
Deaths/injured are legitimate here: they are known when the police estimate
the resumption time.

Outputs: reports/regression_metrics.json, visuals/regression/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from xgboost import XGBRegressor

from accidents.config import RANDOM_STATE, REPORTS_DIR, UNKNOWN_COUNTY
from accidents.data import load_processed
from accidents.features import NUMERIC, build_features, is_accident
from accidents.plotting import plt, save_figure

SUBDIR = "regression"
PHASE2_NUMERIC = ["nr_morti", "nr_raniti", "hour", "day_of_week", "is_weekend"]
CIRC_NUMERIC = ["nr_morti", "nr_raniti", "is_accident"] + NUMERIC


def prepare():
    df = load_processed().dropna(subset=["reluare"])
    df["judet"] = df["judet"].replace(UNKNOWN_COUNTY, "unknown")
    df["drum"] = df["drum"].fillna("unknown")
    X = build_features(df)
    X["drum"] = df["drum"]
    X["nr_morti"] = df["nr_morti"]
    X["nr_raniti"] = df["nr_raniti"]
    X["is_accident"] = is_accident(df["descriere"]).astype(int)
    return X, df["reluare"].astype(float)


def make_pipeline(feature_set, params):
    if feature_set == "phase2":
        prep = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["judet", "drum"]),
            ("num", "passthrough", PHASE2_NUMERIC),
        ])
    else:
        prep = ColumnTransformer([
            ("te", TargetEncoder(target_type="continuous",
                                 cv=KFold(5, shuffle=True, random_state=RANDOM_STATE)), ["judet", "drum"]),
            ("road", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["road_type"]),
            ("num", "passthrough", CIRC_NUMERIC),
        ])
    return Pipeline([("prep", prep), ("model", XGBRegressor(**params, random_state=RANDOM_STATE))])


def tune(feature_set, X_train, y_train, X_val, y_val, n_trials):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 20, log=True),
        )
        pipe = make_pipeline(feature_set, params).fit(X_train, y_train)
        return root_mean_squared_error(y_val, pipe.predict(X_val))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    return study


def main(n_trials=40):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y = prepare()

    # train (60%) / validation (20%) / test (20%)
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

    metrics = {"n_test": int(len(y_test)), "outlier_bounds_min": [round(low, 1), round(high, 1)]}
    predictions = {
        "baseline_mean": np.full(len(y_test), y_trval.mean()),
        "baseline_median": np.full(len(y_test), y_trval.median()),
    }
    final = None
    for feature_set in ("phase2", "circumstances"):
        study = tune(feature_set, X_train, y_train, X_val, y_val, n_trials)
        pipe = make_pipeline(feature_set, study.best_params).fit(X_trval, y_trval)
        predictions[f"xgboost_{feature_set}"] = pipe.predict(X_test)
        metrics[f"xgboost_{feature_set}_validation_rmse"] = round(study.best_value, 2)
        metrics[f"xgboost_{feature_set}_best_params"] = study.best_params
        final = pipe

    for name, pred in predictions.items():
        metrics[f"{name}_mae"] = round(float(mean_absolute_error(y_test, pred)), 2)
        metrics[f"{name}_rmse"] = round(float(root_mean_squared_error(y_test, pred)), 2)
        print(f"{name:<24} MAE {metrics[f'{name}_mae']:6.2f} | RMSE {metrics[f'{name}_rmse']:6.2f}")
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "regression_metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- figures -----------------------------------------------------------
    names = list(predictions)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, [metrics[f"{n}_mae"] for n in names], color=["#bbbbbb"] * 2 + ["#7aa6c2", "#c2410c"])
    for i, n in enumerate(names):
        ax.text(metrics[f"{n}_mae"], i, f" {metrics[f'{n}_mae']:.2f}", va="center")
    ax.set_xlabel("MAE on the test set (minutes)")
    ax.set_title("Traffic resumption time - model vs. constant baselines")
    save_figure("model_comparison", fig=fig, subdir=SUBDIR)

    y_pred = predictions["xgboost_circumstances"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, s=14)
    ax.plot([low, high], [low, high], "k--", lw=1, label="perfect prediction")
    ax.axhline(y_trval.mean(), color="tab:red", lw=1, label="baseline (mean)")
    ax.set_xlabel("Actual resumption time (min)")
    ax.set_ylabel("Predicted resumption time (min)")
    ax.set_title(f"XGBoost (circumstances) - MAE {metrics['xgboost_circumstances_mae']} min")
    ax.legend()
    save_figure("predicted_vs_actual", fig=fig, subdir=SUBDIR)

    names_out = [n.split("__", 1)[-1] for n in final.named_steps["prep"].get_feature_names_out()]
    importance = pd.Series(final.named_steps["model"].feature_importances_, index=names_out).sort_values()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(importance.index, importance.values, color="steelblue")
    ax.set_title("XGBoost feature importance (gain)")
    save_figure("feature_importance", fig=fig, subdir=SUBDIR)


if __name__ == "__main__":
    main()
