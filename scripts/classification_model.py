"""Model 1: predict whether an accident is fatal from its circumstances.

Only rows that describe an actual accident are used (no roadworks, vehicle
fires, oversize transports...). Several models are compared with 5-fold
cross-validation on the training set; the best one (by PR-AUC) is evaluated
once on the held-out test set and explained with SHAP / coefficients.

Outputs: reports/classification_metrics.json, visuals/classification/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd
import shap
from matplotlib.ticker import FuncFormatter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, average_precision_score,
                             classification_report, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from accidents.config import RANDOM_STATE, REPORTS_DIR, UNKNOWN_COUNTY
from accidents.data import load_processed
from accidents.features import (BINARY, CATEGORICAL, CONTINUOUS, NUMERIC,
                                build_features, is_accident)
from accidents.plotting import plt, save_figure

SUBDIR = "classification"
# Phase-1 feature set, kept only as a reference: `nr_raniti` and `reluare`
# are known only *after* the accident, so they are not circumstances.
REFERENCE_FEATURES = ["nr_raniti", "reluare", "judet", "drum"]


def prepare():
    df = load_processed()
    df = df[is_accident(df["descriere"])].copy()
    df["judet"] = df["judet"].replace(UNKNOWN_COUNTY, "unknown")
    df["drum"] = df["drum"].fillna("unknown")
    X = build_features(df)
    X_ref = df[REFERENCE_FEATURES]
    y = (df["nr_morti"] > 0).astype(int)
    return X, X_ref, y


def candidate_models():
    one_hot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)
    logistic = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", one_hot, CATEGORICAL),
            ("num", StandardScaler(), CONTINUOUS),
            ("flag", "passthrough", BINARY),   # 0/1 flags: coefficient = log odds ratio
        ])),
        ("model", LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000)),
    ])
    forest = Pipeline([
        ("prep", ColumnTransformer([
            ("county", TargetEncoder(cv=KFold(5, shuffle=True, random_state=RANDOM_STATE)), ["judet"]),
            ("road", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["road_type"]),
            ("num", "passthrough", NUMERIC),
        ])),
        ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=10, class_weight="balanced",
                                         random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    boosting = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ])),
        ("model", HistGradientBoostingClassifier(
            categorical_features=[0, 1], learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0, class_weight="balanced",
            random_state=RANDOM_STATE)),
    ])
    return {"logistic_regression": logistic, "random_forest": forest, "gradient_boosting": boosting}


def reference_model():
    return Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["judet", "drum"]),
            ("num", SimpleImputer(strategy="median"), ["nr_raniti", "reluare"]),
        ])),
        ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                         random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def feature_names(pipe):
    return [n.split("__", 1)[-1] for n in pipe.named_steps["prep"].get_feature_names_out()]


def explain(name, pipe, X_test, y_test, n_cases=3):
    """SHAP for tree models, coefficients for the logistic regression."""
    names = feature_names(pipe)
    if name == "logistic_regression":
        coef = pd.Series(pipe.named_steps["model"].coef_[0], index=names)
        flags = coef[[n for n in names if n in BINARY]]
        counties = coef[[n for n in names if n.startswith("judet_")]]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        for ax, values, title in [
            (ax1, flags, "Circumstances (vs. not present)"),
            (ax2, pd.concat([counties.nsmallest(6), counties.nlargest(6)]).drop_duplicates(),
             "Counties (vs. average county)"),
        ]:
            odds = np.exp(values).sort_values()
            ax.barh([n.replace("has_", "").replace("judet_", "") for n in odds.index], odds.values,
                    color=np.where(odds.values > 1, "firebrick", "steelblue"))
            ax.axvline(1, color="k", lw=1)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v in (0.3, 0.5, 2, 3, 5) else ""))
            ax.set_xlabel("odds ratio of a fatal outcome (log scale)")
            ax.set_title(title)
        fig.suptitle("Logistic regression - odds ratios, other features held constant")
        fig.tight_layout()
        save_figure("odds_ratios", fig=fig, subdir=SUBDIR)
        return {n: round(float(np.exp(v)), 3) for n, v in flags.sort_values(ascending=False).items()}

    X_t = pd.DataFrame(pipe.named_steps["prep"].transform(X_test), columns=names, index=X_test.index)
    explanation = shap.TreeExplainer(pipe.named_steps["model"])(X_t)
    if explanation.values.ndim == 3:            # (rows, features, classes)
        explanation = explanation[:, :, 1]
    shap.summary_plot(explanation, X_t, plot_type="bar", show=False, max_display=15)
    plt.title("Mean |SHAP value| - contribution to the 'fatal' prediction")
    save_figure("shap_importance", subdir=SUBDIR)
    shap.summary_plot(explanation, X_t, show=False, max_display=15)
    save_figure("shap_beeswarm", subdir=SUBDIR)
    # shap values are indexed by POSITION
    for k, pos in enumerate(np.flatnonzero(y_test.to_numpy() == 1)[:n_cases], start=1):
        shap.plots.waterfall(explanation[pos], show=False, max_display=12)
        save_figure(f"shap_fatal_case_{k}", subdir=SUBDIR)


def main():
    X, X_ref, y = prepare()
    print(f"{len(X)} accidents, {y.sum()} fatal ({y.mean():.1%})")

    idx_train, idx_test = train_test_split(X.index, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = X.loc[idx_train], X.loc[idx_test], y.loc[idx_train], y.loc[idx_test]

    # 1. Model selection with cross-validation on the training set only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
    models = candidate_models()
    cv_results = {}
    for name, pipe in {**models, "reference_phase1_features": reference_model()}.items():
        data = X_ref.loc[idx_train] if name.startswith("reference") else X_train
        res = cross_validate(pipe, data, y_train, cv=cv, scoring=scoring)
        cv_results[name] = {
            "cv_roc_auc": round(float(res["test_roc_auc"].mean()), 4),
            "cv_roc_auc_std": round(float(res["test_roc_auc"].std()), 4),
            "cv_pr_auc": round(float(res["test_pr_auc"].mean()), 4),
            "cv_pr_auc_std": round(float(res["test_pr_auc"].std()), 4),
        }
        print(f"{name:<28} CV ROC-AUC {cv_results[name]['cv_roc_auc']:.3f} | PR-AUC {cv_results[name]['cv_pr_auc']:.3f}")

    best = max(models, key=lambda n: cv_results[n]["cv_pr_auc"])
    print(f"\nSelected model: {best}")

    # 2. Final evaluation on the untouched test set
    pipe = models[best].fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    # Decision threshold: best F1 on OUT-OF-FOLD predictions of the training set
    oof = cross_val_predict(models[best], X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(y_train, oof)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    threshold = float(t[np.argmax(f1[:-1])])
    y_pred = (proba >= threshold).astype(int)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "rows": "accidents only (non-accident reports removed)",
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_test_fatal": int(y_test.sum()), "prevalence": round(float(y_test.mean()), 4),
        "cv_model_comparison": cv_results,
        "selected_model": best,
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "test_pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "threshold": round(threshold, 4),
        "fatal_precision": round(report["1"]["precision"], 4),
        "fatal_recall": round(report["1"]["recall"], 4),
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "classification_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "cv_model_comparison"}, indent=2))

    # 3. Figures
    comparison = pd.DataFrame(cv_results).T[["cv_roc_auc", "cv_pr_auc"]].sort_values("cv_pr_auc")
    errors = pd.DataFrame(cv_results).T.loc[comparison.index, ["cv_roc_auc_std", "cv_pr_auc_std"]].to_numpy().T
    ax = comparison.plot(kind="barh", xerr=errors, figsize=(9, 4.5), capsize=3, color=["#7aa6c2", "#c2410c"])
    ax.axvline(y_train.mean(), color="k", ls=":", lw=1, label="PR-AUC of a random model")
    ax.set_title("Model comparison - 5-fold CV on the training set")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    save_figure("model_comparison", subdir=SUBDIR)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["non-fatal", "fatal"], cmap="Blues")
    plt.title(f"Confusion matrix - {best} (threshold {threshold:.2f})")
    save_figure("confusion_matrix", subdir=SUBDIR)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax1, name=best)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax1.legend()
    ax1.set_title("ROC curve (test set)")
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax2, name=best)
    ax2.axhline(y_test.mean(), color="k", ls="--", lw=1, label="chance (prevalence)")
    ax2.legend()
    ax2.set_title("Precision-recall curve (test set)")
    save_figure("roc_pr_curves", fig=fig, subdir=SUBDIR)

    odds = explain(best, pipe, X_test, y_test)
    if odds:
        metrics["odds_ratios_circumstances"] = odds
        (REPORTS_DIR / "classification_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
