"""Model 1: predict whether an accident is fatal (Random Forest + SHAP).

Outputs: metrics in reports/classification_metrics.json and figures in
visuals/classification/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, average_precision_score,
                             classification_report, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from accidents.config import RANDOM_STATE, REPORTS_DIR
from accidents.data import load_processed, model_frame
from accidents.plotting import plt, save_figure

FEATURES = ["nr_raniti", "reluare", "judet", "drum"]
SUBDIR = "classification"


def prepare():
    df = model_frame(load_processed(), FEATURES + ["nr_morti"])
    df["mortal"] = (df["nr_morti"] > 0).astype(int)
    for col in ["judet", "drum"]:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df[FEATURES], df["mortal"]


def main(n_explained=3):
    X, y = prepare()

    # Split FIRST, on the original (imbalanced) data. The imbalance is handled
    # inside the model (class_weight), using only the training data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_ap = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision")

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    metrics = {
        "n_test": int(len(y_test)),
        "n_test_fatal": int(y_test.sum()),
        "prevalence": round(float(y_test.mean()), 4),
        "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_roc_auc_std": round(float(cv_auc.std()), 4),
        "cv_pr_auc_mean": round(float(cv_ap.mean()), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "test_pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "fatal_precision": round(report["1"]["precision"], 4),
        "fatal_recall": round(report["1"]["recall"], 4),
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "classification_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred, zero_division=0))
    print(json.dumps(metrics, indent=2))

    # --- figures -----------------------------------------------------------
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["non-fatal", "fatal"], cmap="Blues"
    )
    plt.title("Confusion matrix (test set, threshold 0.5)")
    save_figure("confusion_matrix", subdir=SUBDIR)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax1, name="Random Forest")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax1.legend()
    ax1.set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax2, name="Random Forest")
    ax2.axhline(y_test.mean(), color="k", ls="--", lw=1, label="chance (prevalence)")
    ax2.legend()
    ax2.set_title("Precision-recall curve")
    save_figure("roc_pr_curves", fig=fig, subdir=SUBDIR)

    # SHAP on the test set. shap_values are indexed by POSITION.
    shap_values = shap.TreeExplainer(model)(X_test)[:, :, 1]
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Mean |SHAP value| - contribution to the 'fatal' prediction")
    save_figure("shap_importance", subdir=SUBDIR)

    for k, pos in enumerate(np.flatnonzero(y_test.to_numpy() == 1)[:n_explained], start=1):
        shap.plots.waterfall(shap_values[pos], show=False)
        save_figure(f"shap_fatal_case_{k}", subdir=SUBDIR)


if __name__ == "__main__":
    main()
