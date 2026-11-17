from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42

# 1. Load and clean data
df = pd.read_csv(ROOT / "data" / "accidente_processed.csv")
df = df[["nr_morti", "nr_raniti", "reluare", "judet", "drum"]].dropna()
df = df[~df["judet"].str.contains("nementionat", case=False)]

# 2. Create binary label
df["mortal"] = (df["nr_morti"] > 0).astype(int)

# 3. Encode categoric data
df["judet"] = LabelEncoder().fit_transform(df["judet"])
df["drum"] = LabelEncoder().fit_transform(df["drum"])

# 4. Define features and target
FEATURES = ["nr_raniti", "reluare", "judet", "drum"]
X = df[FEATURES]
y = df["mortal"]

# 5. Split train/test FIRST, on the original (imbalanced) data.
# Balancing before the split copies minority rows into both sets and makes the
# test score meaningless, so the imbalance is handled inside the model instead
# (class_weight), using only the training data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# 6. Training
model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Cross-validated score on the training set (more stable than a single split)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
cv_ap = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision")
print(f"CV ROC-AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
print(f"CV PR-AUC : {cv_ap.mean():.3f} ± {cv_ap.std():.3f}")

model.fit(X_train, y_train)

# 7. Evaluate the model on the untouched test set
proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= 0.5).astype(int)
print(f"\nTest set: {len(y_test)} accidents, {y_test.sum()} fatal "
      f"(prevalence {y_test.mean():.3f} = PR-AUC of a random model)")
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
print(f"PR-AUC : {average_precision_score(y_test, proba):.3f}")

# 8. Interpretation of the model with shap (on the test set)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
shap_values_class1 = shap_values[:, :, 1]
shap.summary_plot(shap_values_class1, X_test, plot_type="bar")

# 9. Explain some fatal cases.
# shap_values are indexed by POSITION, not by the DataFrame index label.
fatal_positions = np.flatnonzero(y_test.to_numpy() == 1)
for pos in fatal_positions[:10]:
    shap.plots.waterfall(shap_values_class1[pos])
