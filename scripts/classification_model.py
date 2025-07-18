from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap

# 1. Load and clean data
df = pd.read_csv(ROOT / "data" / "accidente_processed.csv")
df = df[["nr_morti", "nr_raniti", "reluare", "judet", "drum"]].dropna()
df = df[~df["judet"].str.contains("nementionat", case=False)]
df = df[~df["drum"].str.contains("nan", case=False)]

# 2. Create binary label
df["mortal"] = (df["nr_morti"] > 0).astype(int)

# 3. Encode categoric data
df["judet"] = LabelEncoder().fit_transform(df["judet"])
df["drum"] = LabelEncoder().fit_transform(df["drum"])

# 4. Upsampling
df_majority = df[df["mortal"] == 0]
df_minority = df[df["mortal"] == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

# 5. Define features and target
X = df_balanced[["nr_raniti", "reluare", "judet", "drum"]]
y = df_balanced["mortal"]

# 6. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 7. Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# 9. Interpretation of the model with shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)
shap_values_class1 = shap_values[:, :, 1] 
shap.summary_plot(shap_values_class1, X_train, plot_type="bar")

# 10. Explain some mortal cases
a = 0
y_train_mortal = y_train[y_train>0].index
for i in y_train_mortal:
    shap.plots.waterfall(shap_values_class1[i])
    a+=1
    if(a>10):
        break

