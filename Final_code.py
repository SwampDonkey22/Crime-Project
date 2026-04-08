import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, ConfusionMatrixDisplay)

df = pd.read_csv("uk_crime_data.csv")
df = df.dropna(subset=["Last outcome category"])

ACTION_TAKEN = {
    'Offender sent to prison', 'Offender given a caution',
    'Awaiting court outcome', 'Court result unavailable',
    'Offender given community sentence', 'Offender given suspended prison sentence',
    'Offender given penalty notice', 'Offender given a drugs possession warning',
    'Offender given conditional discharge', 'Offender ordered to pay compensation',
    'Offender deprived of property', 'Offender otherwise dealt with',
    'Suspect charged as part of another case',
}
df["outcome_binary"] = df["Last outcome category"].apply(
    lambda x: 1 if x in ACTION_TAKEN else 0
)

df["month_num"] = df["Month"].str.split("-").str[1].astype(int)
df["year"]      = df["Month"].str.split("-").str[0].astype(int)
df["season"]    = df["month_num"].map({
    1: 0, 2: 0, 12: 0,
    3: 1, 4: 1,  5: 1,
    6: 2, 7: 2,  8: 2,
    9: 3, 10: 3, 11: 3
})
df["lat_bin"] = df["Latitude"].round(1)
df["lng_bin"] = df["Longitude"].round(1)

crime_dummies = pd.get_dummies(df["Crime type"], prefix="crime", drop_first=True)
feature_cols = ["month_num", "year", "season", "lat_bin", "lng_bin"]
X = pd.concat([df[feature_cols], crime_dummies], axis=1)
y = df["outcome_binary"]

# IMPROVEMENT 5: Stratified split + 5-fold cross-validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=10,
    min_samples_leaf=5, class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=["No Action", "Action Taken"]))

skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1  = cross_val_score(rf, X, y, cv=skf, scoring="f1")
cv_auc = cross_val_score(rf, X, y, cv=skf, scoring="roc_auc")
print(f"5-Fold CV F1     : {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
print(f"5-Fold CV ROC-AUC: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["No Action", "Action Taken"]).plot(
    ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix", fontweight="bold")

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.nlargest(15).sort_values().plot(kind="barh", ax=axes[1], color="#1565C0")
axes[1].set_title("Top 15 Feature Importances", fontweight="bold")
plt.tight_layout()
plt.savefig("fig_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()