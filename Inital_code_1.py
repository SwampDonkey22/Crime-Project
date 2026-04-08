import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("uk_crime_data.csv")
df = df.dropna(subset=["Last outcome category"])

# IMPROVEMENT 1: Collapse 13 categories into binary target
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

le = LabelEncoder()
df["crime_type_encoded"] = le.fit_transform(df["Crime type"])

X = df[["crime_type_encoded"]]
y = df["outcome_binary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["No Action", "Action Taken"]))