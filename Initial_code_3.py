import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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

# IMPROVEMENT 3: Extract temporal and geographic features
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["No Action", "Action Taken"]))