import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("uk_crime_data.csv")

# Drop rows with no outcome
df = df.dropna(subset=["Last outcome category"])

# Encode crime type as integer
le = LabelEncoder()
df["crime_type_encoded"] = le.fit_transform(df["Crime type"])

# Features and target
X = df[["crime_type_encoded"]]
y = df["Last outcome category"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree (default settings)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")