import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib
df = pd.read_csv("csvbyme.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())
module_encoder = LabelEncoder()
bug_encoder = LabelEncoder()
crash_encoder = LabelEncoder()
repro_encoder = LabelEncoder()
device_encoder = LabelEncoder()
severity_encoder = LabelEncoder()
df["Module"] = module_encoder.fit_transform(df["Module"])
df["Bug_Type"] = bug_encoder.fit_transform(df["Bug_Type"])
df["Crash"] = crash_encoder.fit_transform(df["Crash"])
df["Reproducible"] = repro_encoder.fit_transform(df["Reproducible"])
df["Device"] = device_encoder.fit_transform(df["Device"])
df["Severity"] = severity_encoder.fit_transform(df["Severity"])
X = df[
    [
        "Module",
        "Bug_Type",
        "Crash",
        "Affected_Users",
        "Reports_Per_Day",
        "Reproducible",
        "Device"
    ]
]

y = df["Severity"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 15, None]
}
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, y_train)
print("Best Parameters :", grid.best_params_)
print("Best Estimator :", grid.best_estimator_)
y_pred = grid.predict(X_test)
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(
    y_test,
    y_pred,
    average="weighted"
))
print("Recall :", recall_score(
    y_test,
    y_pred,
    average="weighted"
))
print("F1 Score :", f1_score(
    y_test,
    y_pred,
    average="weighted"
))
joblib.dump(
    grid.best_estimator_,
    "bug_severity_model.pkl"
)
joblib.dump(module_encoder, "module_encoder.pkl")
joblib.dump(bug_encoder, "bug_encoder.pkl")
joblib.dump(crash_encoder, "crash_encoder.pkl")
joblib.dump(repro_encoder, "repro_encoder.pkl")
joblib.dump(device_encoder, "device_encoder.pkl")
joblib.dump(severity_encoder, "severity_encoder.pkl")
print("\nModel Saved Successfully")
print("\nAll Encoders Saved Successfully")