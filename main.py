import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv("data.csv", sep="\t")

print("Dataset Shape:", data.shape)

print("\nFirst 10 Rows:")
print(data.head(10))

print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

# ==========================================
# DROP UNUSED COLUMNS
# ==========================================

data = data.drop(["Name", "Ticket", "Cabin"], axis=1)

# ==========================================
# HANDLE MISSING VALUES (RECOMMENDED FIX)
# ==========================================

data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
data["Age"] = data["Age"].fillna(data["Age"].median())

# ==========================================
# ENCODING (RECOMMENDED SAFE WAY)
# ==========================================

label_encoders = {}

for col in ["Sex", "Embarked"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# ==========================================
# FEATURES & TARGET
# ==========================================

X = data.drop("Survived", axis=1)
y = data["Survived"]

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# MODEL (IMPROVED VERSION)
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,   # improved from 100
    max_depth=10,       # prevents overfitting
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# FEATURE IMPORTANCE (RECOMMENDED)
# ==========================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(feature_importance)