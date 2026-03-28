# ==========================================
# OBESITY LEVEL PREDICTION — ALL MODELS
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# ── Load Data ──────────────────────────────────────────────────────────────────
train = pd.read_csv("/content/drive/MyDrive/Data Science Project/train.csv")
test  = pd.read_csv("/content/drive/MyDrive/Data Science Project/test.csv")

TARGET = 'NObeyesdad_enc'

# Convert boolean to int
bool_cols = train.select_dtypes(include="bool").columns
train[bool_cols] = train[bool_cols].astype(int)
test[bool_cols]  = test[bool_cols].astype(int)

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]
X_test  = test.drop(columns=[TARGET])
y_test  = test[TARGET]

# Class label names (used by Decision Tree)
LABEL_MAP = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight L1",
    3: "Overweight L2",
    4: "Obesity T1",
    5: "Obesity T2",
    6: "Obesity T3",
}
CLASS_NAMES = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]


# ==========================================
# 1. DECISION TREE
# ==========================================
dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X_train, y_train)

y_pred   = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cv_score = cross_val_score(dt, X_train, y_train, cv=5).mean()

print("=" * 50)
print("         Decision Tree — Results")
print("=" * 50)
print(f"  Overall Accuracy : {accuracy * 100:.2f}%")
print(f"  Training samples : {len(train)}")
print(f"  Test samples     : {len(test)}")
print("=" * 50)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

print("Test Accuracy:", round(accuracy * 100, 2), "%")
print("CV Accuracy:",  round(cv_score * 100, 2),  "%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9, 7))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(ticks=range(7), labels=CLASS_NAMES, rotation=35, ha="right")
plt.yticks(ticks=range(7), labels=CLASS_NAMES)
plt.colorbar()
plt.tight_layout()
plt.show()

# Feature Importance
importances = pd.Series(dt.feature_importances_, index=X_train.columns)
print("Feature Importances:")
print(importances.sort_values(ascending=False))

importances.sort_values(ascending=False).head(10).plot(kind="bar")
plt.title("Top 10 Feature Importances")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()


# ==========================================
# 2. KNN
# ==========================================
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)

print("=" * 50)
print("           KNN — Results")
print("=" * 50)
print(f"  Overall Accuracy : {accuracy:.2%}")
print(f"  k (neighbours)   : 5")
print(f"  Training samples : {len(train)}")
print(f"  Test samples     : {len(test)}")
print("=" * 50)
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))


# ==========================================
# 3. LOGISTIC REGRESSION
# ==========================================
model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
test_accuracy    = accuracy_score(y_test, test_predictions)

print("=" * 55)
print("     LOGISTIC REGRESSION — Final Test (test.csv)")
print("=" * 55)
print(f"  Overall Accuracy  : {test_accuracy:.2%}")
print(f"  solver            : lbfgs")
print(f"  max_iter          : 1000")
print(f"  Training samples  : {len(X_train)}")
print(f"  Test samples      : {len(X_test)}")
print("=" * 55)
print("\nClassification Report (test.csv):\n")
print(classification_report(y_test, test_predictions))


# ==========================================
# 4. RANDOM FOREST
# ==========================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy    = accuracy_score(y_test, predictions)

print("=" * 50)
print("        RANDOM FOREST — Results")
print("=" * 50)
print(f"  Overall Accuracy : {accuracy:.2%}")
print(f"  n_estimators     : 100")
print(f"  Training samples : {len(train)}")
print(f"  Test samples     : {len(test)}")
print("=" * 50)
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))
# Feature Importance (Random Forest)
importances = pd.Series(model.feature_importances_, index=X_train.columns)

print("Feature Importances (Random Forest):")
print(importances.sort_values(ascending=False))

# Plot Top 10 (optional)
importances.sort_values(ascending=False).head(10).plot(kind="bar")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# ==========================================
# 5. SVM
# ==========================================
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
test_accuracy    = accuracy_score(y_test, test_predictions)

print("=" * 55)
print("           SVM — Final Test (test.csv)")
print("=" * 55)
print(f"  Overall Accuracy  : {test_accuracy:.2%}")
print(f"  kernel            : RBF")
print(f"  C                 : 1.0")
print(f"  gamma             : scale")
print(f"  Training samples  : {len(X_train)}")
print(f"  Test samples      : {len(X_test)}")
print("=" * 55)
print("\nClassification Report (test.csv):\n")
print(classification_report(y_test, test_predictions))