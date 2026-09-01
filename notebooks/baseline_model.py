import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

TRAIN_PATH = "../data/processed/train.csv"
VALIDATION_PATH = "../data/processed/validation.csv"
TEST_PATH = "../data/processed/test.csv"


print("=" * 75)
print("DAY 4 — BASELINE MODEL")
print("=" * 75)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading prepared datasets...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
test = pd.read_csv(TEST_PATH)

print("Train:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X_train = train.drop(columns=["fault_inducing"])
y_train = train["fault_inducing"]

X_validation = validation.drop(columns=["fault_inducing"])
y_validation = validation["fault_inducing"]

X_test = test.drop(columns=["fault_inducing"])
y_test = test["fault_inducing"]


# ============================================================
# 3. BASELINE MODEL
# ============================================================

print("\nTraining Dummy Classifier...")

model = DummyClassifier(strategy="most_frequent")

model.fit(X_train, y_train)


# ============================================================
# 4. EVALUATION FUNCTION
# ============================================================


def evaluate(name, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(y, predictions, zero_division=0)

    recall = recall_score(y, predictions, zero_division=0)

    f1 = f1_score(y, predictions, zero_division=0)

    pr_auc = average_precision_score(y, probabilities)

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1       :", round(f1, 4))
    print("PR-AUC   :", round(pr_auc, 4))


# ============================================================
# 5. EVALUATE
# ============================================================

evaluate("TRAIN", X_train, y_train)

evaluate("VALIDATION", X_validation, y_validation)

evaluate("TEST", X_test, y_test)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("BASELINE MODEL COMPLETE")
print("=" * 75)
