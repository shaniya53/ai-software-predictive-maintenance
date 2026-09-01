import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "../data/processed/train.csv"
VALIDATION_PATH = "../data/processed/validation.csv"
TEST_PATH = "../data/processed/test.csv"


print("=" * 75)
print("DAY 4 — LOGISTIC REGRESSION")
print("=" * 75)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading datasets...")

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
# 3. LOGISTIC REGRESSION PIPELINE
# ============================================================

print("\nCreating Logistic Regression pipeline...")

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        ),
    ]
)


# ============================================================
# 4. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 5. EVALUATION FUNCTION
# ============================================================


def evaluate(name, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(y, predictions, zero_division=0)

    recall = recall_score(y, predictions, zero_division=0)

    f1 = f1_score(y, predictions, zero_division=0)

    pr_auc = average_precision_score(y, probabilities)

    roc_auc = roc_auc_score(y, probabilities)

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1       :", round(f1, 4))
    print("PR-AUC   :", round(pr_auc, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))

    print("\nClassification Report:")
    print(classification_report(y, predictions, zero_division=0))


# ============================================================
# 6. EVALUATE TRAINING SET
# ============================================================

evaluate("TRAIN", X_train, y_train)


# ============================================================
# 7. EVALUATE VALIDATION SET
# ============================================================

evaluate("VALIDATION", X_validation, y_validation)


# ============================================================
# 8. EVALUATE TEST SET
# ============================================================

evaluate("TEST", X_test, y_test)


# ============================================================
# 9. FEATURE COEFFICIENTS
# ============================================================

print("\n" + "=" * 75)
print("LOGISTIC REGRESSION FEATURE COEFFICIENTS")
print("=" * 75)


classifier = model.named_steps["classifier"]


coefficients = pd.DataFrame(
    {"feature": X_train.columns, "coefficient": classifier.coef_[0]}
)


coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()


coefficients = coefficients.sort_values("absolute_coefficient", ascending=False)


print(coefficients.to_string(index=False))


# ============================================================
# 10. COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("LOGISTIC REGRESSION COMPLETE")
print("=" * 75)
