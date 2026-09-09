import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "../data/processed/train.csv"
VALIDATION_PATH = "../data/processed/validation.csv"
TEST_PATH = "../data/processed/test.csv"


print("=" * 80)
print("DAY 4 — MODEL COMPARISON")
print("=" * 80)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
test = pd.read_csv(TEST_PATH)

X_train = train.drop(columns=["fault_inducing"])
y_train = train["fault_inducing"]

X_validation = validation.drop(columns=["fault_inducing"])
y_validation = validation["fault_inducing"]

X_test = test.drop(columns=["fault_inducing"])
y_test = test["fault_inducing"]

print("Train:", X_train.shape)
print("Validation:", X_validation.shape)
print("Test:", X_test.shape)


# ============================================================
# 2. DEFINE MODELS
# ============================================================

models = {
    "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=2000, random_state=42
                ),
            ),
        ]
    ),
    "Random Forest Baseline": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Random Forest Tuned": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=5,
        max_features=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}


# ============================================================
# 3. EVALUATION FUNCTION
# ============================================================


def evaluate_model(model_name, model, X, y, split_name):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(y, predictions, zero_division=0)

    recall = recall_score(y, predictions, zero_division=0)

    f1 = f1_score(y, predictions, zero_division=0)

    pr_auc = average_precision_score(y, probabilities)

    roc_auc = roc_auc_score(y, probabilities)

    return {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


# ============================================================
# 4. TRAIN AND EVALUATE
# ============================================================

results = []

for model_name, model in models.items():

    print("\n" + "=" * 80)
    print("Training:", model_name)
    print("=" * 80)

    model.fit(X_train, y_train)

    print("Training complete.")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    results.append(evaluate_model(model_name, model, X_train, y_train, "Train"))

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    results.append(
        evaluate_model(model_name, model, X_validation, y_validation, "Validation")
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    results.append(evaluate_model(model_name, model, X_test, y_test, "Test"))


# ============================================================
# 5. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("COMPLETE MODEL COMPARISON")
print("=" * 80)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# 6. TEST-SET COMPARISON
# ============================================================

test_results = results_df[results_df["split"] == "Test"].copy()

test_results = test_results.sort_values("pr_auc", ascending=False)

print("\n" + "=" * 80)
print("TEST-SET MODEL RANKING BY PR-AUC")
print("=" * 80)

print(test_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# 7. VALIDATION-SET COMPARISON
# ============================================================

validation_results = results_df[results_df["split"] == "Validation"].copy()

validation_results = validation_results.sort_values("pr_auc", ascending=False)

print("\n" + "=" * 80)
print("VALIDATION MODEL RANKING BY PR-AUC")
print("=" * 80)

print(validation_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# 8. SAVE RESULTS
# ============================================================

output_path = "../data/processed/model_comparison.csv"

results_df.to_csv(output_path, index=False)

print("\nResults saved to:")
print(output_path)


# ============================================================
# 9. COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON COMPLETE")
print("=" * 80)
