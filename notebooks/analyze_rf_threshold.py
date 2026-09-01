import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
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


print("=" * 75)
print("DAY 4 — RANDOM FOREST THRESHOLD ANALYSIS")
print("=" * 75)


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
# 2. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 3. GET VALIDATION PROBABILITIES
# ============================================================

print("\nGenerating validation probabilities...")

validation_probabilities = model.predict_proba(X_validation)[:, 1]

validation_pr_auc = average_precision_score(y_validation, validation_probabilities)

validation_roc_auc = roc_auc_score(y_validation, validation_probabilities)

print("Validation PR-AUC :", round(validation_pr_auc, 4))
print("Validation ROC-AUC:", round(validation_roc_auc, 4))


# ============================================================
# 4. THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 75)

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

results = []

for threshold in thresholds:

    predictions = (validation_probabilities >= threshold).astype(int)

    precision = precision_score(y_validation, predictions, zero_division=0)

    recall = recall_score(y_validation, predictions, zero_division=0)

    f1 = f1_score(y_validation, predictions, zero_division=0)

    results.append(
        {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1}
    )


results_df = pd.DataFrame(results)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# 5. SELECT BEST THRESHOLD USING VALIDATION F1
# ============================================================

best_row = results_df.loc[results_df["f1"].idxmax()]

best_threshold = best_row["threshold"]

print("\n" + "=" * 75)
print("BEST VALIDATION THRESHOLD")
print("=" * 75)

print("Threshold :", best_threshold)
print("Precision :", round(best_row["precision"], 4))
print("Recall    :", round(best_row["recall"], 4))
print("F1        :", round(best_row["f1"], 4))


# ============================================================
# 6. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("TEST EVALUATION USING SELECTED THRESHOLD")
print("=" * 75)

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (test_probabilities >= best_threshold).astype(int)

test_precision = precision_score(y_test, test_predictions, zero_division=0)

test_recall = recall_score(y_test, test_predictions, zero_division=0)

test_f1 = f1_score(y_test, test_predictions, zero_division=0)

test_pr_auc = average_precision_score(y_test, test_probabilities)

test_roc_auc = roc_auc_score(y_test, test_probabilities)

print("Threshold :", best_threshold)
print("Precision :", round(test_precision, 4))
print("Recall    :", round(test_recall, 4))
print("F1        :", round(test_f1, 4))
print("PR-AUC    :", round(test_pr_auc, 4))
print("ROC-AUC   :", round(test_roc_auc, 4))


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("RANDOM FOREST THRESHOLD ANALYSIS COMPLETE")
print("=" * 75)
