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
print("DAY 4 — RANDOM FOREST HYPERPARAMETER TUNING")
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
# 3. HYPERPARAMETER CONFIGURATIONS
# ============================================================

configs = [
    {
        "name": "Baseline",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "name": "Shallow Trees",
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "name": "Medium Trees",
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "name": "Larger Leaf",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },
    {
        "name": "More Conservative",
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    },
    {
        "name": "More Features",
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 5,
        "max_features": None,
    },
]


# ============================================================
# 4. TRAIN AND EVALUATE CONFIGURATIONS
# ============================================================

results = []

for config in configs:

    print("\n" + "=" * 75)
    print("Testing:", config["name"])
    print("=" * 75)

    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("Training...")

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    probabilities = model.predict_proba(X_validation)[:, 1]

    # Use default threshold 0.5 for model comparison
    predictions = (probabilities >= 0.5).astype(int)

    precision = precision_score(y_validation, predictions, zero_division=0)

    recall = recall_score(y_validation, predictions, zero_division=0)

    f1 = f1_score(y_validation, predictions, zero_division=0)

    pr_auc = average_precision_score(y_validation, probabilities)

    roc_auc = roc_auc_score(y_validation, probabilities)

    print("Validation Precision:", round(precision, 4))
    print("Validation Recall   :", round(recall, 4))
    print("Validation F1       :", round(f1, 4))
    print("Validation PR-AUC   :", round(pr_auc, 4))
    print("Validation ROC-AUC  :", round(roc_auc, 4))

    results.append(
        {
            "configuration": config["name"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
        }
    )


# ============================================================
# 5. COMPARE CONFIGURATIONS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("VALIDATION RESULTS")
print("=" * 75)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# 6. SELECT BEST CONFIGURATION
# ============================================================

best_row = results_df.loc[results_df["pr_auc"].idxmax()]

best_name = best_row["configuration"]

print("\n" + "=" * 75)
print("BEST CONFIGURATION")
print("=" * 75)

print("Configuration:", best_name)
print("PR-AUC       :", round(best_row["pr_auc"], 4))
print("ROC-AUC      :", round(best_row["roc_auc"], 4))
print("F1           :", round(best_row["f1"], 4))
print("Precision    :", round(best_row["precision"], 4))
print("Recall       :", round(best_row["recall"], 4))


# ============================================================
# 7. RETRAIN BEST MODEL
# ============================================================

best_config = next(config for config in configs if config["name"] == best_name)

print("\n" + "=" * 75)
print("RETRAINING BEST MODEL")
print("=" * 75)

best_model = RandomForestClassifier(
    n_estimators=best_config["n_estimators"],
    max_depth=best_config["max_depth"],
    min_samples_split=best_config["min_samples_split"],
    min_samples_leaf=best_config["min_samples_leaf"],
    max_features=best_config["max_features"],
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

best_model.fit(X_train, y_train)

print("Best model trained.")


# ============================================================
# 8. TEST EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_predictions = (test_probabilities >= 0.5).astype(int)

test_precision = precision_score(y_test, test_predictions, zero_division=0)

test_recall = recall_score(y_test, test_predictions, zero_division=0)

test_f1 = f1_score(y_test, test_predictions, zero_division=0)

test_pr_auc = average_precision_score(y_test, test_probabilities)

test_roc_auc = roc_auc_score(y_test, test_probabilities)

print("Precision:", round(test_precision, 4))
print("Recall   :", round(test_recall, 4))
print("F1       :", round(test_f1, 4))
print("PR-AUC   :", round(test_pr_auc, 4))
print("ROC-AUC  :", round(test_roc_auc, 4))


# ============================================================
# 9. COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("RANDOM FOREST TUNING COMPLETE")
print("=" * 75)
