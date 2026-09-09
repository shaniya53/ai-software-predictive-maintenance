import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# 1. Load processed datasets
# ============================================================

train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")
test = pd.read_csv("data/processed/test.csv")

TARGET = "fault_inducing"

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_val = validation.drop(columns=[TARGET])
y_val = validation[TARGET]

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]


# ============================================================
# 2. Train Logistic Regression
# ============================================================

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "logistic",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_train, y_train)


# ============================================================
# 3. Get validation probabilities
# ============================================================

val_probabilities = model.predict_proba(X_val)[:, 1]

val_pr_auc = average_precision_score(y_val, val_probabilities)
val_roc_auc = roc_auc_score(y_val, val_probabilities)

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION THRESHOLD ANALYSIS")
print("=" * 70)

print(f"\nValidation PR-AUC : {val_pr_auc:.4f}")
print(f"Validation ROC-AUC: {val_roc_auc:.4f}")


# ============================================================
# 4. Evaluate different thresholds on validation set
# ============================================================

thresholds = np.arange(0.10, 0.51, 0.05)

results = []

print("\nValidation threshold results:")
print("-" * 70)
print(f"{'Threshold':<12}" f"{'Precision':<12}" f"{'Recall':<12}" f"{'F1':<12}")

for threshold in thresholds:

    val_predictions = (val_probabilities >= threshold).astype(int)

    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    print(f"{threshold:<12.2f}" f"{precision:<12.4f}" f"{recall:<12.4f}" f"{f1:<12.4f}")


# ============================================================
# 5. Select threshold using validation F1 only
# ============================================================

results_df = pd.DataFrame(results)

best_row = results_df.loc[results_df["f1"].idxmax()]

best_threshold = float(best_row["threshold"])

print("\n" + "=" * 70)
print("SELECTED THRESHOLD")
print("=" * 70)

print(f"Best validation threshold : {best_threshold:.2f}")
print(f"Validation precision      : {best_row['precision']:.4f}")
print(f"Validation recall         : {best_row['recall']:.4f}")
print(f"Validation F1             : {best_row['f1']:.4f}")


# ============================================================
# 6. Apply frozen threshold to TEST SET
# ============================================================

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (test_probabilities >= best_threshold).astype(int)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_pr_auc = average_precision_score(
    y_test,
    test_probabilities,
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)

cm = confusion_matrix(
    y_test,
    test_predictions,
)


# ============================================================
# 7. Print test results
# ============================================================

print("\n" + "=" * 70)
print("TEST RESULTS USING FROZEN VALIDATION THRESHOLD")
print("=" * 70)

print(f"Threshold : {best_threshold:.2f}")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"F1        : {test_f1:.4f}")
print(f"PR-AUC    : {test_pr_auc:.4f}")
print(f"ROC-AUC   : {test_roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 8. Save threshold results
# ============================================================

results_df.to_csv(
    "data/processed/logistic_threshold_results.csv",
    index=False,
)

print("\nSaved threshold results to " "data/processed/logistic_threshold_results.csv")
