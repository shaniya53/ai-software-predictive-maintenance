import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "../data/processed/research_dataset_v1_1.csv"

TRAIN_PATH = "../data/processed/train.csv"
VALIDATION_PATH = "../data/processed/validation.csv"
TEST_PATH = "../data/processed/test.csv"


print("=" * 75)
print("DAY 4 — MODEL DATA PREPARATION")
print("=" * 75)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 2. CONVERT DATE
# ============================================================

print("\nConverting commit dates...")

df["commit_date"] = pd.to_datetime(df["commit_date"], utc=True, errors="coerce")

print("Missing dates:", df["commit_date"].isna().sum())


# ============================================================
# 3. SORT CHRONOLOGICALLY
# ============================================================

print("\nSorting chronologically...")

df = df.sort_values(["commit_date", "PROJECT_ID", "COMMIT_HASH"]).reset_index(drop=True)

print("Chronologically sorted:", df["commit_date"].is_monotonic_increasing)


# ============================================================
# 4. CHECK TARGET
# ============================================================

print("\nTarget distribution:")

print(df["fault_inducing"].value_counts())

print("\nTarget percentages:")

print((df["fault_inducing"].value_counts(normalize=True) * 100).round(2))


# ============================================================
# 5. DEFINE FEATURES
# ============================================================

print("\nDefining model features...")


feature_columns = [
    "lines_added",
    "lines_removed",
    "files_changed",
    "total_lines_changed",
    "avg_lines_per_file",
    "MERGE",
    "IN_MAIN_BRANCH",
    "previous_commit_count",
    "previous_author_commit_count",
    "previous_file_changes",
    "days_since_previous_change",
    "previous_fault_count",
    "has_refactoring",
    "refactoring_count",
    "refactoring_type_count",
]


target_column = "fault_inducing"


print("\nFeatures:")
for feature in feature_columns:
    print("-", feature)

print("\nTarget:")
print("-", target_column)


# ============================================================
# 6. CHECK FEATURE AVAILABILITY
# ============================================================

missing_features = [feature for feature in feature_columns if feature not in df.columns]

if missing_features:

    print("\nERROR: Missing features:")
    print(missing_features)

    raise ValueError("Required features are missing.")


# ============================================================
# 7. TEMPORAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nCreating temporal split...")


# 70% earliest commits → training
# 15% next commits → validation
# 15% latest commits → test

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)


train_df = df.iloc[:train_end].copy()

validation_df = df.iloc[train_end:validation_end].copy()

test_df = df.iloc[validation_end:].copy()


# ============================================================
# 8. SPLIT INFORMATION
# ============================================================

print("\n" + "=" * 75)
print("SPLIT INFORMATION")
print("=" * 75)


for name, dataset in [
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df),
]:

    print(f"\n{name}")

    print("Rows:", len(dataset))

    print(
        "Date range:", dataset["commit_date"].min(), "→", dataset["commit_date"].max()
    )

    print("\nTarget distribution:")

    print(dataset["fault_inducing"].value_counts())

    print("\nTarget percentage:")

    print((dataset["fault_inducing"].value_counts(normalize=True) * 100).round(2))


# ============================================================
# 9. VERIFY TEMPORAL SEPARATION
# ============================================================

print("\n" + "=" * 75)
print("TEMPORAL SEPARATION CHECK")
print("=" * 75)


train_max = train_df["commit_date"].max()
validation_min = validation_df["commit_date"].min()
validation_max = validation_df["commit_date"].max()
test_min = test_df["commit_date"].min()


print("Train latest:", train_max)
print("Validation earliest:", validation_min)
print("Validation latest:", validation_max)
print("Test earliest:", test_min)


print("\nTrain → Validation chronological:", train_max <= validation_min)

print("Validation → Test chronological:", validation_max <= test_min)


# ============================================================
# 10. CREATE ML DATASETS
# ============================================================

print("\nCreating feature matrices...")


X_train = train_df[feature_columns].copy()
y_train = train_df[target_column].copy()

X_validation = validation_df[feature_columns].copy()
y_validation = validation_df[target_column].copy()

X_test = test_df[feature_columns].copy()
y_test = test_df[target_column].copy()


# ============================================================
# 11. APPLY LOG TRANSFORMATION
# ============================================================

print("\nApplying log1p transformation to heavily skewed features...")


log_features = [
    "lines_added",
    "lines_removed",
    "files_changed",
    "total_lines_changed",
    "avg_lines_per_file",
    "previous_file_changes",
    "previous_fault_count",
    "refactoring_count",
    "refactoring_type_count",
]


for feature in log_features:

    X_train[feature] = np.log1p(X_train[feature])

    X_validation[feature] = np.log1p(X_validation[feature])

    X_test[feature] = np.log1p(X_test[feature])


print("Log transformation complete.")


# ============================================================
# 12. CHECK MISSING VALUES
# ============================================================

print("\nChecking prepared datasets...")


print("Training missing values:", X_train.isna().sum().sum())

print("Validation missing values:", X_validation.isna().sum().sum())

print("Test missing values:", X_test.isna().sum().sum())


# ============================================================
# 13. SAVE PREPARED DATASETS
# ============================================================

print("\nSaving datasets...")


output_paths = [
    (X_train.assign(fault_inducing=y_train), TRAIN_PATH),
    (X_validation.assign(fault_inducing=y_validation), VALIDATION_PATH),
    (X_test.assign(fault_inducing=y_test), TEST_PATH),
]


for dataset, path in output_paths:

    output_path = Path(path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(output_path, index=False)

    print("Saved:", output_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("DAY 4 — MODEL DATA PREPARATION COMPLETE")
print("=" * 75)
