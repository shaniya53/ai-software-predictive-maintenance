import pandas as pd

DATA_PATH = "../data/processed/research_dataset_v1_1.csv"

print("Loading V1.1 dataset...")
df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 70)
print("1. BASIC DATASET CHECK")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate rows:", df.duplicated().sum())

duplicate_commits = df.duplicated(subset=["PROJECT_ID", "COMMIT_HASH"]).sum()

print("Duplicate project + commit combinations:", duplicate_commits)


print("\n" + "=" * 70)
print("2. MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()
print(missing[missing > 0])

if missing.sum() == 0:
    print("No missing values.")


print("\n" + "=" * 70)
print("3. TARGET CHECK")
print("=" * 70)

print(df["fault_inducing"].value_counts())

print("\nExpected:")
print("0 -> 136668")
print("1 -> 17326")


print("\n" + "=" * 70)
print("4. NEGATIVE VALUE CHECK")
print("=" * 70)

numeric_features = [
    "lines_added",
    "lines_removed",
    "files_changed",
    "total_lines_changed",
    "avg_lines_per_file",
    "previous_commit_count",
    "previous_author_commit_count",
    "previous_file_changes",
    "previous_fault_count",
    "refactoring_count",
    "refactoring_type_count",
]

for col in numeric_features:
    negative_count = (df[col] < 0).sum()
    print(f"{col}: {negative_count} negative values")


print("\n" + "=" * 70)
print("5. FEATURE SUMMARY")
print("=" * 70)

print(df[numeric_features].describe().T.to_string())


print("\n" + "=" * 70)
print("6. MERGE / MAIN BRANCH VALUES")
print("=" * 70)

print("MERGE:")
print(df["MERGE"].value_counts(dropna=False))

print("\nIN_MAIN_BRANCH:")
print(df["IN_MAIN_BRANCH"].value_counts(dropna=False))


print("\n" + "=" * 70)
print("7. REFACTORING CHECK")
print("=" * 70)

print("Commits with refactoring:", (df["has_refactoring"] == 1).sum())

print("Commits without refactoring:", (df["has_refactoring"] == 0).sum())

print(
    "Refactoring count > 0 but has_refactoring = 0:",
    ((df["refactoring_count"] > 0) & (df["has_refactoring"] == 0)).sum(),
)


print("\n" + "=" * 70)
print("8. HISTORICAL FEATURE CHECK")
print("=" * 70)

print("previous_fault_count < 0:", (df["previous_fault_count"] < 0).sum())

print("previous_file_changes < 0:", (df["previous_file_changes"] < 0).sum())

print("previous_commit_count < 0:", (df["previous_commit_count"] < 0).sum())

print(
    "previous_author_commit_count < 0:", (df["previous_author_commit_count"] < 0).sum()
)


print("\n" + "=" * 70)
print("9. TIME FEATURE CHECK")
print("=" * 70)

print("days_since_previous_change < -1:", (df["days_since_previous_change"] < -1).sum())

print("First-change records (-1):", (df["days_since_previous_change"] == -1).sum())

print("Normal positive intervals:", (df["days_since_previous_change"] >= 0).sum())


print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
