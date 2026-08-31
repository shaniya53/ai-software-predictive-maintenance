import pandas as pd
import numpy as np

DATA_PATH = "../data/processed/research_dataset_v1_1.csv"

print("=" * 75)
print("DAY 3 — FEATURE & OUTLIER ANALYSIS")
print("=" * 75)

df = pd.read_csv(DATA_PATH)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 1. FEATURE TYPES
# ============================================================

print("\n" + "=" * 75)
print("1. FEATURE TYPES")
print("=" * 75)

for col in df.columns:
    print(f"{col:<35} {df[col].dtype}")


# ============================================================
# 2. NUMERIC FEATURE SUMMARY
# ============================================================

numeric_features = [
    "lines_added",
    "lines_removed",
    "files_changed",
    "total_lines_changed",
    "avg_lines_per_file",
    "previous_commit_count",
    "previous_author_commit_count",
    "previous_file_changes",
    "days_since_previous_change",
    "previous_fault_count",
    "refactoring_count",
    "refactoring_type_count",
]

print("\n" + "=" * 75)
print("2. NUMERIC FEATURE SUMMARY")
print("=" * 75)

summary = df[numeric_features].describe().T

summary["skewness"] = df[numeric_features].skew()

print(
    summary[
        [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "skewness",
        ]
    ]
    .round(3)
    .to_string()
)


# ============================================================
# 3. OUTLIER COUNTS USING IQR
# ============================================================

print("\n" + "=" * 75)
print("3. OUTLIER ANALYSIS — IQR METHOD")
print("=" * 75)

outlier_results = []

for col in numeric_features:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((df[col] < lower) | (df[col] > upper)).sum()

    percentage = (outliers / len(df)) * 100

    outlier_results.append(
        [
            col,
            q1,
            q3,
            lower,
            upper,
            outliers,
            percentage,
        ]
    )

outlier_df = pd.DataFrame(
    outlier_results,
    columns=[
        "feature",
        "Q1",
        "Q3",
        "lower_bound",
        "upper_bound",
        "outlier_count",
        "outlier_percentage",
    ],
)

print(outlier_df.round(3).to_string(index=False))


# ============================================================
# 4. TOP EXTREME VALUES
# ============================================================

print("\n" + "=" * 75)
print("4. TOP 10 EXTREME VALUES")
print("=" * 75)

for col in numeric_features:

    print(f"\n--- {col} ---")

    top_values = (
        df[["PROJECT_ID", "COMMIT_HASH", "commit_date", col]]
        .sort_values(col, ascending=False)
        .head(10)
    )

    print(top_values.to_string(index=False))


# ============================================================
# 5. FAULT VS NON-FAULT COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("5. FAULT VS NON-FAULT FEATURE COMPARISON")
print("=" * 75)

comparison = df.groupby("fault_inducing")[numeric_features].median().T

comparison.columns = ["non_fault_median", "fault_median"]

comparison["difference"] = comparison["fault_median"] - comparison["non_fault_median"]

comparison["ratio"] = np.where(
    comparison["non_fault_median"] != 0,
    comparison["fault_median"] / comparison["non_fault_median"],
    np.nan,
)

print(comparison.round(3).to_string())


# ============================================================
# 6. FAULT RATE BY FEATURE PRESENCE
# ============================================================

print("\n" + "=" * 75)
print("6. FAULT RATE BY REFACTORING")
print("=" * 75)

refactoring_fault_rate = df.groupby("has_refactoring")["fault_inducing"].agg(
    ["count", "sum", "mean"]
)

refactoring_fault_rate["mean"] *= 100

print(refactoring_fault_rate.round(3).to_string())


# ============================================================
# 7. FAULT RATE FOR MERGE COMMITS
# ============================================================

print("\n" + "=" * 75)
print("7. FAULT RATE BY MERGE STATUS")
print("=" * 75)

merge_fault_rate = df.groupby("MERGE")["fault_inducing"].agg(["count", "sum", "mean"])

merge_fault_rate["mean"] *= 100

print(merge_fault_rate.round(3).to_string())


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 75)
print("FEATURE & OUTLIER ANALYSIS — PART 1 COMPLETE")
print("=" * 75)
