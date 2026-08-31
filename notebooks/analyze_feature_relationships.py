import pandas as pd
import numpy as np

DATA_PATH = "../data/processed/research_dataset_v1_1.csv"

print("=" * 75)
print("DAY 3 — FEATURE RELATIONSHIP & SELECTION ANALYSIS")
print("=" * 75)

df = pd.read_csv(DATA_PATH)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 1. FEATURES USED FOR ANALYSIS
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
    "has_refactoring",
    "refactoring_count",
    "refactoring_type_count",
]

print("\n" + "=" * 75)
print("1. NUMERIC FEATURE CORRELATION")
print("=" * 75)

correlation = df[numeric_features].corr()

print(correlation.round(3).to_string())


# ============================================================
# 2. HIGHLY CORRELATED FEATURE PAIRS
# ============================================================

print("\n" + "=" * 75)
print("2. HIGHLY CORRELATED FEATURE PAIRS")
print("=" * 75)

threshold = 0.80

pairs = []

for i in range(len(correlation.columns)):
    for j in range(i + 1, len(correlation.columns)):

        feature_1 = correlation.columns[i]
        feature_2 = correlation.columns[j]

        value = correlation.iloc[i, j]

        if abs(value) >= threshold:
            pairs.append([feature_1, feature_2, value])

if pairs:
    high_corr = pd.DataFrame(
        pairs, columns=["feature_1", "feature_2", "correlation"]
    ).sort_values("correlation", key=lambda x: abs(x), ascending=False)

    print(high_corr.round(3).to_string(index=False))

else:
    print("No feature pairs have absolute correlation >= 0.80.")


# ============================================================
# 3. CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 75)
print("3. FEATURE CORRELATION WITH FAULT TARGET")
print("=" * 75)

target_correlations = (
    df[numeric_features + ["fault_inducing"]]
    .corr()["fault_inducing"]
    .drop("fault_inducing")
    .sort_values(key=lambda x: abs(x), ascending=False)
)

print(target_correlations.round(4).to_string())


# ============================================================
# 4. FAULT RATE BY REFACTORING PRESENCE
# ============================================================

print("\n" + "=" * 75)
print("4. REFACTORING VS FAULT RATE")
print("=" * 75)

refactoring_analysis = df.groupby("has_refactoring")["fault_inducing"].agg(
    commits="count", fault_commits="sum", fault_rate="mean"
)

refactoring_analysis["fault_rate"] *= 100

print(refactoring_analysis.round(3).to_string())


# ============================================================
# 5. FAULT RATE BY COMMIT SIZE
# ============================================================

print("\n" + "=" * 75)
print("5. FAULT RATE BY TOTAL LINES CHANGED")
print("=" * 75)

df["change_size_group"] = pd.qcut(df["total_lines_changed"], q=5, duplicates="drop")

size_analysis = df.groupby("change_size_group", observed=True)["fault_inducing"].agg(
    commits="count", fault_commits="sum", fault_rate="mean"
)

size_analysis["fault_rate"] *= 100

print(size_analysis.round(3).to_string())


# ============================================================
# 6. FAULT RATE BY PREVIOUS FAULT HISTORY
# ============================================================

print("\n" + "=" * 75)
print("6. FAULT RATE BY PREVIOUS FAULT HISTORY")
print("=" * 75)

df["fault_history_group"] = pd.qcut(df["previous_fault_count"], q=5, duplicates="drop")

history_analysis = df.groupby("fault_history_group", observed=True)[
    "fault_inducing"
].agg(commits="count", fault_commits="sum", fault_rate="mean")

history_analysis["fault_rate"] *= 100

print(history_analysis.round(3).to_string())


# ============================================================
# 7. FEATURE ZERO-RATE
# ============================================================

print("\n" + "=" * 75)
print("7. ZERO-VALUE RATE")
print("=" * 75)

zero_rate = (df[numeric_features] == 0).mean().sort_values(ascending=False) * 100

print(zero_rate.round(2).to_string())


# ============================================================
# 8. INITIAL FEATURE ASSESSMENT
# ============================================================

print("\n" + "=" * 75)
print("8. INITIAL FEATURE ASSESSMENT")
print("=" * 75)

for feature in numeric_features:

    unique_values = df[feature].nunique()
    zero_percentage = (df[feature] == 0).mean() * 100

    print(
        f"{feature:<35} "
        f"unique={unique_values:<8} "
        f"zero_rate={zero_percentage:>6.2f}%"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("DAY 3 — PART 2 ANALYSIS COMPLETE")
print("=" * 75)
