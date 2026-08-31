import pandas as pd

DATA_PATH = "../data/processed/research_dataset_v1_1.csv"

print("=" * 70)
print("TEMPORAL FEATURE VALIDATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

df["commit_date"] = pd.to_datetime(df["commit_date"], utc=True, errors="coerce")

print("Rows:", len(df))
print("Missing dates:", df["commit_date"].isna().sum())


# ============================================================
# 1. GLOBAL DATE ORDER
# ============================================================

print("\n" + "=" * 70)
print("1. GLOBAL DATE ORDER")
print("=" * 70)

is_sorted = df["commit_date"].is_monotonic_increasing

print("Dataset sorted by commit date:", is_sorted)

if not is_sorted:
    print("WARNING: Dataset is not globally chronological.")


# ============================================================
# 2. DATE RANGE
# ============================================================

print("\n" + "=" * 70)
print("2. DATE RANGE")
print("=" * 70)

print("Earliest commit:", df["commit_date"].min())
print("Latest commit:", df["commit_date"].max())


# ============================================================
# 3. DAYS SINCE PREVIOUS CHANGE
# ============================================================

print("\n" + "=" * 70)
print("3. DAYS SINCE PREVIOUS CHANGE")
print("=" * 70)

print(df["days_since_previous_change"].describe().to_string())

print("Negative values below -1:", (df["days_since_previous_change"] < -1).sum())

print("First-change records (-1):", (df["days_since_previous_change"] == -1).sum())


# ============================================================
# 4. HISTORICAL FEATURE SANITY
# ============================================================

print("\n" + "=" * 70)
print("4. HISTORICAL FEATURE SANITY")
print("=" * 70)

historical_features = [
    "previous_commit_count",
    "previous_author_commit_count",
    "previous_file_changes",
    "previous_fault_count",
]

for col in historical_features:

    print(f"{col:<35}", "min =", df[col].min(), "| zero =", (df[col] == 0).sum())


# ============================================================
# 5. FIRST COMMIT / NO HISTORY CHECK
# ============================================================

print("\n" + "=" * 70)
print("5. NO-HISTORY RECORDS")
print("=" * 70)

no_history = df[df["previous_commit_count"] == 0]

print("Commits with no previous commit history:", len(no_history))

print(
    "Their previous_fault_count values:",
    no_history["previous_fault_count"].value_counts().to_dict(),
)


# ============================================================
# 6. CHECK HISTORICAL FAULT COUNT
# ============================================================

print("\n" + "=" * 70)
print("6. PREVIOUS FAULT COUNT")
print("=" * 70)

print(df["previous_fault_count"].describe().to_string())

print("Commits with previous_fault_count = 0:", (df["previous_fault_count"] == 0).sum())


# ============================================================
# 7. FEATURE RELATIONSHIPS
# ============================================================

print("\n" + "=" * 70)
print("7. BASIC TEMPORAL RELATIONSHIPS")
print("=" * 70)

# A commit cannot have more previous author commits
# than previous total commits.

invalid_author_history = (
    df["previous_author_commit_count"] > df["previous_commit_count"]
)

print("Author history > total history:", invalid_author_history.sum())


# If there is no previous commit history,
# previous file changes should normally be zero.

invalid_file_history = (df["previous_commit_count"] == 0) & (
    df["previous_file_changes"] != 0
)

print("No previous commits but previous_file_changes != 0:", invalid_file_history.sum())


# ============================================================
# 8. INVESTIGATE SUSPICIOUS HISTORY RECORDS
# ============================================================

print("\n" + "=" * 70)
print("8. INVESTIGATE SUSPICIOUS HISTORY RECORDS")
print("=" * 70)

suspicious = df[(df["previous_commit_count"] == 0) & (df["previous_file_changes"] != 0)]

print("Suspicious records:", len(suspicious))

if len(suspicious) > 0:

    print("\nSuspicious records:")

    print(
        suspicious[
            [
                "PROJECT_ID",
                "COMMIT_HASH",
                "commit_date",
                "previous_commit_count",
                "previous_file_changes",
                "previous_fault_count",
                "lines_added",
                "lines_removed",
                "files_changed",
            ]
        ].to_string(index=False)
    )

else:
    print("No suspicious records found.")


# ============================================================
# 9. TEMPORAL FEATURE CORRELATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("9. HISTORICAL FEATURE CORRELATION")
print("=" * 70)

print(
    df[
        [
            "previous_commit_count",
            "previous_author_commit_count",
            "previous_file_changes",
            "previous_fault_count",
            "days_since_previous_change",
        ]
    ]
    .corr()
    .round(3)
    .to_string()
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL VALIDATION COMPLETE")
print("=" * 70)
